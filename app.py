import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import os
import time
from datetime import datetime

# ================================
# 1. 核心参数与看板设置
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x 终极双向评分 AI (OKX)", page_icon="⚖️")

SYMBOL = "ETH/USDT:USDT"            # OKX 永续合约
REFRESH_MS = 2500                   # 2.5秒刷新
CIRCUIT_BREAKER_PCT = 0.003         # 0.3% 熔断

# 权重配置
TREND_WEIGHT = 0.5
MOMENTUM_WEIGHT = 0.3
MODEL_WEIGHT = 0.2

# 波动率过滤：ATR百分比 < 0.15% 时禁止交易（原为0.25%）
MIN_ATR_PCT = 0.0015

# 多空信心分最小差值，低于此值不交易
MIN_SCORE_GAP = 10

# 成交量放大倍数要求（原为1.2，现放宽至0.8，但仍需高于均量）
VOLUME_RATIO_MIN = 0.8

# 模型概率方向确认门槛（低于此值即使最终分够也不交易）
MODEL_DIRECTION_MIN = 55  # 55%

# 模型概率差值最小要求（避免模型模糊）
MODEL_GAP_MIN = 5

# 风险收益比（统一为2.0）
RR = 2.0

# 止损距离下限（0.15%），防止过小止损被噪音扫掉
MIN_SL_PCT = 0.0015

# 趋势强度指数阈值（基于原始分数）
MIN_TREND_STRENGTH = 15
STRONG_TREND_THRESH = 35

# 冷却K线数量（至少间隔2根5m K线）
COOLDOWN_CANDLES = 2
CANDLE_5M_MS = 5 * 60 * 1000  # 5分钟对应的毫秒数

# 爆发识别阈值
BREAKOUT_VOL_RATIO = 1.5       # 成交量放大倍数
BREAKOUT_ADX_MIN = 25          # ADX最小值
BREAKOUT_CONF_THRES = 75       # 爆发行情下的信心分门槛
FINAL_CONF_THRES = 80          # 常规信心分门槛

st_autorefresh(interval=REFRESH_MS, key="bidirectional_ai_final")

# ================================
# 2. 初始化交易所和模型
# ================================
@st.cache_resource
def init_system():
    exch = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    # 加载双模型（兼容通用模型）
    m_l = joblib.load("eth_ai_model_long.pkl") if os.path.exists("eth_ai_model_long.pkl") else None
    m_s = joblib.load("eth_ai_model_short.pkl") if os.path.exists("eth_ai_model_short.pkl") else None
    if m_l is None or m_s is None:
        generic = joblib.load("eth_ai_model.pkl") if os.path.exists("eth_ai_model.pkl") else None
        if generic:
            m_l = m_s = generic
            st.sidebar.info("💡 使用通用模型镜像多空")
        else:
            st.sidebar.error("❌ 未找到任何模型文件（eth_ai_model.pkl），请上传模型至应用根目录。")
            st.stop()   # 无模型时直接停止运行
    return exch, m_l, m_s

exchange, model_long, model_short = init_system()

# ================================
# 3. 状态管理
# ================================
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0
if 'system_halted' not in st.session_state:
    st.session_state.system_halted = False
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []  # 历史信号记录（包含盈亏）
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = 0
if 'active_signal' not in st.session_state:
    st.session_state.active_signal = None   # 当前活动信号（持仓）
if 'last_signal_candle' not in st.session_state:
    st.session_state.last_signal_candle = None
if 'position' not in st.session_state:
    st.session_state.position = None  # 持仓信息：{'side','entry','sl','tp','entry_time','score'}
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_trades': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'max_consecutive_losses': 0,
        'current_consecutive_losses': 0,
        'last_update': None
    }

# ================================
# 4. 数据获取函数（多时间框架，处理None）
# ================================
def fetch_ohlcv(timeframe, limit=200):
    return exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)

def get_multi_timeframe_data():
    ohlcv_5m = fetch_ohlcv("5m", 200)
    if not ohlcv_5m:
        st.error("无法获取 5m 数据，请检查网络或交易所状态。")
        st.stop()
    df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    ohlcv_15m = fetch_ohlcv("15m", 100)
    if not ohlcv_15m:
        st.error("无法获取 15m 数据，请检查网络或交易所状态。")
        st.stop()
    df_15m = pd.DataFrame(ohlcv_15m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    ohlcv_1h = fetch_ohlcv("1h", 100)
    if not ohlcv_1h:
        st.error("无法获取 1h 数据，请检查网络或交易所状态。")
        st.stop()
    df_1h = pd.DataFrame(ohlcv_1h, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    for df in [df_5m, df_15m, df_1h]:
        df.replace([None], np.nan, inplace=True)
    
    return df_5m, df_15m, df_1h

# ================================
# 5. 指标计算函数（使用标准列名）
# ================================
def compute_features(df_5m, df_15m, df_1h):
    for df in [df_5m, df_15m, df_1h]:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

    # ----- 5m 指标 -----
    df_5m["rsi"] = ta.rsi(df_5m["close"], length=14)
    df_5m["ma20"] = ta.sma(df_5m["close"], length=20)
    df_5m["ma60"] = ta.sma(df_5m["close"], length=60)
    # 更灵敏的MACD (10,22,8)
    macd = ta.macd(df_5m["close"], fast=10, slow=22, signal=8)
    df_5m["macd"] = macd["MACD_10_22_8"]
    df_5m["macd_signal"] = macd["MACDs_10_22_8"]
    df_5m["atr"] = ta.atr(df_5m["high"], df_5m["low"], df_5m["close"], length=14)
    df_5m["atr_pct"] = df_5m["atr"] / df_5m["close"]
    df_5m["adx"] = ta.adx(df_5m["high"], df_5m["low"], df_5m["close"], length=14)["ADX_14"]
    
    # 动量核（EMA5/EMA20）
    df_5m["ema5"] = ta.ema(df_5m["close"], length=5)
    df_5m["ema20"] = ta.ema(df_5m["close"], length=20)
    vwap = ta.vwap(df_5m["high"], df_5m["low"], df_5m["close"], df_5m["volume"])
    df_5m["VWAP"] = vwap
    df_5m["volume_ma20"] = ta.sma(df_5m["volume"], length=20)
    df_5m["atr_ma20"] = df_5m["atr"].rolling(20).mean()
    df_5m["atr_surge"] = (df_5m["atr"] > df_5m["atr_ma20"] * 1.2).fillna(False)
    
    # 新增特征（用于模型）
    df_5m["volume_ratio"] = df_5m["volume"] / df_5m["volume_ma20"]
    bb = ta.bbands(df_5m["close"], length=20, std=2)
    df_5m["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / df_5m["close"]

    # ----- 15m 指标 -----
    df_15m["ema200"] = ta.ema(df_15m["close"], length=200)
    df_15m["adx"] = ta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)["ADX_14"]
    vwap_15 = ta.vwap(df_15m["high"], df_15m["low"], df_15m["close"], df_15m["volume"])
    df_15m["VWAP"] = vwap_15
    df_15m["hh"] = df_15m["high"].rolling(20).max()
    df_15m["ll"] = df_15m["low"].rolling(20).min()
    df_15m["ema200_slope"] = df_15m["ema200"] - df_15m["ema200"].shift(5)
    
    # ----- 1h 指标 -----
    df_1h["ema200"] = ta.ema(df_1h["close"], length=200)
    df_1h["adx"] = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)["ADX_14"]
    vwap_1h = ta.vwap(df_1h["high"], df_1h["low"], df_1h["close"], df_1h["volume"])
    df_1h["VWAP"] = vwap_1h
    df_1h["hh"] = df_1h["high"].rolling(20).max()
    df_1h["ll"] = df_1h["low"].rolling(20).min()
    df_1h["ema200_slope"] = df_1h["ema200"] - df_1h["ema200"].shift(3)
    
    # 填充NaN
    df_5m = df_5m.ffill().bfill().infer_objects(copy=False)
    df_15m = df_15m.ffill().bfill().infer_objects(copy=False)
    df_1h = df_1h.ffill().bfill().infer_objects(copy=False)
    
    df_5m["atr_surge"] = df_5m["atr_surge"].fillna(False)
    
    # 最新一行特征（用于模型预测）- 注意新增两个特征
    feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx', 'volume_ratio', 'bb_width']
    latest_feat = df_5m[feat_cols].iloc[-1:].copy()
    
    return df_5m, df_15m, df_1h, latest_feat

# ================================
# 6. 评分函数（与之前相同，略去详细注释以节省篇幅）
# ================================
def compute_trend_score(df_15m, df_1h):
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]
    long_score = 0
    short_score = 0

    if pd.notna(c15['close']) and pd.notna(c15['ema200']) and pd.notna(c15['ema200_slope']):
        if c15['close'] > c15['ema200'] and c15['ema200_slope'] > 0:
            long_score += 15
        elif c15['close'] < c15['ema200'] and c15['ema200_slope'] < 0:
            short_score += 15

    if pd.notna(c1h['close']) and pd.notna(c1h['ema200']) and pd.notna(c1h['ema200_slope']):
        if c1h['close'] > c1h['ema200'] and c1h['ema200_slope'] > 0:
            long_score += 15
        elif c1h['close'] < c1h['ema200'] and c1h['ema200_slope'] < 0:
            short_score += 15

    if pd.notna(c15['close']) and pd.notna(c15['VWAP']):
        if c15['close'] > c15['VWAP']:
            long_score += 10
        else:
            short_score += 10

    if pd.notna(c1h['close']) and pd.notna(c1h['VWAP']):
        if c1h['close'] > c1h['VWAP']:
            long_score += 10
        else:
            short_score += 10

    range_15 = c15['hh'] - c15['ll'] if pd.notna(c15['hh']) and pd.notna(c15['ll']) else 0
    if range_15 > 0 and pd.notna(c15['close']):
        if (c15['close'] - c15['ll']) / range_15 > 0.5:
            long_score += 10
        else:
            short_score += 10

    range_1h = c1h['hh'] - c1h['ll'] if pd.notna(c1h['hh']) and pd.notna(c1h['ll']) else 0
    if range_1h > 0 and pd.notna(c1h['close']):
        if (c1h['close'] - c1h['ll']) / range_1h > 0.5:
            long_score += 10
        else:
            short_score += 10

    raw_long = min(long_score, 100)
    raw_short = min(short_score, 100)

    if pd.notna(c15['adx']) and pd.notna(c1h['adx']) and c15['adx'] > 25 and c1h['adx'] > 25:
        long_score = int(long_score * 1.15)
        short_score = int(short_score * 1.15)

    long_score = min(long_score, 100)
    short_score = min(short_score, 100)

    return long_score, short_score, raw_long, raw_short

def compute_momentum_score(df_5m):
    c = df_5m.iloc[-1]
    long_score = 0
    short_score = 0

    if pd.notna(c['ema5']) and pd.notna(c['ema20']):
        if c['ema5'] > c['ema20']:
            long_score += 30
        else:
            short_score += 30

    if pd.notna(c['close']) and pd.notna(c['VWAP']):
        if c['close'] > c['VWAP']:
            long_score += 20
        else:
            short_score += 20

    if pd.notna(c['volume']) and pd.notna(c['volume_ma20']) and c['volume_ma20'] > 0:
        if c['volume'] > c['volume_ma20'] * VOLUME_RATIO_MIN:
            long_score += 25
            short_score += 25

    if pd.notna(c['atr_surge']) and c['atr_surge']:
        if pd.notna(c['ema5']) and pd.notna(c['ema20']) and c['ema5'] > c['ema20']:
            long_score += 25
        else:
            short_score += 25

    return min(long_score, 100), min(short_score, 100)

def compute_model_prob(df_5m, latest_feat, trend_long, trend_short):
    if model_long is None or model_short is None:
        return 50, 50
    
    if latest_feat.isnull().all().any():
        st.sidebar.warning("⚠️ 模型特征全为NaN，使用0填充")
        latest_feat = latest_feat.fillna(0)
    else:
        latest_feat = latest_feat.fillna(0)
    
    try:
        proba_l = model_long.predict_proba(latest_feat)[0]
        proba_s = model_short.predict_proba(latest_feat)[0]
        prob_l = proba_l[1] * 100
        prob_s = proba_s[1] * 100
        
        if prob_l == 0 and prob_s == 0:
            st.sidebar.warning("⚠️ 模型概率均为0，使用趋势核方向作为默认概率")
            if trend_long > trend_short:
                prob_l, prob_s = 60, 40
            elif trend_short > trend_long:
                prob_l, prob_s = 40, 60
            else:
                prob_l, prob_s = 50, 50
        elif prob_l == 0:
            prob_l = 50
        elif prob_s == 0:
            prob_s = 50
    except Exception as e:
        st.sidebar.error(f"模型预测异常: {e}")
        prob_l = prob_s = 50
        
    return prob_l, prob_s

def detect_momentum_decay(df_5m):
    if len(df_5m) < 5:
        return False
    macd_vals = df_5m['macd'].iloc[-5:].values
    if any(pd.isna(v) for v in macd_vals):
        return False
    return (macd_vals[4] < macd_vals[3] and
            macd_vals[3] < macd_vals[2] and
            macd_vals[2] < macd_vals[1] and
            macd_vals[1] < macd_vals[0])

def detect_breakout(df_5m):
    c = df_5m.iloc[-1]
    if pd.isna(c['volume']) or pd.isna(c['volume_ma20']) or c['volume_ma20'] <= 0:
        vol_ratio = 0
    else:
        vol_ratio = c['volume'] / c['volume_ma20']
    atr_surge = pd.notna(c['atr_surge']) and c['atr_surge']
    adx_ok = pd.notna(c['adx']) and c['adx'] > BREAKOUT_ADX_MIN
    return (atr_surge and vol_ratio > BREAKOUT_VOL_RATIO and adx_ok)

# ================================
# 7. 盈亏统计函数
# ================================
def check_position_exit(position, current_price):
    if position is None:
        return None
    side = position['side']
    entry = position['entry']
    sl = position['sl']
    tp = position['tp']
    
    if side == 'LONG':
        if current_price <= sl:
            pnl = (sl - entry) / entry
            return pnl, '止损'
        elif current_price >= tp:
            pnl = (tp - entry) / entry
            return pnl, '止盈'
    else:
        if current_price >= sl:
            pnl = (entry - sl) / entry
            return pnl, '止损'
        elif current_price <= tp:
            pnl = (entry - tp) / entry
            return pnl, '止盈'
    return None

def update_stats(pnl):
    stats = st.session_state.stats
    stats['total_trades'] += 1
    stats['total_pnl'] += pnl * 100
    if pnl > 0:
        stats['wins'] += 1
        stats['current_consecutive_losses'] = 0
    else:
        stats['losses'] += 1
        stats['current_consecutive_losses'] += 1
        if stats['current_consecutive_losses'] > stats['max_consecutive_losses']:
            stats['max_consecutive_losses'] = stats['current_consecutive_losses']
    stats['last_update'] = datetime.now()

# ================================
# 8. 侧边栏（含统计面板）
# ================================
with st.sidebar:
    st.header("📊 实时审计")
    try:
        funding = exchange.fetch_funding_rate(SYMBOL)
        f_rate = funding['fundingRate'] * 100
        st.metric("OKX 资金费率", f"{f_rate:.4f}%", delta="看多成本高" if f_rate > 0.03 else "")
    except:
        st.write("费率加载中...")
    
    st.markdown("---")
    st.subheader("📈 实时统计")
    stats = st.session_state.stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总交易次数", stats['total_trades'])
        win_rate = (stats['wins'] / max(stats['total_trades'], 1)) * 100
        st.metric("胜率", f"{win_rate:.1f}%")
        st.metric("最大连亏", stats['max_consecutive_losses'])
    with col2:
        st.metric("盈利次数", stats['wins'])
        st.metric("亏损次数", stats['losses'])
        st.metric("总盈亏", f"{stats['total_pnl']:.2f}%")
    
    st.markdown("---")
    st.subheader("📝 历史信号")
    if st.session_state.signal_log:
        log_df = pd.DataFrame(st.session_state.signal_log).iloc[::-1]
        st.dataframe(log_df.head(20), width='stretch', height=350)
        if st.button("清除日志"):
            st.session_state.signal_log = []
            st.rerun()
    else:
        st.info("等待高置信度信号...")
    
    if st.button("🔌 重置熔断"):
        st.session_state.system_halted = False
        st.session_state.last_price = 0
        st.session_state.last_signal_time = 0
        st.session_state.active_signal = None
        st.session_state.last_signal_candle = None
        st.session_state.position = None

# ================================
# 9. 主界面（含UI增强）
# ================================
st.title("⚖️ ETH 100x 终极双向评分 AI 决策终端 (趋势+动量+模型)")

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 熔断检测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error("🚨 触发熔断保护！价格剧烈波动。")
    else:
        # 检查当前持仓是否触发止损/止盈
        if st.session_state.position:
            exit_info = check_position_exit(st.session_state.position, current_price)
            if exit_info:
                pnl_percent, reason = exit_info
                net_pnl = pnl_percent - 0.002
                update_stats(net_pnl)
                pos = st.session_state.position
                st.session_state.signal_log.append({
                    "时间": datetime.now().strftime("%H:%M:%S"),
                    "方向": pos['side'],
                    "入场价": pos['entry'],
                    "出场价": current_price,
                    "盈亏%": f"{net_pnl*100:.2f}",
                    "原因": reason
                })
                st.session_state.position = None
        
        # 获取多周期数据并计算指标
        df_5m, df_15m, df_1h = get_multi_timeframe_data()
        df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)
        
        # 计算各项评分
        trend_long, trend_short, raw_trend_long, raw_trend_short = compute_trend_score(df_15m, df_1h)
        mom_long, mom_short = compute_momentum_score(df_5m)
        prob_l, prob_s = compute_model_prob(df_5m, latest_feat, trend_long, trend_short)
        
        # 归一化分数
        trend_long_norm = trend_long / 100.0
        trend_short_norm = trend_short / 100.0
        mom_long_norm = mom_long / 100.0
        mom_short_norm = mom_short / 100.0
        prob_l_norm = prob_l / 100.0
        prob_s_norm = prob_s / 100.0
        
        # 计算最终多空信心分
        final_long = (trend_long_norm * TREND_WEIGHT +
                      mom_long_norm * MOMENTUM_WEIGHT +
                      prob_l_norm * MODEL_WEIGHT) * 100
        final_short = (trend_short_norm * TREND_WEIGHT +
                       mom_short_norm * MOMENTUM_WEIGHT +
                       prob_s_norm * MODEL_WEIGHT) * 100
        
        # 获取最新值用于条件检查
        c5 = df_5m.iloc[-1]
        c15 = df_15m.iloc[-1]
        c1h = df_1h.iloc[-1]
        
        # 安全计算 vol_ratio
        if pd.notna(c5['volume']) and pd.notna(c5['volume_ma20']) and c5['volume_ma20'] > 0:
            vol_ratio = c5['volume'] / c5['volume_ma20']
        else:
            vol_ratio = 0
        atr_pct = c5['atr_pct'] if pd.notna(c5['atr_pct']) else 0
        
        # 趋势强度指数
        trend_strength_raw = abs(raw_trend_long - raw_trend_short)
        score_gap = abs(final_long - final_short)
        model_gap = abs(prob_l - prob_s)
        
        # 市场状态识别
        adx_15 = c15['adx'] if pd.notna(c15['adx']) else 0
        adx_1h = c1h['adx'] if pd.notna(c1h['adx']) else 0
        if adx_15 < 20 and adx_1h < 20:
            market_state = "RANGE"
        elif trend_strength_raw > STRONG_TREND_THRESH:
            market_state = "STRONG_TREND"
        else:
            market_state = "NORMAL"
        
        # 检测动量衰减和爆发
        momentum_decay = detect_momentum_decay(df_5m)
        is_breakout = detect_breakout(df_5m)
        
        # 当前K线时间戳（毫秒）
        current_candle_time = df_5m.index[-1].value / 10**6
        
        # 冷却时间检查
        if st.session_state.last_signal_candle is not None:
            candles_since_last = (current_candle_time - st.session_state.last_signal_candle) / CANDLE_5M_MS
            cooling = candles_since_last < COOLDOWN_CANDLES
        else:
            cooling = False
        
        # 初始化无信号
        direction = None
        final_score = 0
        filter_reasons = []
        
        # 过滤条件
        if cooling:
            filter_reasons.append(f"冷却中，还需 {COOLDOWN_CANDLES - candles_since_last:.1f} 根K线")
        if atr_pct < MIN_ATR_PCT:
            filter_reasons.append(f"波动率过低 (ATR% = {atr_pct:.3%})")
        if vol_ratio < VOLUME_RATIO_MIN:
            filter_reasons.append(f"成交量不足 (倍数 {vol_ratio:.2f})")
        if trend_strength_raw < MIN_TREND_STRENGTH:
            filter_reasons.append(f"趋势强度过弱 ({trend_strength_raw} < {MIN_TREND_STRENGTH})")
        if score_gap < MIN_SCORE_GAP:
            filter_reasons.append(f"多空信心分差过小 ({score_gap:.1f} < {MIN_SCORE_GAP})")
        if market_state == "RANGE":
            filter_reasons.append("市场处于震荡期 (双ADX<20)")
        if momentum_decay:
            filter_reasons.append("动量衰减 (MACD连续下降)")
        
        # 如果基础条件满足，进行方向判断
        if not filter_reasons:
            current_thres = BREAKOUT_CONF_THRES if is_breakout else FINAL_CONF_THRES
            
            if final_long > final_short and final_long >= current_thres:
                candidate_dir = "LONG"
                candidate_score = final_long
            elif final_short > final_long and final_short >= current_thres:
                candidate_dir = "SHORT"
                candidate_score = final_short
            else:
                candidate_dir = None
            
            # 模型方向确认
            if candidate_dir == "LONG" and prob_l < MODEL_DIRECTION_MIN:
                filter_reasons.append(f"模型多头概率不足 ({prob_l:.1f}% < {MODEL_DIRECTION_MIN}%)")
                candidate_dir = None
            elif candidate_dir == "SHORT" and prob_s < MODEL_DIRECTION_MIN:
                filter_reasons.append(f"模型空头概率不足 ({prob_s:.1f}% < {MODEL_DIRECTION_MIN}%)")
                candidate_dir = None
            
            if candidate_dir and model_gap < MODEL_GAP_MIN:
                filter_reasons.append(f"模型概率差过小 ({model_gap:.1f} < {MODEL_GAP_MIN})")
                candidate_dir = None
            
            # 趋势同步锁
            if candidate_dir == "LONG":
                if not (pd.notna(c15['close']) and pd.notna(c15['ema200']) and
                        pd.notna(c1h['close']) and pd.notna(c1h['ema200']) and
                        c15['close'] > c15['ema200'] and c1h['close'] > c1h['ema200']):
                    filter_reasons.append("大周期未支持多头趋势 (15m或1h价格低于EMA200)")
                    candidate_dir = None
            elif candidate_dir == "SHORT":
                if not (pd.notna(c15['close']) and pd.notna(c15['ema200']) and
                        pd.notna(c1h['close']) and pd.notna(c1h['ema200']) and
                        c15['close'] < c15['ema200'] and c1h['close'] < c1h['ema200']):
                    filter_reasons.append("大周期未支持空头趋势 (15m或1h价格高于EMA200)")
                    candidate_dir = None
            
            if candidate_dir:
                direction = candidate_dir
                final_score = candidate_score
        
        # 更新信号锁
        if direction and st.session_state.last_signal_candle != current_candle_time:
            st.session_state.active_signal = direction
            st.session_state.last_signal_candle = current_candle_time
            st.session_state.last_signal_time = time.time()
        elif not direction and st.session_state.last_signal_candle != current_candle_time:
            st.session_state.active_signal = None
        
        # ========== 顶部仪表盘（原有指标） ==========
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("趋势核 (多/空)", f"{trend_long}/{trend_short}")
        col3.metric("动量核 (多/空)", f"{mom_long}/{mom_short}")
        col4.metric("模型 (多/空)", f"{prob_l:.0f}%/{prob_s:.0f}%")
        # 最终信心分带方向图标和颜色
        if final_long > final_short:
            final_text = f"🟢 {final_long:.0f} ▲ / {final_short:.0f}"
        elif final_short > final_long:
            final_text = f"🔴 {final_long:.0f} / {final_short:.0f} ▼"
        else:
            final_text = f"⚪ {final_long:.0f} / {final_short:.0f} ●"
        col5.markdown(f"**最终信心**<br><span style='font-size:1.2rem;'>{final_text}</span>", unsafe_allow_html=True)
        
        # ========== UI增强：信号强度条与状态卡片 ==========
        # 计算信号置信度和方向强度
        confidence = max(final_long, final_short) / 100.0  # 0~1
        strength = abs(final_long - final_short) / 100.0   # 差距强度
        signal_direction = "LONG" if final_long > final_short else "SHORT" if final_short > final_long else "NEUTRAL"
        
        # 根据置信度设置颜色
        if confidence < 0.3:
            color = "#F44336"  # 红
            status = "弱或无信号"
        elif confidence < 0.6:
            color = "#FFC107"  # 黄
            status = "观察"
        else:
            color = "#4CAF50"  # 绿
            status = "可交易"
        
        # 卡片化显示信号状态
        with st.container():
            st.markdown("---")
            col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
            with col_s1:
                st.metric("信号方向", signal_direction, delta=f"强度 {strength*100:.0f}%")
            with col_s2:
                st.markdown(f"**状态**<br><span style='color:{color}; font-size:1.2rem;'>{status}</span>", unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"**成交量**<br>{'✔ 高于均线' if vol_ratio >= 1.0 else '⚠ 低于均线'}", unsafe_allow_html=True)
            
            # 强度条
            st.progress(confidence, text=f"信号置信度 {confidence*100:.0f}%")
        
        # 动态背景提示
        if confidence >= 0.6:
            st.success(f"✔ 信号已确认：{signal_direction} 趋势（强度 {strength*100:.0f}%）")
        elif confidence >= 0.3:
            st.warning(f"⚠ 信号较弱：{signal_direction} 观察中（强度 {strength*100:.0f}%）")
        else:
            st.info("🔍 无明显信号，等待确认")
        
        # 显示过滤状态（如果被过滤）
        if filter_reasons:
            st.warning("⛔ 当前不满足信号条件: " + " | ".join(filter_reasons))
        else:
            st.success("✅ 所有基础过滤条件通过，等待高置信度信号...")
        
        st.markdown("---")
        
        # ========== 开仓逻辑 ==========
        if st.session_state.active_signal and st.session_state.last_signal_candle == current_candle_time and st.session_state.position is None:
            side = st.session_state.active_signal
            st.success(f"🎯 **高置信度交易信号：{side}** (信心分 {final_score:.1f})")
            
            # 计算止损止盈
            atr_raw = df_5m['atr'].iloc[-1] if pd.notna(df_5m['atr'].iloc[-1]) else current_price * 0.001
            max_sl = current_price * 0.003
            atr_sl = atr_raw * 1.5
            min_sl = current_price * MIN_SL_PCT
            sl_dist = max(min_sl, min(atr_sl, max_sl))
            sl = current_price - sl_dist if side == "LONG" else current_price + sl_dist
            tp = current_price + sl_dist * RR if side == "LONG" else current_price - sl_dist * RR
            
            st.session_state.position = {
                'side': side,
                'entry': current_price,
                'sl': sl,
                'tp': tp,
                'entry_time': datetime.now(),
                'score': final_score
            }
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.write(f"**入场价:** {current_price}")
            sc2.write(f"**止损 (SL):** {round(sl, 2)}")
            sc3.write(f"**止盈 (TP):** {round(tp, 2)}")
        else:
            st.info("🔎 当前无符合要求的信号")
        
        # 显示K线图
        fig = go.Figure(data=[go.Candlestick(
            x=df_5m.index,
            open=df_5m['open'], high=df_5m['high'], low=df_5m['low'], close=df_5m['close']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    import traceback
    st.sidebar.error(f"系统运行异常: {e}")
    st.sidebar.code(traceback.format_exc())
    st.stop()
