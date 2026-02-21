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

# 设置 pandas 选项，抑制未来版本的 downcasting 警告（可选）
pd.set_option('future.no_silent_downcasting', True)

# ================================
# 1. 核心参数与看板设置
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x 终极双向 AI (OKX)", page_icon="⚖️")

SYMBOL = "ETH/USDT:USDT"            # OKX 永续合约
REFRESH_MS = 2500                   # 2.5秒刷新
CIRCUIT_BREAKER_PCT = 0.003         # 0.3% 熔断
FINAL_CONF_THRES = 80                # 最终信心分门槛（满分100）
BREAKOUT_CONF_THRES = 75             # 爆发行情下的降低门槛

# 权重配置
TREND_WEIGHT = 0.5
MOMENTUM_WEIGHT = 0.3
MODEL_WEIGHT = 0.2

# 波动率过滤：ATR百分比 < 0.25% 时禁止交易
MIN_ATR_PCT = 0.0025

# 多空信心分最小差值，低于此值不交易
MIN_SCORE_GAP = 10

# 成交量放大倍数要求
VOLUME_RATIO_MIN = 1.2

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
BREAKOUT_ATR_SURGE = True      # ATR surge需为True
BREAKOUT_ADX_MIN = 25          # ADX最小值

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
        m_l = m_s = generic
        if generic:
            st.sidebar.info("💡 使用通用模型镜像多空")
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
    st.session_state.signal_log = []
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = 0  # 上一次信号触发的时间戳（秒）
if 'active_signal' not in st.session_state:
    st.session_state.active_signal = None  # 当前活动信号
if 'last_signal_candle' not in st.session_state:
    st.session_state.last_signal_candle = None  # 上一次触发信号的K线时间戳（毫秒）

# ================================
# 4. 数据获取函数（多时间框架）
# ================================
def fetch_ohlcv(timeframe, limit=200):
    """获取指定周期的K线数据"""
    return exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)

def get_multi_timeframe_data():
    """获取5m、15m、1h数据并计算指标"""
    ohlcv_5m = fetch_ohlcv("5m", 200)
    df_5m = pd.DataFrame(ohlcv_5m, columns=["t", "o", "h", "l", "c", "v"])
    
    ohlcv_15m = fetch_ohlcv("15m", 100)
    df_15m = pd.DataFrame(ohlcv_15m, columns=["t", "o", "h", "l", "c", "v"])
    
    ohlcv_1h = fetch_ohlcv("1h", 100)
    df_1h = pd.DataFrame(ohlcv_1h, columns=["t", "o", "h", "l", "c", "v"])
    
    return df_5m, df_15m, df_1h

# ================================
# 5. 指标计算函数（修复VWAP索引）
# ================================
def compute_features(df_5m, df_15m, df_1h):
    """计算所有需要的指标，返回DataFrame和最新特征向量"""
    # 将时间戳列转换为datetime并设置为索引，确保有序（VWAP要求）
    for df in [df_5m, df_15m, df_1h]:
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df.sort_index(inplace=True)  # 确保按时间升序排列

    # ----- 5m 指标（用于动量核 + 模型）-----
    df_5m["rsi"] = ta.rsi(df_5m["c"], length=14)
    df_5m["ma20"] = ta.sma(df_5m["c"], length=20)
    df_5m["ma60"] = ta.sma(df_5m["c"], length=60)
    macd = ta.macd(df_5m["c"])
    df_5m["macd"] = macd["MACD_12_26_9"]
    # 注意：请根据您的训练脚本调整此行，如果训练时用了MACD线作为信号线，则使用MACD_12_26_9；如果用了标准信号线，则使用MACDs_12_26_9
    df_5m["macd_signal"] = macd["MACD_12_26_9"]   # 默认与训练对齐（请根据实际情况修改）
    df_5m["atr"] = ta.atr(df_5m["h"], df_5m["l"], df_5m["c"], length=14)
    df_5m["atr_pct"] = df_5m["atr"] / df_5m["c"]
    df_5m["adx"] = ta.adx(df_5m["h"], df_5m["l"], df_5m["c"], length=14)["ADX_14"]
    
    # 动量核所需指标
    df_5m["ema9"] = ta.ema(df_5m["c"], length=9)
    df_5m["ema21"] = ta.ema(df_5m["c"], length=21)
    # 使用 DataFrame 方法计算 VWAP，避免索引警告
    df_5m = df_5m.ta.vwap(append=True)   # 添加 'VWAP' 列
    df_5m["volume_ma20"] = ta.sma(df_5m["v"], length=20)
    # ATR扩张判断：当前ATR > 20期平均ATR * 1.2
    df_5m["atr_ma20"] = df_5m["atr"].rolling(20).mean()
    df_5m["atr_surge"] = df_5m["atr"] > df_5m["atr_ma20"] * 1.2
    
    # ----- 15m 指标（用于趋势核）-----
    df_15m["ema200"] = ta.ema(df_15m["c"], length=200)
    df_15m["adx"] = ta.adx(df_15m["h"], df_15m["l"], df_15m["c"], length=14)["ADX_14"]
    df_15m = df_15m.ta.vwap(append=True)
    df_15m["hh"] = df_15m["h"].rolling(20).max()      # 20周期最高点
    df_15m["ll"] = df_15m["l"].rolling(20).min()      # 20周期最低点
    # EMA斜率（当前值与前5根比较）
    df_15m["ema200_slope"] = df_15m["ema200"] - df_15m["ema200"].shift(5)
    
    # ----- 1h 指标（用于趋势核）-----
    df_1h["ema200"] = ta.ema(df_1h["c"], length=200)
    df_1h["adx"] = ta.adx(df_1h["h"], df_1h["l"], df_1h["c"], length=14)["ADX_14"]
    df_1h = df_1h.ta.vwap(append=True)
    df_1h["hh"] = df_1h["h"].rolling(20).max()
    df_1h["ll"] = df_1h["l"].rolling(20).min()
    # EMA斜率（当前值与前3根比较，1h周期长，取3根足够）
    df_1h["ema200_slope"] = df_1h["ema200"] - df_1h["ema200"].shift(3)
    
    # 填充NaN
    df_5m = df_5m.ffill().bfill()
    df_15m = df_15m.ffill().bfill()
    df_1h = df_1h.ffill().bfill()
    
    # 最新一行特征（用于模型预测）
    feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
    latest_feat = df_5m[feat_cols].iloc[-1:]
    
    return df_5m, df_15m, df_1h, latest_feat

# ================================
# 6. 双向评分函数
# ================================
def compute_trend_score(df_15m, df_1h):
    """计算趋势核的多空分数 (0-100)，ADX作为倍率因子，返回原始和放大后分数"""
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]

    long_score = 0
    short_score = 0

    # EMA200 (每项15分) + 斜率验证
    if c15['c'] > c15['ema200'] and c15['ema200_slope'] > 0:
        long_score += 15
    elif c15['c'] < c15['ema200'] and c15['ema200_slope'] < 0:
        short_score += 15
    # 如果价格方向与斜率不一致，不加分（视为假突破）

    if c1h['c'] > c1h['ema200'] and c1h['ema200_slope'] > 0:
        long_score += 15
    elif c1h['c'] < c1h['ema200'] and c1h['ema200_slope'] < 0:
        short_score += 15

    # VWAP (每项10分)
    if c15['c'] > c15['VWAP']:
        long_score += 10
    else:
        short_score += 10

    if c1h['c'] > c1h['VWAP']:
        long_score += 10
    else:
        short_score += 10

    # 价格结构高低点 (每项10分)
    range_15 = c15['hh'] - c15['ll']
    if range_15 > 0:
        if (c15['c'] - c15['ll']) / range_15 > 0.5:
            long_score += 10
        else:
            short_score += 10

    range_1h = c1h['hh'] - c1h['ll']
    if range_1h > 0:
        if (c1h['c'] - c1h['ll']) / range_1h > 0.5:
            long_score += 10
        else:
            short_score += 10

    # 保存原始分数（未放大）
    raw_long = min(long_score, 100)
    raw_short = min(short_score, 100)

    # ADX 作为倍率因子（仅当两个周期都强趋势）
    if c15['adx'] > 25 and c1h['adx'] > 25:
        long_score = int(long_score * 1.15)
        short_score = int(short_score * 1.15)

    # 确保不超过100
    long_score = min(long_score, 100)
    short_score = min(short_score, 100)

    return long_score, short_score, raw_long, raw_short

def compute_momentum_score(df_5m):
    """计算动量核的多空分数 (0-100)，ATR扩张定向增强"""
    c = df_5m.iloc[-1]

    long_score = 0
    short_score = 0

    # EMA9 vs EMA21 (30分)
    if c['ema9'] > c['ema21']:
        long_score += 30
    else:
        short_score += 30

    # 价格 vs VWAP (20分)
    if c['c'] > c['VWAP']:
        long_score += 20
    else:
        short_score += 20

    # 成交量放大 (25分，多空都加)
    if c['v'] > c['volume_ma20'] * VOLUME_RATIO_MIN:
        long_score += 25
        short_score += 25

    # ATR扩张定向增强（只增强当前动量方向）
    if c['atr_surge']:
        if c['ema9'] > c['ema21']:
            long_score += 25
        else:
            short_score += 25

    return min(long_score, 100), min(short_score, 100)

def compute_model_prob(df_5m, latest_feat):
    """获取模型概率并转换为分数 (0-100)"""
    if model_long is None or model_short is None:
        return 50, 50
    prob_l = model_long.predict_proba(latest_feat)[0][1] * 100
    prob_s = model_short.predict_proba(latest_feat)[0][1] * 100
    return prob_l, prob_s

def detect_momentum_decay(df_5m):
    """检测动量是否衰减：MACD连续3根下降"""
    if len(df_5m) < 4:
        return False
    macd_vals = df_5m['macd'].iloc[-4:].values
    # 检查最后3个差值是否均为负（即连续下降）
    return (macd_vals[3] < macd_vals[2] and
            macd_vals[2] < macd_vals[1] and
            macd_vals[1] < macd_vals[0])

def detect_breakout(df_5m):
    """检测是否处于爆发结构"""
    c = df_5m.iloc[-1]
    vol_ratio = c['v'] / c['volume_ma20'] if c['volume_ma20'] > 0 else 0
    return (c['atr_surge'] and
            vol_ratio > BREAKOUT_VOL_RATIO and
            c['adx'] > BREAKOUT_ADX_MIN)

# ================================
# 7. 侧边栏
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
    st.subheader("📝 历史信号")
    if st.session_state.signal_log:
        log_df = pd.DataFrame(st.session_state.signal_log).iloc[::-1]
        st.dataframe(log_df, width='stretch', height=350)
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

# ================================
# 8. 主界面
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
        # 获取多周期数据并计算指标
        df_5m, df_15m, df_1h = get_multi_timeframe_data()
        df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)
        
        # 计算各项评分
        trend_long, trend_short, raw_trend_long, raw_trend_short = compute_trend_score(df_15m, df_1h)
        mom_long, mom_short = compute_momentum_score(df_5m)
        prob_l, prob_s = compute_model_prob(df_5m, latest_feat)
        
        # 归一化分数
        trend_long_norm = trend_long / 100.0
        trend_short_norm = trend_short / 100.0
        mom_long_norm = mom_long / 100.0
        mom_short_norm = mom_short / 100.0
        prob_l_norm = prob_l / 100.0
        prob_s_norm = prob_s / 100.0
        
        # 计算最终多空信心分（归一化后加权再乘100）
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
        vol_ratio = c5['v'] / c5['volume_ma20'] if c5['volume_ma20'] > 0 else 0
        atr_pct = c5['atr_pct']
        
        # 趋势强度指数（使用原始分数，避免ADX放大虚增）
        trend_strength_raw = abs(raw_trend_long - raw_trend_short)
        
        # 多空信心分差值
        score_gap = abs(final_long - final_short)
        
        # 模型概率差值
        model_gap = abs(prob_l - prob_s)
        
        # 市场状态识别
        adx_15 = c15['adx']
        adx_1h = c1h['adx']
        if adx_15 < 20 and adx_1h < 20:
            market_state = "RANGE"
        elif trend_strength_raw > STRONG_TREND_THRESH:
            market_state = "STRONG_TREND"
        else:
            market_state = "NORMAL"
        
        # 检测动量衰减
        momentum_decay = detect_momentum_decay(df_5m)
        
        # 检测爆发结构
        is_breakout = detect_breakout(df_5m)
        
        # 当前K线时间戳（毫秒）
        current_candle_time = df_5m.index[-1].value / 10**6  # 转换为毫秒
        
        # 冷却时间检查（基于K线数量）
        if st.session_state.last_signal_candle is not None:
            candles_since_last = (current_candle_time - st.session_state.last_signal_candle) / CANDLE_5M_MS
            cooling = candles_since_last < COOLDOWN_CANDLES
        else:
            cooling = False
        
        # 初始化无信号
        direction = None
        final_score = 0
        filter_reasons = []
        
        # 冷却检查
        if cooling:
            filter_reasons.append(f"冷却中，还需 {COOLDOWN_CANDLES - candles_since_last:.1f} 根K线")
        
        # 波动率过滤
        if atr_pct < MIN_ATR_PCT:
            filter_reasons.append(f"波动率过低 (ATR% = {atr_pct:.3%})")
        
        # 成交量放大
        if vol_ratio < VOLUME_RATIO_MIN:
            filter_reasons.append(f"成交量不足 (倍数 {vol_ratio:.2f})")
        
        # 趋势强度指数过滤（基于原始分数）
        if trend_strength_raw < MIN_TREND_STRENGTH:
            filter_reasons.append(f"趋势强度过弱 ({trend_strength_raw} < {MIN_TREND_STRENGTH})")
        
        # 多空差值
        if score_gap < MIN_SCORE_GAP:
            filter_reasons.append(f"多空信心分差过小 ({score_gap:.1f} < {MIN_SCORE_GAP})")
        
        # 市场状态过滤（震荡期禁止交易）
        if market_state == "RANGE":
            filter_reasons.append("市场处于震荡期 (双ADX<20)")
        
        # 动量衰减过滤
        if momentum_decay:
            filter_reasons.append("动量衰减 (MACD连续下降)")
        
        # 如果上述基础条件不满足，直接跳过方向判断
        if not filter_reasons:
            # 根据是否爆发调整信心门槛
            current_thres = BREAKOUT_CONF_THRES if is_breakout else FINAL_CONF_THRES
            
            # 确定初步方向
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
            
            # 模型概率差值过滤
            if candidate_dir and model_gap < MODEL_GAP_MIN:
                filter_reasons.append(f"模型概率差过小 ({model_gap:.1f} < {MODEL_GAP_MIN})")
                candidate_dir = None
            
            # 趋势同步锁绑定方向
            if candidate_dir == "LONG":
                if not (c15['c'] > c15['ema200'] and c1h['c'] > c1h['ema200']):
                    filter_reasons.append("大周期未支持多头趋势 (15m或1h价格低于EMA200)")
                    candidate_dir = None
            elif candidate_dir == "SHORT":
                if not (c15['c'] < c15['ema200'] and c1h['c'] < c1h['ema200']):
                    filter_reasons.append("大周期未支持空头趋势 (15m或1h价格高于EMA200)")
                    candidate_dir = None
            
            if candidate_dir:
                direction = candidate_dir
                final_score = candidate_score
        
        # 更新信号锁（基于K线时间戳）
        if direction and st.session_state.last_signal_candle != current_candle_time:
            st.session_state.active_signal = direction
            st.session_state.last_signal_candle = current_candle_time
            st.session_state.last_signal_time = time.time()
        elif not direction:
            # 方向消失，如果当前K线已变，清除活动信号
            if st.session_state.last_signal_candle != current_candle_time:
                st.session_state.active_signal = None
        
        # 顶部仪表盘（使用索引作为x轴不影响显示，但需确保x使用索引）
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("趋势核 (多/空)", f"{trend_long}/{trend_short}")
        col3.metric("动量核 (多/空)", f"{mom_long}/{mom_short}")
        col4.metric("模型 (多/空)", f"{prob_l:.0f}%/{prob_s:.0f}%")
        col5.metric("最终信心", f"{final_long:.0f}/{final_short:.0f}")
        
        # 显示当前过滤状态
        if filter_reasons:
            st.warning("⛔ 当前不满足信号条件: " + " | ".join(filter_reasons))
        else:
            st.success("✅ 所有过滤条件通过，等待信号触发...")
        
        st.markdown("---")
        
        # 如果存在活动信号且与当前K线时间戳一致，显示交易计划
        if st.session_state.active_signal and st.session_state.last_signal_candle == current_candle_time:
            side = st.session_state.active_signal
            st.success(f"🎯 **高置信度交易信号：{side}** (信心分 {final_score:.1f})")
            
            # 止损止盈计算
            atr_raw = df_5m['atr'].iloc[-1]
            max_sl = current_price * 0.003   # 绝对止损上限 0.3%
            atr_sl = atr_raw * 1.5
            min_sl = current_price * MIN_SL_PCT  # 下限 0.15%
            sl_dist = max(min_sl, min(atr_sl, max_sl))
            sl = current_price - sl_dist if side == "LONG" else current_price + sl_dist
            tp = current_price + sl_dist * RR if side == "LONG" else current_price - sl_dist * RR
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.write(f"**入场价:** {current_price}")
            sc2.write(f"**止损 (SL):** {round(sl, 2)}")
            sc3.write(f"**止盈 (TP):** {round(tp, 2)}")
            
            # 记录日志
            t_now = datetime.now().strftime("%H:%M:%S")
            if not st.session_state.signal_log or st.session_state.signal_log[-1]['时间'] != t_now:
                st.session_state.signal_log.append({
                    "时间": t_now,
                    "方向": side,
                    "价格": current_price,
                    "信心分": f"{final_score:.1f}",
                    "趋势": f"{trend_long}/{trend_short}",
                    "动量": f"{mom_long}/{mom_short}",
                    "模型": f"{prob_l:.0f}%/{prob_s:.0f}%",
                    "爆发": "是" if is_breakout else "否"
                })
        else:
            st.info("🔎 当前无符合要求的信号")
        
        # 显示K线图（5m），使用索引作为x轴
        fig = go.Figure(data=[go.Candlestick(
            x=df_5m.index,  # 现在是 DatetimeIndex
            open=df_5m['o'], high=df_5m['h'], low=df_5m['l'], close=df_5m['c']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch')

except Exception as e:
    st.sidebar.error(f"系统运行异常: {e}")
