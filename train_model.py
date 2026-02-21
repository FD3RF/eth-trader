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

pd.set_option('future.no_silent_downcasting', True)

# ================================
# 1. 核心参数
# ================================
st.set_page_config(layout="wide", page_title="ETH 100x 终极双向评分 AI (OKX)", page_icon="⚖️")
SYMBOL = "ETH/USDT:USDT"
REFRESH_MS = 2500
CIRCUIT_BREAKER_PCT = 0.003
FINAL_CONF_THRES = 80
BREAKOUT_CONF_THRES = 75

TREND_WEIGHT = 0.5
MOMENTUM_WEIGHT = 0.3
MODEL_WEIGHT = 0.2

MIN_ATR_PCT = 0.0025
MIN_SCORE_GAP = 10
VOLUME_RATIO_MIN = 1.2
MODEL_DIRECTION_MIN = 55
MODEL_GAP_MIN = 5
RR = 2.0
MIN_SL_PCT = 0.0015
MIN_TREND_STRENGTH = 15
STRONG_TREND_THRESH = 35
COOLDOWN_CANDLES = 2
CANDLE_5M_MS = 5 * 60 * 1000
BREAKOUT_VOL_RATIO = 1.5
BREAKOUT_ADX_MIN = 25

st_autorefresh(interval=REFRESH_MS, key="bidirectional_ai_final")

# ================================
# 2. 初始化
# ================================
@st.cache_resource
def init_system():
    exch = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
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
# 3. 状态
# ================================
if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False
if 'signal_log' not in st.session_state: st.session_state.signal_log = []
if 'last_signal_time' not in st.session_state: st.session_state.last_signal_time = 0
if 'active_signal' not in st.session_state: st.session_state.active_signal = None
if 'last_signal_candle' not in st.session_state: st.session_state.last_signal_candle = None
if 'position' not in st.session_state: st.session_state.position = None
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0,
        'max_consecutive_losses': 0, 'current_consecutive_losses': 0
    }

# ================================
# 4. 数据获取
# ================================
def fetch_ohlcv(timeframe, limit=200):
    return exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)

def get_multi_timeframe_data():
    df_5m = pd.DataFrame(fetch_ohlcv("5m", 200), columns=["t", "o", "h", "l", "c", "v"])
    df_15m = pd.DataFrame(fetch_ohlcv("15m", 100), columns=["t", "o", "h", "l", "c", "v"])
    df_1h = pd.DataFrame(fetch_ohlcv("1h", 100), columns=["t", "o", "h", "l", "c", "v"])
    return df_5m, df_15m, df_1h

# ================================
# 5. 指标计算（修复 VWAP 列名）
# ================================
def compute_features(df_5m, df_15m, df_1h):
    for df in [df_5m, df_15m, df_1h]:
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df.sort_index(inplace=True)

    # 5m
    df_5m["rsi"] = ta.rsi(df_5m["c"], 14)
    df_5m["ma20"] = ta.sma(df_5m["c"], 20)
    df_5m["ma60"] = ta.sma(df_5m["c"], 60)
    macd = ta.macd(df_5m["c"])
    df_5m["macd"] = macd["MACD_12_26_9"]
    df_5m["macd_signal"] = macd["MACD_12_26_9"]   # 根据训练脚本调整
    df_5m["atr"] = ta.atr(df_5m["h"], df_5m["l"], df_5m["c"], 14)
    df_5m["atr_pct"] = df_5m["atr"] / df_5m["c"]
    df_5m["adx"] = ta.adx(df_5m["h"], df_5m["l"], df_5m["c"], 14)["ADX_14"]
    df_5m["ema9"] = ta.ema(df_5m["c"], 9)
    df_5m["ema21"] = ta.ema(df_5m["c"], 21)
    # 修正：明确指定列名
    df_5m = df_5m.ta.vwap(high='h', low='l', close='c', volume='v', append=True)
    df_5m["volume_ma20"] = ta.sma(df_5m["v"], 20)
    df_5m["atr_ma20"] = df_5m["atr"].rolling(20).mean()
    df_5m["atr_surge"] = df_5m["atr"] > df_5m["atr_ma20"] * 1.2

    # 15m
    df_15m["ema200"] = ta.ema(df_15m["c"], 200)
    df_15m["adx"] = ta.adx(df_15m["h"], df_15m["l"], df_15m["c"], 14)["ADX_14"]
    df_15m = df_15m.ta.vwap(high='h', low='l', close='c', volume='v', append=True)
    df_15m["hh"] = df_15m["h"].rolling(20).max()
    df_15m["ll"] = df_15m["l"].rolling(20).min()
    df_15m["ema200_slope"] = df_15m["ema200"] - df_15m["ema200"].shift(5)

    # 1h
    df_1h["ema200"] = ta.ema(df_1h["c"], 200)
    df_1h["adx"] = ta.adx(df_1h["h"], df_1h["l"], df_1h["c"], 14)["ADX_14"]
    df_1h = df_1h.ta.vwap(high='h', low='l', close='c', volume='v', append=True)
    df_1h["hh"] = df_1h["h"].rolling(20).max()
    df_1h["ll"] = df_1h["l"].rolling(20).min()
    df_1h["ema200_slope"] = df_1h["ema200"] - df_1h["ema200"].shift(3)

    df_5m = df_5m.fillna(0)
    df_15m = df_15m.fillna(0)
    df_1h = df_1h.fillna(0)

    feat_cols = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
    latest_feat = df_5m[feat_cols].iloc[-1:].fillna(0)
    return df_5m, df_15m, df_1h, latest_feat

# ================================
# 6. 评分函数（注意 VWAP 列名）
# ================================
def compute_trend_score(df_15m, df_1h):
    c15 = df_15m.iloc[-1]
    c1h = df_1h.iloc[-1]
    long_score = 0
    short_score = 0

    if c15['c'] > c15['ema200'] and c15['ema200_slope'] > 0:
        long_score += 15
    elif c15['c'] < c15['ema200'] and c15['ema200_slope'] < 0:
        short_score += 15

    if c1h['c'] > c1h['ema200'] and c1h['ema200_slope'] > 0:
        long_score += 15
    elif c1h['c'] < c1h['ema200'] and c1h['ema200_slope'] < 0:
        short_score += 15

    if c15['c'] > c15['VWAP']:
        long_score += 10
    else:
        short_score += 10

    if c1h['c'] > c1h['VWAP']:
        long_score += 10
    else:
        short_score += 10

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

    raw_long = min(long_score, 100)
    raw_short = min(short_score, 100)

    if c15['adx'] > 25 and c1h['adx'] > 25:
        long_score = int(long_score * 1.15)
        short_score = int(short_score * 1.15)

    return min(long_score,100), min(short_score,100), raw_long, raw_short

def compute_momentum_score(df_5m):
    c = df_5m.iloc[-1]
    long_score = 0
    short_score = 0

    if c['ema9'] > c['ema21']:
        long_score += 30
    else:
        short_score += 30

    if c['c'] > c['VWAP']:
        long_score += 20
    else:
        short_score += 20

    if c['v'] > c['volume_ma20'] * VOLUME_RATIO_MIN:
        long_score += 25
        short_score += 25

    if c['atr_surge']:
        if c['ema9'] > c['ema21']:
            long_score += 25
        else:
            short_score += 25

    return min(long_score,100), min(short_score,100)

def compute_model_prob(df_5m, latest_feat):
    if model_long is None or model_short is None:
        return 50, 50
    latest_feat = latest_feat.fillna(0)
    prob_l = model_long.predict_proba(latest_feat)[0][1] * 100
    prob_s = model_short.predict_proba(latest_feat)[0][1] * 100
    return prob_l, prob_s

def detect_momentum_decay(df_5m):
    if len(df_5m) < 4: return False
    macd_vals = df_5m['macd'].iloc[-4:].values
    return (macd_vals[3] < macd_vals[2] and macd_vals[2] < macd_vals[1] and macd_vals[1] < macd_vals[0])

def detect_breakout(df_5m):
    c = df_5m.iloc[-1]
    vol_ratio = c['v'] / c['volume_ma20'] if c['volume_ma20'] > 0 else 0
    return (c['atr_surge'] and vol_ratio > BREAKOUT_VOL_RATIO and c['adx'] > BREAKOUT_ADX_MIN)

# ================================
# 7. 盈亏统计
# ================================
def check_position_exit(position, current_price):
    if position is None: return None
    side, entry, sl, tp = position['side'], position['entry'], position['sl'], position['tp']
    if side == 'LONG':
        if current_price <= sl: return (sl - entry) / entry, '止损'
        if current_price >= tp: return (tp - entry) / entry, '止盈'
    else:
        if current_price >= sl: return (entry - sl) / entry, '止损'
        if current_price <= tp: return (entry - tp) / entry, '止盈'
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
        win_rate = (stats['wins'] / max(stats['total_trades'],1)) * 100
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
# 9. 主界面
# ================================
st.title("⚖️ ETH 100x 终极双向评分 AI 决策终端 (趋势+动量+模型)")

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error("🚨 触发熔断保护！价格剧烈波动。")
    else:
        # 检查持仓退出
        if st.session_state.position:
            exit_info = check_position_exit(st.session_state.position, current_price)
            if exit_info:
                pnl_percent, reason = exit_info
                net_pnl = pnl_percent - 0.002  # 简化手续费+滑点
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

        # 获取数据
        df_5m, df_15m, df_1h = get_multi_timeframe_data()
        df_5m, df_15m, df_1h, latest_feat = compute_features(df_5m, df_15m, df_1h)

        # 计算分数
        trend_long, trend_short, raw_trend_long, raw_trend_short = compute_trend_score(df_15m, df_1h)
        mom_long, mom_short = compute_momentum_score(df_5m)
        prob_l, prob_s = compute_model_prob(df_5m, latest_feat)

        trend_long_norm = trend_long / 100.0
        trend_short_norm = trend_short / 100.0
        mom_long_norm = mom_long / 100.0
        mom_short_norm = mom_short / 100.0
        prob_l_norm = prob_l / 100.0
        prob_s_norm = prob_s / 100.0

        final_long = (trend_long_norm * TREND_WEIGHT + mom_long_norm * MOMENTUM_WEIGHT + prob_l_norm * MODEL_WEIGHT) * 100
        final_short = (trend_short_norm * TREND_WEIGHT + mom_short_norm * MOMENTUM_WEIGHT + prob_s_norm * MODEL_WEIGHT) * 100

        c5 = df_5m.iloc[-1]
        c15 = df_15m.iloc[-1]
        c1h = df_1h.iloc[-1]
        vol_ratio = c5['v'] / c5['volume_ma20'] if c5['volume_ma20'] > 0 else 0
        atr_pct = c5['atr_pct']
        trend_strength_raw = abs(raw_trend_long - raw_trend_short)
        score_gap = abs(final_long - final_short)
        model_gap = abs(prob_l - prob_s)
        adx_15, adx_1h = c15['adx'], c1h['adx']
        market_state = "RANGE" if (adx_15 < 20 and adx_1h < 20) else "STRONG_TREND" if trend_strength_raw > STRONG_TREND_THRESH else "NORMAL"
        momentum_decay = detect_momentum_decay(df_5m)
        is_breakout = detect_breakout(df_5m)
        current_candle_time = df_5m.index[-1].value / 10**6

        if st.session_state.last_signal_candle is not None:
            candles_since_last = (current_candle_time - st.session_state.last_signal_candle) / CANDLE_5M_MS
            cooling = candles_since_last < COOLDOWN_CANDLES
        else:
            cooling = False

        direction = None
        final_score = 0
        filter_reasons = []

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

            if candidate_dir == "LONG" and prob_l < MODEL_DIRECTION_MIN:
                filter_reasons.append(f"模型多头概率不足 ({prob_l:.1f}% < {MODEL_DIRECTION_MIN}%)")
                candidate_dir = None
            elif candidate_dir == "SHORT" and prob_s < MODEL_DIRECTION_MIN:
                filter_reasons.append(f"模型空头概率不足 ({prob_s:.1f}% < {MODEL_DIRECTION_MIN}%)")
                candidate_dir = None

            if candidate_dir and model_gap < MODEL_GAP_MIN:
                filter_reasons.append(f"模型概率差过小 ({model_gap:.1f} < {MODEL_GAP_MIN})")
                candidate_dir = None

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

        if direction and st.session_state.last_signal_candle != current_candle_time:
            st.session_state.active_signal = direction
            st.session_state.last_signal_candle = current_candle_time
            st.session_state.last_signal_time = time.time()
        elif not direction and st.session_state.last_signal_candle != current_candle_time:
            st.session_state.active_signal = None

        # 顶部指标
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ETH 实时价", f"${current_price}")
        col2.metric("趋势核 (多/空)", f"{trend_long}/{trend_short}")
        col3.metric("动量核 (多/空)", f"{mom_long}/{mom_short}")
        col4.metric("模型 (多/空)", f"{prob_l:.0f}%/{prob_s:.0f}%")
        col5.metric("最终信心", f"{final_long:.0f}/{final_short:.0f}")

        if filter_reasons:
            st.warning("⛔ 当前不满足信号条件: " + " | ".join(filter_reasons))
        else:
            st.success("✅ 所有过滤条件通过，等待信号触发...")

        st.markdown("---")

        # 开仓
        if st.session_state.active_signal and st.session_state.last_signal_candle == current_candle_time and st.session_state.position is None:
            side = st.session_state.active_signal
            st.success(f"🎯 **高置信度交易信号：{side}** (信心分 {final_score:.1f})")

            atr_raw = df_5m['atr'].iloc[-1]
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

        # K线图
        fig = go.Figure(data=[go.Candlestick(x=df_5m.index, open=df_5m['o'], high=df_5m['h'], low=df_5m['l'], close=df_5m['c'])])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch')

except Exception as e:
    st.sidebar.error(f"系统运行异常: {e}")
