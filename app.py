import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# =============================
# 1. 核心参数与配置
# =============================
SYMBOL = "ETH/USDT"
REFRESH_MS = 3000  # 3秒刷新一次，适合 100x 节奏
LEVERAGE = 100
CIRCUIT_BREAKER_PCT = 0.005 # 0.5% 闪崩熔断

st.set_page_config(layout="wide", page_title="ETH 100x AI Pro")

# 强制使用 Streamlit Autorefresh 防止 503 超时
st_autorefresh(interval=REFRESH_MS, key="data_update")

# 初始化交易所
@st.cache_resource
def get_exchange():
    return ccxt.binance({"enableRateLimit": True})

exchange = get_exchange()

# 状态管理
if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 2. 核心算法逻辑 (10级 Pro 版)
# =============================

def detect_regime(df):
    """15m 级别结构判定"""
    df["adx"] = ta.adx(df["h"], df["l"], df["c"], 14)["ADX_14"]
    df["atr"] = ta.atr(df["h"], df["l"], df["c"], 14)
    df["ema21"] = ta.ema(df["c"], 21)
    
    adx_mean = df["adx"].tail(20).mean()
    atr_val = df["atr"].iloc[-1]
    slope = df["ema21"].iloc[-1] - df["ema21"].iloc[-5]
    
    if adx_mean > 25 and abs(slope) > 0.1: return "TREND", atr_val
    elif atr_val > (df["c"].iloc[-1] * 0.003): return "VOLATILE", atr_val
    else: return "RANGE", atr_val

def tf_score(df):
    """多维度动能评分"""
    df["ema9"] = ta.ema(df["c"], 9)
    df["ema21"] = ta.ema(df["c"], 21)
    df["rsi"] = ta.rsi(df["c"], 14)
    df["adx"] = ta.adx(df["h"], df["l"], df["c"], 14)["ADX_14"]
    macd = ta.macd(df["c"])
    df["hist"] = macd["MACDh_12_26_9"]

    last = df.iloc[-1]
    score = 0
    score += 20 if last["ema9"] > last["ema21"] else -20
    score += 20 if last["hist"] > 0 else -20
    score += 25 if last["adx"] > 25 else 0
    if last["rsi"] > 60: score += 15
    elif last["rsi"] < 40: score -= 15
    
    vol_mean = df["v"].rolling(20).mean().iloc[-1]
    if last["v"] > vol_mean * 1.2:
        score += 20 if score > 0 else -20
    return score

def exhaustion_prob(df):
    """衰竭概率计算"""
    adx_drop = df["adx"].iloc[-1] < df["adx"].iloc[-3]
    hist_shrink = abs(df["hist"].iloc[-1]) < abs(df["hist"].iloc[-2])
    vol_drop = df["v"].iloc[-1] < df["v"].rolling(20).mean().iloc[-1]
    return sum([adx_drop, hist_shrink, vol_drop]) / 3

# =============================
# 3. UI 渲染与执行
# =============================
st.title("🛡️ ETH 100x AI 专家自适应系统")

# 侧边栏：状态控制
if st.sidebar.button("🔌 重置系统熔断"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    # 1. 实时价格获取与熔断检查
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    
    st.session_state.last_price = current_price
    
    if st.session_state.system_halted:
        st.error(f"🚨 触发熔断！检测到瞬时异常波动。")
    else:
        # 2. 数据获取
        b5 = exchange.fetch_ohlcv(SYMBOL, "5m", 100)
        b15 = exchange.fetch_ohlcv(SYMBOL, "15m", 100)
        df5 = pd.DataFrame(b5, columns=["t","o","h","l","c","v"])
        df15 = pd.DataFrame(b15, columns=["t","o","h","l","c","v"])

        # 3. 计算指标
        regime, atr = detect_regime(df15)
        score_5 = tf_score(df5)
        exhaust = exhaustion_prob(df5)

        # 4. 动态 TP/SL
        strength = abs(score_5) / 100
        tp_multiplier = 1.2 + (strength * 2.5)
        if exhaust > 0.66: tp_multiplier *= 0.7

        # 5. 顶层看板
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ETH Price", f"${current_price}")
        m2.metric("Regime", regime)
        m3.metric("5m Score", f"{score_5} pt")
        m4.metric("Exhaustion", f"{round(exhaust*100, 1)}%")

        # 6. 信号输出
        if abs(score_5) >= 50:
            side = "LONG" if score_5 > 0 else "SHORT"
            sl_dist = min(atr * 1.2, current_price * 0.003)
            sl = current_price - sl_dist if side == "LONG" else current_price + sl_dist
            tp = current_price + (current_price - sl) * tp_multiplier if side == "LONG" else current_price - (sl - current_price) * tp_multiplier
            
            st.warning(f"🎯 **AI 作战建议: {side}**")
            c1, c2, c3 = st.columns(3)
            c1.write(f"**入场价:** {current_price}")
            c2.write(f"**动态止损:** {round(sl, 2)}")
            c3.write(f"**动态止盈 (1:{round(tp_multiplier,1)}):** {round(tp, 2)}")
        else:
            st.info("💎 市场结构演变中，暂无高胜率信号...")

        # 7. K线图
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df5['t'], unit='ms'),
            open=df5['o'], high=df5['h'], low=df5['l'], close=df5['c']
        )])
        fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        # 修复弃用警告：使用 width="stretch"
        st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.sidebar.error(f"API Error: {e}")
