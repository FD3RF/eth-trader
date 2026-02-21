import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# =============================
# 1. 配置与本地环境优化
# =============================
SYMBOL = "ETH/USDT"
REFRESH_MS = 1000  # 本地部署建议 1 秒刷新一次，极致响应
LEVERAGE = 100
CIRCUIT_BREAKER_PCT = 0.005 # 0.5% 闪崩熔断

st.set_page_config(layout="wide", page_title="ETH 100x Local-Pro", page_icon="⚡")

# 启动自动刷新
st_autorefresh(interval=REFRESH_MS, key="local_update")

# =============================
# 2. 交易所初始化 (支持代理)
# =============================
@st.cache_resource
def get_exchange():
    # 如果你本地需要梯子才能上币安，请在 proxies 里填入你的代理端口（通常是 7890 或 1080）
    # 如果直连能上，就把 proxies 删掉
    config = {
        "enableRateLimit": True,
        # "proxies": {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}, 
    }
    return ccxt.binance(config)

exchange = get_exchange()

if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 3. 核心算法 (Pro 10级结构)
# =============================

def detect_regime(df):
    """15m 级别判定市场环境"""
    df["adx"] = ta.adx(df["h"], df["l"], df["c"], 14)["ADX_14"]
    df["atr"] = ta.atr(df["h"], df["l"], df["c"], 14)
    df["ema21"] = ta.ema(df["c"], 21)
    adx_mean = df["adx"].tail(20).mean()
    slope = df["ema21"].iloc[-1] - df["ema21"].iloc[-5]
    
    if adx_mean > 25 and abs(slope) > 0.1: return "TREND", df["atr"].iloc[-1]
    elif df["atr"].iloc[-1] > (df["c"].iloc[-1] * 0.003): return "VOLATILE", df["atr"].iloc[-1]
    else: return "RANGE", df["atr"].iloc[-1]

def tf_score(df):
    """多周期因子评分"""
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
    """5m 级别衰竭概率检测"""
    adx_drop = df["adx"].iloc[-1] < df["adx"].iloc[-3]
    hist_shrink = abs(df["hist"].iloc[-1]) < abs(df["hist"].iloc[-2])
    vol_drop = df["v"].iloc[-1] < df["v"].rolling(20).mean().iloc[-1]
    return sum([adx_drop, hist_shrink, vol_drop]) / 3

# =============================
# 4. 仪表盘渲染
# =============================
st.title("⚡ ETH 100x 极速自适应系统 (本地版)")

if st.sidebar.button("🔌 重置熔断系统"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    # 极速价格捕获
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 熔断监测逻辑
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    
    st.session_state.last_price = current_price
    
    if st.session_state.system_halted:
        st.error(f"🚨 触发熔断！检测到瞬时异常波动 ({change:.4%})。")
        st.toast("检测到异常波动，系统已锁定", icon="⚠️")
    else:
        # 并发获取 5m/15m 数据
        b5 = exchange.fetch_ohlcv(SYMBOL, "5m", 150)
        b15 = exchange.fetch_ohlcv(SYMBOL, "15m", 150)
        df5 = pd.DataFrame(b5, columns=["t","o","h","l","c","v"])
        df15 = pd.DataFrame(b15, columns=["t","o","h","l","c","v"])

        regime, atr = detect_regime(df15)
        score_5 = tf_score(df5)
        exhaust = exhaustion_prob(df5)

        # 动态 TP/SL 计算
        strength = abs(score_5) / 100
        tp_multiplier = 1.2 + (strength * 2.5)
        if exhaust > 0.66: tp_multiplier *= 0.7

        # UI 板块
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ETH Price", f"${current_price}", delta=f"{round(current_price - b5[-2][4], 2)}")
        m2.metric("Regime (15m)", regime)
        m3.metric("5m Score", f"{score_5} pt")
        m4.metric("Exhaustion", f"{round(exhaust*100, 1)}%")

        if abs(score_5) >= 50:
            side = "LONG" if score_5 > 0 else "SHORT"
            sl_dist = min(atr * 1.2, current_price * 0.003) # 100x 硬止损
            sl = current_price - sl_dist if side == "LONG" else current_price + sl_dist
            tp = current_price + (current_price - sl) * tp_multiplier if side == "LONG" else current_price - (sl - current_price) * tp_multiplier
            
            st.markdown(f"### 🎯 作战建议: <span style='color:{'#00ff00' if side=='LONG' else '#ff4b4b'}'>{side}</span>", unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            sc1.success(f"**入场:** {current_price}")
            sc2.error(f"**止损:** {round(sl, 2)}")
            sc3.info(f"**止盈:** {round(tp, 2)} (1:{round(tp_multiplier, 1)})")
        else:
            st.info("📊 结构等待中，动能未达标...")

        # 图表渲染
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df5['t'], unit='ms'),
            open=df5['o'], high=df5['h'], low=df5['l'], close=df5['c'],
            name="ETH/USDT 5m"
        )])
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), 
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"运行异常: {e}")
