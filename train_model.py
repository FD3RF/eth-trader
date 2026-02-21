import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import os

# =============================
# CONFIG
# =============================
st.set_page_config(layout="wide", page_title="ETH 100x AI-Pro (Bybit)")

SYMBOL = st.sidebar.text_input("Trading Pair", "ETH/USDT:USDT", help="Bybit linear perpetual")
LEVERAGE = st.sidebar.slider("Leverage (1-100x)", 1, 100, 100)
REFRESH_MS = st.sidebar.slider("Refresh (ms)", 1000, 5000, 2000)
CIRCUIT_BREAKER_PCT = 0.003  # tightened for 100x safety

st_autorefresh(interval=REFRESH_MS, key="bybit_monitor")

# Init
@st.cache_resource
def init_system():
    exch = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "linear"},
        # "proxies": {"http": "...", "https": "..."}  # ← UNCOMMENT & FILL
    })
    model = None
    if os.path.exists("eth_ai_model.pkl"):
        model = joblib.load("eth_ai_model.pkl")
    return exch, model

exchange, ai_model = init_system()

if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# DATA + FEATURES (NaN-safe)
# =============================
def get_analysis_data():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, "5m", limit=150)  # extra history for indicators
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ma20"] = ta.sma(df["close"], length=20)
    df["ma60"] = ta.sma(df["close"], length=60)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    
    df = df.dropna().reset_index(drop=True)  # ← CRITICAL FIX
    features = df[['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr', 'adx']].iloc[-1:].values
    return df, features

# =============================
# UI
# =============================
st.title("🛡️ ETH 100x AI 专家级自适应监控 (Bybit版)")

if st.sidebar.button("🔌 重置系统熔断"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    funding = ticker.get('fundingRate', None)

    # Circuit breaker
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error(f"🚨 系统熔断触发！价格波动 > {CIRCUIT_BREAKER_PCT*100}%")
    else:
        df, current_feat = get_analysis_data()
        
        prediction = None
        if ai_model is not None:
            prediction = ai_model.predict(current_feat)[0]

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ETH 实时价", f"${current_price:,.2f}")
        c2.metric("AI 预测", "看涨 🚀" if prediction == 1 else "观望/看跌 ⌛")
        c3.metric("ATR (5m)", f"{df['atr'].iloc[-1]:.4f}")
        c4.metric("ADX 强度", f"{df['adx'].iloc[-1]:.1f}")
        c5.metric("资金费率", f"{funding*100:.4f}%" if funding else "N/A")

        # Trading Plan
        if prediction == 1:
            st.success("### 🎯 AI 作战建议：**LONG (多)**")
            atr = df["atr"].iloc[-1]
            sl_dist = min(atr * 1.8, current_price * 0.0025)  # safer
            sl = current_price - sl_dist
            tp = current_price + sl_dist * 3.0   # 1:3 RR
            liq_price = current_price * (1 - 1/LEVERAGE * 1.05)  # approx

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("入场", f"${current_price:,.2f}")
            sc2.metric("止损", f"${sl:,.2f} (-{sl_dist/current_price*100:.2f}%)")
            sc3.metric("止盈", f"${tp:,.2f} (+{sl_dist*3/current_price*100:.2f}%)")
            sc4.metric("约爆仓价", f"${liq_price:,.2f}")
        else:
            st.info("📊 市场动能积蓄中，AI 建议**持币观望**...")

        # Chart
        fig = go.Figure(data=[go.Candlestick(x=pd.to_datetime(df['timestamp'], unit='ms'),
                        open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ 连接异常: {str(e)}\n\n**提示**: 尝试开启VPN/代理，或检查网络")
