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
# 1. 核心参数与 UI
# =============================
SYMBOL = 'ETH/USDT:USDT'  # Bybit 合约格式
REFRESH_MS = 2000        # 2秒刷新，兼顾性能与实时性
CIRCUIT_BREAKER_PCT = 0.005 

st.set_page_config(layout="wide", page_title="ETH 100x AI (Bybit)", page_icon="🤖")
st_autorefresh(interval=REFRESH_MS, key="bybit_ai_update")

# =============================
# 2. 交易所与模型加载
# =============================
@st.cache_resource
def init_system():
    # 初始化交易所
    exch = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    
    # 加载 AI 模型
    model = None
    if os.path.exists('eth_ai_model.pkl'):
        model = joblib.load('eth_ai_model.pkl')
    return exch, model

exchange, ai_model = init_system()

# 状态管理
if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 3. 数据处理与 AI 预测
# =============================
def get_latest_analysis():
    # 获取 5m 数据
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 计算特征 (必须与 train_model.py 保持高度一致)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ma20'] = ta.sma(df['close'], length=20)
    df['ma60'] = ta.sma(df['close'], length=60)
    macd = ta.macd(df['close'])
    df['hist'] = macd['MACDh_12_26_9']
    df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    
    # 为 AI 准备最新的特征向量
    features = df[['rsi', 'ma20', 'ma60', 'hist', 'adx']].tail(1)
    
    prediction = None
    if ai_model:
        # 假设模型输出 1 为看涨，0 为看平/跌
        prediction = ai_model.predict(features)[0]
        
    return df, prediction

# =============================
# 4. 实时看板渲染
# =============================
st.title("🤖 ETH 100x AI 智能作战系统 (Bybit 版)")

if st.sidebar.button("🔌 重置系统"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 熔断监测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error(f"🚨 触发熔断保护！检测到异常瞬间波动。")
    else:
        # 获取分析数据
        df, pred = get_latest_analysis()
        last_row = df.iloc[-1]
        
        # 状态展示
        c1, c2, c3 = st.columns(3)
        c1.metric("ETH Bybit Price", f"${current_price}")
        c2.metric("AI Model Status", "ACTIVE ✅" if ai_model else "INDICATOR ONLY ⚠️")
        c3.metric("Trend Strength (ADX)", f"{round(last_row['adx'], 1)}")

        # 核心信号区
        st.divider()
        if pred == 1:
            st.success("🎯 **AI 预测信号：看涨 (LONG)**")
            st.balloons()
        elif pred == 0:
            st.error("🎯 **AI 预测信号：看跌 (SHORT)**")
        else:
            st.info("📊 AI 正在观察市场结构，暂无高置信度预测...")

        # 可视化
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df['timestamp'], unit='ms'),
            open=df['open'], high=df['high'], low=df['low'], close=df['close']
        )])
        fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"连接或预测异常: {e}")
