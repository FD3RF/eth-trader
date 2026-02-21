import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =============================
# 1. 核心参数 (本地极速优化)
# =============================
SYMBOL = 'ETH/USDT:USDT'  # 适配 Bybit 永续合约
REFRESH_MS = 1000        # 1秒极速刷新
CIRCUIT_BREAKER_PCT = 0.005 

st.set_page_config(layout="wide", page_title="ETH 100x Pro (Bybit)", page_icon="📈")
st_autorefresh(interval=REFRESH_MS, key="bybit_monitor")

# =============================
# 2. 交易所初始化 (切换至 Bybit)
# =============================
@st.cache_resource
def get_exchange():
    return ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'} # 线性合约
        # 如果依然无法连接，可在此处添加本地代理：
        # 'proxies': {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    })

exchange = get_exchange()

if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 3. 核心算法 (同步 train_model.py 指标)
# =============================
def get_analysis():
    # 获取 5m 数据进行实时预测
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 匹配 train_model.py 中的指标计算
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ma20'] = ta.sma(df['close'], length=20)
    df['ma60'] = ta.sma(df['close'], length=60)
    macd = ta.macd(df['close'])
    df['hist'] = macd['MACDh_12_26_9']
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    
    return df

# =============================
# 4. 界面渲染
# =============================
st.title("🛡️ ETH 100x Bybit Pro 监控系统")

if st.sidebar.button("🔌 重置系统"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    # 实时价格与熔断检测
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    
    st.session_state.last_price = current_price

    if st.session_state.system_halted:
        st.error(f"🚨 触发熔断保护！检测到波动异常。")
    else:
        # 数据分析
        df = get_analysis()
        last = df.iloc[-1]
        
        # 信号逻辑 (基于你训练脚本的特征)
        score = 0
        if last['close'] > last['ma20']: score += 25
        if last['hist'] > 0: score += 25
        if last['adx'] > 25: score += 25
        if 45 < last['rsi'] < 65: score += 25

        # 布局展示
        col1, col2, col3 = st.columns(3)
        col1.metric("ETH Price (Bybit)", f"${current_price}")
        col2.metric("Trend Score", f"{score} pt")
        col3.metric("RSI (14)", f"{round(last['rsi'], 2)}")

        # 信号预警
        if score >= 75:
            st.success("🎯 **建议方向：LONG (多)**")
        elif score <= 25:
            st.error("🎯 **建议方向：SHORT (空)**")
        else:
            st.info("📊 市场震荡中，等待高强度动能...")

        # K线图
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df['timestamp'], unit='ms'),
            open=df['open'], high=df['high'], low=df['low'], close=df['close']
        )])
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ 连接异常: {e}")
