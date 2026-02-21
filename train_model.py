import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =============================
# 1. 核心参数与 UI 配置
# =============================
SYMBOL = "ETH/USDT:USDT"  # Bybit 永续合约格式
REFRESH_MS = 1000        # 1秒刷新
CIRCUIT_BREAKER_PCT = 0.005 

st.set_page_config(layout="wide", page_title="ETH 100x Pro (Bybit)", page_icon="📈")
st_autorefresh(interval=REFRESH_MS, key="bybit_monitor")

# =============================
# 2. 交易所初始化 (同步你的 train_model 逻辑)
# =============================
@st.cache_resource
def get_exchange():
    return ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'} # 使用线性合约
    })

exchange = get_exchange()

# 状态管理
if 'last_price' not in st.session_state: st.session_state.last_price = 0
if 'system_halted' not in st.session_state: st.session_state.system_halted = False

# =============================
# 3. 算法逻辑 (集成你的技术指标)
# =============================
def get_analysis_data():
    # 获取 5m 数据
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 匹配你 train_model.py 中的指标
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ma20'] = ta.sma(df['close'], length=20)
    df['ma60'] = ta.sma(df['close'], length=60)
    macd = ta.macd(df['close'])
    df['hist'] = macd['MACDh_12_26_9']
    df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    
    return df

# =============================
# 4. 实时仪表盘渲染
# =============================
st.title("🛡️ ETH 100x Bybit Pro 监控系统")

if st.sidebar.button("🔌 重置系统"):
    st.session_state.system_halted = False
    st.session_state.last_price = 0

try:
    # 获取最新价
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = ticker['last']
    
    # 极速熔断检测
    if st.session_state.last_price != 0:
        change = abs(current_price - st.session_state.last_price) / st.session_state.last_price
        if change > CIRCUIT_BREAKER_PCT:
            st.session_state.system_halted = True
    
    st.session_state.last_price = current_price
    
    if st.session_state.system_halted:
        st.error(f"🚨 触发熔断保护！检测到波动异常。")
    else:
        # 数据分析
        df = get_analysis_data()
        last_row = df.iloc[-1]
        
        # 简单的评分逻辑 (示例)
        score = 0
        if last_row['close'] > last_row['ma20']: score += 25
        if last_row['hist'] > 0: score += 25
        if last_row['adx'] > 25: score += 25
        if 40 < last_row['rsi'] < 60: score += 25

        # 显示看板
        m1, m2, m3 = st.columns(3)
        m1.metric("ETH Bybit Price", f"${current_price}")
        m2.metric("Trend Score", f"{score} pt")
        m3.metric("RSI (14)", f"{round(last_row['rsi'], 2)}")

        # 信号判定
        if score >= 75:
            st.success("🎯 **AI 建议：多单 (LONG)**")
        elif score <= 25:
            st.error("🎯 **AI 建议：空单 (SHORT)**")
        else:
            st.info("📊 市场震荡中，等待高强度动能...")

        # 绘图
        fig = go.Figure(data=[go.Candlestick(
            x=pd.to_datetime(df['timestamp'], unit='ms'),
            open=df['open'], high=df['high'], low=df['low'], close=df['close']
        )])
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"连接异常: {e}")
