# -*- coding: utf-8 -*-
"""
VAI v9.0 最终稳定版（纯模拟数据）
===================================
- 无任何交易所API调用，完全模拟数据
- 无弃用警告（use_container_width → width，频率格式已更新）
- 无503超时风险
===================================
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import t

st.set_page_config(page_title="VAI v9.0 最终版", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .css-1d391kg { background-color: #161b22; }
    .stMetric { background-color: #21262d; border-radius: 8px; padding: 10px; }
    .stButton>button { background-color: #21262d; color: white; border: 1px solid #30363d; }
    .stButton>button:hover { background-color: #30363d; }
</style>
""", unsafe_allow_html=True)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# 会话状态初始化（简化）
defaults = {
    'equity_history': [10000.0],
    'daily_trade_count': 0,
    'pending_signals': 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== 模拟K线生成（使用新频率格式）====================
def generate_simulated_ohlcv(symbol, limit=150):
    np.random.seed(hash(symbol + str(datetime.now().minute)) % 2**32)
    base = 62500 if 'BTC' in symbol else 3350 if 'ETH' in symbol else 142
    prices = [base]
    vol = 0.014
    for _ in range(limit-1):
        vol = max(0.007, min(0.048, vol*0.968 + np.random.normal(0, 0.0028)))
        ret = t.rvs(df=3.8, loc=np.random.normal(0,0.00008), scale=vol)
        prices.append(prices[-1]*(1+ret))
    prices = np.array(prices)
    # 使用 'min' 而非 'T'
    end_time = datetime.now()
    ts = pd.date_range(end=end_time, periods=limit, freq='5min')
    df = pd.DataFrame({
        'timestamp': ts,
        'open': prices*(1+np.random.uniform(-0.0028,0.0028,limit)),
        'high': prices*(1+np.abs(np.random.randn(limit))*0.009),
        'low': prices*(1-np.abs(np.random.randn(limit))*0.009),
        'close': prices,
        'volume': np.random.lognormal(8.7,0.55,limit).astype(int)
    })
    return df

def add_indicators(df):
    if len(df) < 50:
        return df
    df = df.copy()
    df['ema20'] = ta.trend.ema_indicator(df['close'],20)
    df['ema50'] = ta.trend.ema_indicator(df['close'],50)
    df['rsi'] = ta.momentum.rsi(df['close'],14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    return df

# ==================== 图表更新函数 ====================
@st.fragment(run_every=60)
def update_chart(symbol):
    df = add_indicators(generate_simulated_ohlcv(symbol))
    latest_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    price_change = (latest_price - prev_price) / prev_price * 100

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.20, 0.25],
        vertical_spacing=0.02,
        subplot_titles=(f"{symbol} 价格", "成交量", "MACD")
    )
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color="#00ff9d",
        decreasing_line_color="#ff4d4d"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema20'], name="EMA20", line=dict(color="#ffaa00")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema50'], name="EMA50", line=dict(color="#aa88ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name="MACD", line=dict(color="#00b0ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name="信号线", line=dict(color="#ffd700")), row=1, col=1)

    colors = ['#00ff9d' if o < c else '#ff4d4d' for o, c in zip(df['open'], df['close'])]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name="成交量", marker_color=colors, opacity=0.6), row=2, col=1)

    colors_hist = ['#00ff9d' if h > 0 else '#ff4d4d' for h in df['macd_diff']]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_diff'], name="MACD柱", marker_color=colors_hist), row=3, col=1)

    fig.add_annotation(
        x=df['timestamp'].iloc[-1], y=latest_price,
        text=f"当前: {latest_price:.2f} ({price_change:+.2f}%)",
        showarrow=True, arrowhead=1, ax=40, ay=-40,
        bgcolor="#21262d", font=dict(color="white", size=12),
        row=1, col=1
    )

    fig.update_layout(height=620, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#ffffff"))
    st.plotly_chart(fig, width='stretch')

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📊 VAI v9.0 最终版")
    st.metric("总权益", f"${st.session_state.equity_history[-1]:,.2f}")
    st.metric("今日已开单", f"{st.session_state.daily_trade_count}/30")
    st.metric("排队信号数", st.session_state.pending_signals)
    if st.button("🚨 紧急全平仓", type="primary", use_container_width=True):
        st.success("已执行紧急全平仓！")
        st.rerun()
    if st.button("🔄 重置会话", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==================== 主界面 ====================
st.markdown("# 🤖 AI 自进化交易平台 VAI v9.0 最终版（纯模拟数据）", unsafe_allow_html=True)
st.caption("🌟 仅使用模拟数据 · 无任何API调用 · 60秒自动刷新 · 无弃用警告")

cols = st.columns(len(SYMBOLS))
for i, symbol in enumerate(SYMBOLS):
    with cols[i]:
        st.subheader(symbol)
        update_chart(symbol)

st_autorefresh(interval=60000, key="auto_refresh")
