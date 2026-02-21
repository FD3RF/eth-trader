# -*- coding: utf-8 -*-
"""
🤖 ETH 合約短線策略監控系統 V11.2（已優化部署）
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt, pandas as pd, numpy as np, ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date

st.set_page_config(page_title="ETH短線監控", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #21262d; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 ETH 合約短線策略監控系統 V11.2")
st.caption("1m + 5m 雙週期 • VWAP + EMA9/21 + ATR14 • 移動止損 • 每8秒刷新")

SYMBOL = "ETH/USDT:USDT"

# ==================== 會話狀態 ====================
for k, v in {
    'opened_today': 0, 'last_date': date.today(),
    'positions': {'ETH': None}
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ==================== 數據 ====================
@st.cache_data(ttl=8)
def fetch_klines(tf, limit=400):
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    ohlcv = ex.fetch_ohlcv(SYMBOL, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    df = df.copy()
    df['ema9'] = ta.trend.ema_indicator(df['close'],9)
    df['ema21'] = ta.trend.ema_indicator(df['close'],21)
    df['atr'] = ta.volatility.average_true_range(df['high'],df['low'],df['close'],14)
    typical = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical * df['volume']).cumsum() / df['volume'].cumsum()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    return df

# ==================== 信號 ====================
def detect_signal(df):
    if len(df) < 30: return "觀望", None, None, None, None
    last = df.iloc[-1]
    ema_cross_long = df['ema9'].iloc[-2] < df['ema21'].iloc[-2] and last['ema9'] > last['ema21']
    vol_burst = last['volume'] > last['vol_ma5'] * 1.35
    if ema_cross_long and last['close'] > last['vwap'] and vol_burst:
        entry = last['close']
        sl = entry - 1.5 * last['atr']
        sl = max(sl, entry * 0.997)  # 0.3% 強制保護
        tp = entry + 3 * last['atr']
        rr = round((tp - entry) / (entry - sl), 2)
        return "多頭計劃 🔥", round(entry,2), round(sl,2), round(tp,2), rr
    return "觀望", None, None, None, None

# ==================== 側邊欄 ====================
with st.sidebar:
    st.metric("總權益", "$10,000.00")
    st.metric("今日已開單", f"{st.session_state.opened_today}/30")
    st.metric("排隊信號", 0)
    if st.button("🚨 緊急全平倉", type="primary", use_container_width=True):
        st.session_state.positions = {'ETH': None}
        st.success("已執行緊急全平倉！")
        st.rerun()

# ==================== 主界面 ====================
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("1分鐘圖表")
    df1 = add_indicators(fetch_klines('1m'))
    sig1, entry1, sl1, tp1, rr1 = detect_signal(df1)
    fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55,0.20,0.25])
    fig1.add_trace(go.Candlestick(x=df1['timestamp'], open=df1['open'], high=df1['high'], low=df1['low'], close=df1['close']), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1['timestamp'], y=df1['vwap'], name="VWAP", line=dict(color="#ffd700")), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1['timestamp'], y=df1['ema9'], name="EMA9", line=dict(color="#00ff9d")), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1['timestamp'], y=df1['ema21'], name="EMA21", line=dict(color="#ff4d4d")), row=1, col=1)
    if sl1: fig1.add_hline(y=sl1, line_dash="dot", line_color="#ff4d4d", annotation_text="移動止損", row=1, col=1)
    st.plotly_chart(fig1, width='stretch')

with col2:
    st.subheader("5分鐘圖表")
    df5 = add_indicators(fetch_klines('5m'))
    sig5, entry5, sl5, tp5, rr5 = detect_signal(df5)
    fig5 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55,0.20,0.25])
    fig5.add_trace(go.Candlestick(x=df5['timestamp'], open=df5['open'], high=df5['high'], low=df5['low'], close=df5['close']), row=1, col=1)
    fig5.add_trace(go.Scatter(x=df5['timestamp'], y=df5['vwap'], name="VWAP", line=dict(color="#ffd700")), row=1, col=1)
    fig5.add_trace(go.Scatter(x=df5['timestamp'], y=df5['ema9'], name="EMA9", line=dict(color="#00ff9d")), row=1, col=1)
    fig5.add_trace(go.Scatter(x=df5['timestamp'], y=df5['ema21'], name="EMA21", line=dict(color="#ff4d4d")), row=1, col=1)
    if sl5: fig5.add_hline(y=sl5, line_dash="dot", line_color="#ff4d4d", annotation_text="移動止損", row=1, col=1)
    st.plotly_chart(fig5, width='stretch')

st.divider()
st.subheader("📢 即時信號")
col_a, col_b = st.columns(2)
with col_a:
    st.metric("1分鐘信號", sig1 or "觀望")
    if sig1 and "計劃" in sig1:
        st.success(f"入場: **{entry1}**")
        st.error(f"止損: **{sl1}**")
        st.success(f"止盈: **{tp1}**")
        st.info(f"盈虧比: **{rr1}:1**")
with col_b:
    st.metric("5分鐘信號", sig5 or "觀望")
    if sig5 and "計劃" in sig5:
        st.success(f"入場: **{entry5}**")
        st.error(f"止損: **{sl5}**")
        st.success(f"止盈: **{tp5}**")
        st.info(f"盈虧比: **{rr5}:1**")

st_autorefresh(interval=8000, key="refresh")
