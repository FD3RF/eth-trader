import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ====================== 页面 ======================
st.set_page_config(page_title="AI量价策略", layout="wide")
st.title("📊 ETH 永续合约 5分钟 AI 量价策略")
st.caption("实时永续数据 · 量价口诀 · 策略播报")

# ====================== 获取真实永续数据 ======================
@st.cache_data(ttl=10)
def fetch_klines():
    try:
        exchange = ccxt.binanceusdm()
        ohlcv = exchange.fetch_ohlcv('ETH/USDT:USDT', timeframe='5m', limit=120)
        df = pd.DataFrame(ohlcv, columns=['open_time','open','high','low','close','volume'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

# ====================== 策略逻辑 ======================
def signal_logic(df):
    if df is None or df.empty or len(df) < 2:
        return "no", "数据不足"

    last = df.iloc[-1]
    prev = df.iloc[-51:-1]

    recent_high = prev['high'].max()
    recent_low = prev['low'].min()
    avg_vol = prev['volume'].mean()

    vol_ratio = last['volume'] / avg_vol if avg_vol > 0 else 1
    is_shrink = vol_ratio < 0.6
    is_expand = vol_ratio > 1.8

    near_low = abs(last['low'] - recent_low) / recent_low < 0.003
    near_high = abs(last['high'] - recent_high) / recent_high < 0.003
    broke_high = last['close'] > recent_high
    broke_low = last['low'] < recent_low * 0.997

    drop_pct = (last['open'] - last['low']) / last['open']

    # 量价口诀
    if is_expand and broke_high:
        return "buy", "放量起涨，突破前高，直接开多"
    if is_expand and broke_low:
        return "sell", "放量下跌，跌破前低，直接开空"
    if is_expand and near_low and drop_pct > 0.012:
        return "buy", "放量暴跌低点不破，这是机会"
    if is_expand and near_high:
        return "sell", "放量急涨顶部不破，这是机会"
    if is_shrink and near_low:
        return "observe", "缩量回踩低点不破，准备动手"
    if is_shrink and near_high:
        return "observe", "缩量反弹高点不破，准备动手"
    if is_shrink:
        return "observe", "缩量横盘，等待放量方向"
    return "observe", "量能不明"

# ====================== 图表 ======================
def plot_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(
        x=df['open_time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="K线"
    ), row=1, col=1)

    colors = ['#00ff9d' if o < c else '#ff0066' for o, c in zip(df['open'], df['close'])]
    fig.add_trace(go.Bar(
        x=df['open_time'],
        y=df['volume'],
        marker_color=colors,
        name="成交量"
    ), row=2, col=1)

    fig.update_layout(
        height=620,
        template="plotly_dark",
        title="ETH 永续 5分钟K线"
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================== 主流程 ======================
df = fetch_klines()

if df.empty:
    st.warning("暂无行情数据")
    st.stop()

signal, reason = signal_logic(df)
last = df.iloc[-1]

# 图表
plot_chart(df)

# 播报
st.subheader("🤖 AI 播报")
st.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"信号：{signal}")
st.write(f"原因：{reason}")
st.write(f"当前价：{last['close']:.2f} USDT")

if signal == "buy":
    st.success("📈 多单倾向")
elif signal == "sell":
    st.error("📉 空单倾向")
else:
    st.info("⏳ 观察区")

st.caption("数据来自 Binance 永续合约 · 量价策略")
