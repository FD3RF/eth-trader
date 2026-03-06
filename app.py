import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(page_title="ETH 合约AI播报", layout="wide")
st.title("📊 ETH 合约5分钟量价AI播报")

# ======= 加载本地CSV（如果有）=======
def load_data():
    try:
        df = pd.read_csv("ETHUSDT_5m_last_90days.csv")
        df["open_time"] = pd.to_datetime(df["open_time"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except:
        return None

# ======= 如果没有CSV → 生成示例数据 =======
def load_sample_data():
    n = 200
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = np.cumsum(np.random.randn(n)) + 2000
    high = price + np.random.rand(n) * 5
    low = price - np.random.rand(n) * 5
    open_ = price + np.random.randn(n)
    close = price
    volume = np.abs(np.random.randn(n) * 1000)

    return pd.DataFrame({
        "open_time": rng,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })

# ======= 数据加载 =======
df = load_data()
if df is None:
    df = load_sample_data()
    st.warning("未找到CSV，使用示例数据")

# ======= 信号逻辑 =======
def signal_logic(df):
    if df is None or df.empty or len(df) < 2:
        return "no", "数据不足"

    last = df.iloc[-1]
    prev_vol = df["volume"].iloc[-6:-1].mean() if len(df) > 6 else df["volume"].mean()

    is_low_vol = last["volume"] < (prev_vol * 0.6 if prev_vol > 0 else 1)
    is_high_vol = last["volume"] > (prev_vol * 1.5 if prev_vol > 0 else 1)

    recent_high = df["high"].tail(20).max()
    recent_low = df["low"].tail(20).min()
    close = last["close"]
    low = last["low"]
    high = last["high"]

    if is_low_vol and low >= recent_low:
        return "observe", "缩量回踩低点不破"
    if is_high_vol and close > df["high"].iloc[-2]:
        return "buy", "放量突破前高"
    if is_low_vol and high <= recent_high:
        return "observe", "缩量反弹高点不破"
    if is_high_vol and close < df["low"].iloc[-2]:
        return "sell", "放量跌破前低"

    return "observe", "等待信号"

# ======= 策略信号 =======
signal, reason = signal_logic(df)

# ======= K线图 =======
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["open_time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
))

# 关键位
fig.add_trace(go.Scatter(
    x=[df["open_time"].iloc[0], df["open_time"].iloc[-1]],
    y=[df["high"].tail(20).max(), df["high"].tail(20).max()],
    mode="lines",
    name="前高"
))
fig.add_trace(go.Scatter(
    x=[df["open_time"].iloc[0], df["open_time"].iloc[-1]],
    y=[df["low"].tail(20).min(), df["low"].tail(20).min()],
    mode="lines",
    name="前低"
))

st.plotly_chart(fig, use_container_width=True)

# ======= 播报 =======
st.subheader("🤖 AI 播报")
st.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"信号：{signal}")
st.write(f"原因：{reason}")

if signal == "buy":
    st.success("📈 多单信号")
elif signal == "sell":
    st.error("📉 空单信号")
else:
    st.info("⏳ 观察区")

# ======= 数据表 =======
st.subheader("📋 最近K线")
st.dataframe(df.tail())
