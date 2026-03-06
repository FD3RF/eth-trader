import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(page_title="ETH 合约AI播报(本地数据版)", layout="wide")
st.title("📊 ETH 合约5分钟量价AI播报 - 本地示例数据")

# ======= 生成/加载本地示例K线数据 =======
def load_sample_data():
    # 如果有 CSV，可在这里加载：
    # df = pd.read_csv("your_file.csv")
    # return df

    # 没有外部数据 → 生成模拟5分钟数据
    n = 200
    rng = pd.date_range("2024-01-01", periods=n, freq="5T")
    price = np.cumsum(np.random.randn(n)) + 2000
    high = price + np.random.rand(n) * 5
    low = price - np.random.rand(n) * 5
    open_ = price + np.random.randn(n)
    close = price
    volume = np.abs(np.random.randn(n) * 1000)

    df = pd.DataFrame({
        "open_time": rng,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    return df

# ======= 信号逻辑 =======
def signal_logic(df):
    if df is None or df.empty or len(df) < 2:
        return False, False, "数据不足", None, None

    last = df.iloc[-1]
    prev_vol = df["volume"].iloc[-6:-1].mean() if len(df) > 6 else df["volume"].mean()

    is_low_vol = last["volume"] < (prev_vol * 0.6 if prev_vol > 0 else 1)
    is_high_vol = last["volume"] > (prev_vol * 1.5 if prev_vol > 0 else 1)

    recent_high = df["high"].tail(20).max()
    recent_low = df["low"].tail(20).min()
    close = last["close"]
    low = last["low"]
    high = last["high"]

    buy = False
    sell = False
    motto = "等待信号"

    if is_low_vol and low >= recent_low:
        motto = "缩量回踩，低点不破 → 观察"
    if is_high_vol and close > df["high"].iloc[-2]:
        buy = True
        motto = "放量起涨，突破前高 → 做多"
    if is_low_vol and high <= recent_high:
        motto = "缩量反弹，高点不破 → 观察"
    if is_high_vol and close < df["low"].iloc[-2]:
        sell = True
        motto = "放量跌破前低 → 做空"

    return buy, sell, motto, recent_high, recent_low

# ======= 主流程 =======
df = load_sample_data()

buy, sell, motto, high, low = signal_logic(df)

# ======= K线图 =======
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["open_time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="ETH 5m"
))

# 关键位
fig.add_trace(go.Scatter(
    x=[df["open_time"].iloc[0], df["open_time"].iloc[-1]],
    y=[high, high],
    mode="lines",
    name="前高"
))
fig.add_trace(go.Scatter(
    x=[df["open_time"].iloc[0], df["open_time"].iloc[-1]],
    y=[low, low],
    mode="lines",
    name="前低"
))

st.plotly_chart(fig, use_container_width=True)

# ======= 播报 =======
st.subheader("🤖 AI 播报")
st.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"智能口诀：{motto}")

if buy:
    st.success("📈 多单信号（模拟）")
elif sell:
    st.error("📉 空单信号（模拟）")
else:
    st.info("⏳ 观察区")

# ======= 数据表 =======
st.subheader("📋 示例K线")
st.dataframe(df.tail())
