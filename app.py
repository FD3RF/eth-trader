import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="ETH 合约AI播报", layout="wide")
st.title("📊 ETH 合约5分钟量价AI播报(OKX数据源)")

# ======= 获取K线 =======
def get_klines():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": "ETH-USDT-SWAP", "bar": "5m", "limit": "100"}
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
    except Exception as e:
        st.error(f"请求失败: {e}")
        return pd.DataFrame()

    if "data" not in data or len(data["data"]) == 0:
        st.warning("OKX返回空数据")
        return pd.DataFrame()

    rows = data["data"]
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "volCcy", "volCcyQuote", "confirm"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df

# ======= 信号逻辑（安全版）=======
def signal_logic(df):
    # 数据保护：行数不足直接返回
    if df is None or df.empty or len(df) < 2:
        return False, False, "数据不足", None, None

    last = df.iloc[-1]
    if last is None:
        return False, False, "数据异常", None, None

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
df = get_klines()

if df is None or df.empty:
    st.warning("暂无数据，等待下一次刷新")
    st.stop()

# 信号
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
st.write(f"口诀：{motto}")

if buy:
    st.success("📈 多单信号")
elif sell:
    st.error("📉 空单信号")
else:
    st.info("⏳ 观察区")

# ======= 数据表 =======
st.subheader("📋 最新K线")
st.dataframe(df.tail())
