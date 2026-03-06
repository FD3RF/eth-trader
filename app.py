import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="ETH 合约AI智能播报", layout="wide")

st.title("📊 ETH 合约5分钟量价AI播报（含真实成交量）")

# ======= 获取K线与合约成交量（Binance）=======
def get_klines():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "ETHUSDT",
        "interval": "5m",
        "limit": 100
    }
    resp = requests.get(url, params=params)
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)  # 合约真实成交量（基础成交量）

    return df

# ======= 止损止盈计算 =======
def risk_calc(entry, direction, recent_high, recent_low):
    """
    direction: 'long' or 'short'
    止损：关键位外扩1-2个价位
    止盈：1:1.5 盈亏比
    """
    if direction == "long":
        stop = recent_low - (recent_low * 0.001)  # 低点下方0.1%
        risk = entry - stop
        target = entry + risk * 1.5
    else:
        stop = recent_high + (recent_high * 0.001)
        risk = stop - entry
        target = entry - risk * 1.5

    return round(stop, 4), round(target, 4)

# ======= 量价与口诀信号 =======
def signal_logic(df):
    last = df.iloc[-1]
    prev_vol = df["volume"].iloc[-6:-1].mean()

    is_low_vol = last["volume"] < prev_vol * 0.6
    is_high_vol = last["volume"] > prev_vol * 1.5

    recent_high = df["high"].iloc[-20:].max()
    recent_low = df["low"].iloc[-20:].min()
    close = last["close"]
    low = last["low"]
    high = last["high"]

    buy = False
    sell = False
    motto = "等待信号"

    # 做多口诀
    if is_low_vol and low >= recent_low:
        motto = "缩量回踩，低点不破 → 观察"
    if is_high_vol and close > df["high"].iloc[-2]:
        buy = True
        motto = "放量起涨，突破前高 → 做多"

    # 做空口诀
    if is_low_vol and high <= recent_high:
        motto = "缩量反弹，高点不破 → 观察"
    if is_high_vol and close < df["low"].iloc[-2]:
        sell = True
        motto = "放量跌破前低 → 做空"

    return buy, sell, motto, recent_high, recent_low

# ======= 数据与信号 =======
df = get_klines()
buy, sell, motto, high, low = signal_logic(df)

entry_price = df.iloc[-1]["close"]
direction = "long" if buy else "short" if sell else None

stop = target = None
if direction:
    stop, target = risk_calc(entry_price, direction, high, low)

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

# ======= 播报区 =======
st.subheader("🤖 AI 智能播报")
st.write(f"最新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"智能口诀：{motto}")

if buy:
    st.success("📈 多单信号：放量突破 → 做多观察")
elif sell:
    st.error("📉 空单信号：放量跌破 → 做空观察")
else:
    st.info("⏳ 观察区：等待放量信号")

# ======= 止损止盈 =======
st.subheader("💰 风险与目标计算")
if direction:
    st.write(f"方向：{'做多' if direction=='long' else '做空'}")
    st.write(f"开仓价：{entry_price}")
    st.write(f"止损位：{stop}")
    st.write(f"止盈位：{target}")
    st.write("盈亏比：1 : 1.5（固定）")
else:
    st.write("暂无开仓信号，风险计算关闭")

# ======= 最新K线表 =======
st.subheader("📋 最新5根K线（含成交量）")
st.dataframe(df.tail())
