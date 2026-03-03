import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🚀 小利润策略（工程级）")

SYMBOL = "ETH-USDT-SWAP"

# ----------------------------
# 获取OKX数据
# ----------------------------
def fetch_ohlcv(bar="5m", limit=200):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": bar, "limit": limit}
    r = requests.get(url, params=params, timeout=5)
    data = r.json()
    if data.get("code") != "0":
        return None
    df = pd.DataFrame(data["data"], columns=[
        "ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values("ts").reset_index(drop=True)

# ----------------------------
# 小利润信号函数
# ----------------------------
def calc_signal(df):
    df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=12)
    df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 多周期共振（5分钟趋势 + 15分钟趋势）
    df15 = fetch_ohlcv("15m", 100)
    if df15 is None or df15.empty:
        tf_ok = True
    else:
        df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
        tf_ok = df15.iloc[-1]["EMA20"] > df15.iloc[-2]["EMA20"]

    trend_up = latest["EMA_fast"] > latest["EMA_slow"] and latest["ADX"] > 20 and tf_ok
    trend_down = latest["EMA_fast"] < latest["EMA_slow"] and latest["ADX"] > 20 and tf_ok

    signal = None

    # ----------------------------
    # 小利润多单
    # ----------------------------
    if trend_up:
        structure_ok = latest["low"] > prev["low"]  # 低点抬高
        pullback_done = latest["close"] > latest["EMA_fast"]  # 回到快线之上
        distance_ok = abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * 0.7

        if structure_ok and pullback_done and distance_ok:
            body = abs(latest["close"] - latest["open"])
            lower_shadow = min(latest["close"], latest["open"]) - latest["low"]
            if lower_shadow > body * 1.2 or latest["close"] > latest["open"]:
                signal = "多"

    # ----------------------------
    # 小利润空单
    # ----------------------------
    if trend_down:
        structure_ok = latest["high"] < prev["high"]
        pullback_done = latest["close"] < latest["EMA_fast"]
        distance_ok = abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * 0.7

        if structure_ok and pullback_done and distance_ok:
            body = abs(latest["close"] - latest["open"])
            upper_shadow = latest["high"] - max(latest["close"], latest["open"])
            if upper_shadow > body * 1.2 or latest["close"] < latest["open"]:
                signal = "空"

    return signal, latest

# ----------------------------
# 回测
# ----------------------------
def backtest(days=7):
    end = datetime.now()
    start = end - timedelta(days=days)

    df = fetch_ohlcv()
    df = df[df["ts"] >= start].reset_index(drop=True)
    if len(df) < 50:
        return None

    balance = 1000
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        sub = df.iloc[:i+1]
        signal, latest = calc_signal(sub)

        if not signal:
            continue

        atr = latest["ATR"]
        entry = latest["close"]

        if signal == "多":
            stop = entry - atr * 0.6
            tp = entry + atr * 0.8
            if row["low"] <= stop:
                pnl = (stop - entry)
            elif row["high"] >= tp:
                pnl = (tp - entry)
            else:
                continue
        else:
            stop = entry + atr * 0.6
            tp = entry - atr * 0.8
            if row["high"] >= stop:
                pnl = (entry - stop)
            elif row["low"] <= tp:
                pnl = (entry - tp)
            else:
                continue

        balance += pnl
        trades.append(pnl)

    win = sum(1 for x in trades if x > 0)
    loss = len(trades) - win
    win_rate = win / len(trades) * 100 if trades else 0

    return {
        "balance": balance,
        "trades": len(trades),
        "win_rate": win_rate,
        "profit": balance - 1000
    }

# ----------------------------
# Streamlit UI
# ----------------------------
df = fetch_ohlcv()
if df is None:
    st.error("无法获取数据")
    st.stop()

signal, latest = calc_signal(df)

# 信号提示
if signal:
    st.success(f"📢 信号：{signal}")
else:
    st.info("无信号")

# 图表
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], name="EMA12"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], name="EMA50"))
st.plotly_chart(fig, use_container_width=True)

# 回测面板
st.subheader("📊 回测")
days = st.slider("回测天数", 3, 30, 7)
if st.button("开始回测"):
    with st.spinner("回测中..."):
        res = backtest(days)
    if res:
        st.metric("最终资金", f"{res['balance']:.2f}")
        st.metric("胜率", f"{res['win_rate']:.1f}%")
        st.metric("交易数", res["trades"])
        st.metric("利润", f"{res['profit']:+.2f}")
