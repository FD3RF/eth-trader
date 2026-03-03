"""
小利润策略终极版
- OKX真实数据
- 多周期共振
- 假突破过滤
- 动态止盈
- 回测
- 进场提示
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🚀 小利润策略终极版")

SYMBOL = "ETH-USDT-SWAP"

# -------------------------
# 数据获取（OKX）
# -------------------------
def fetch_kline(bar="5m", limit=300):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": bar, "limit": limit}
    r = requests.get(url, params=params, timeout=5)
    data = r.json()
    if data.get("code") == "0":
        df = pd.DataFrame(data["data"], columns=[
            "ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    return None

df = fetch_kline()
if df is None or df.empty:
    st.error("无法获取OKX数据")
    st.stop()

# -------------------------
# 多周期数据（15m）
# -------------------------
df_15m = fetch_kline("15m", 200)
if df_15m is not None:
    df_15m["EMA20"] = ta.trend.ema_indicator(df_15m["close"], window=20)
    df_15m["ADX"] = ta.trend.adx(df_15m["high"], df_15m["low"], df_15m["close"], window=14)

# -------------------------
# 指标计算
# -------------------------
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=12)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=50)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

df = df.dropna().reset_index(drop=True)
latest = df.iloc[-1]
prev = df.iloc[-2]

# -------------------------
# 多周期共振
# -------------------------
tf_ok = True
if df_15m is not None and len(df_15m) > 1:
    latest_15 = df_15m.iloc[-1]
    prev_15 = df_15m.iloc[-2]
    tf_ok = latest_15["EMA20"] > prev_15["EMA20"]

# -------------------------
# 假突破过滤
# -------------------------
def fake_breakout(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]

    # 假突破特征：影线大、实体小
    if upper_shadow > body * 2 or lower_shadow > body * 2:
        return True
    return False

breakout_fake = fake_breakout(df)

# -------------------------
# 趋势与结构
# -------------------------
trend_up = latest["EMA_fast"] > latest["EMA_slow"] and latest["ADX"] > 20
structure_up = latest["low"] > prev["low"]
distance_ok = abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * 0.7

signal = None
if trend_up and structure_up and distance_ok and tf_ok and not breakout_fake:
    if latest["close"] > latest["EMA_fast"]:
        signal = "多"

# -------------------------
# 动态止损止盈（小利润）
# -------------------------
stop_loss = latest["close"] - latest["ATR"] * 0.6
take_profit = latest["close"] + latest["ATR"] * 0.8

# -------------------------
# 回测（100模拟）
# -------------------------
def backtest(df, days=7):
    end = df["ts"].max()
    start = end - timedelta(days=days)
    data = df[df["ts"] >= start].copy()

    balance = 100
    trades = []

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i-1]

        # 信号同上
        trend = row["EMA_fast"] > row["EMA_slow"] and row["ADX"] > 20
        structure = row["low"] > prev["low"]
        distance = abs(row["close"] - row["EMA_fast"]) < row["ATR"] * 0.7
        if trend and structure and distance:
            entry = row["close"]
            stop = entry - row["ATR"] * 0.6
            tp = entry + row["ATR"] * 0.8
            qty = balance * 0.02 / abs(entry - stop)

            # 模拟下一根
            next_row = data.iloc[i]
            if next_row["low"] <= stop:
                pnl = (stop - entry) * qty
            elif next_row["high"] >= tp:
                pnl = (tp - entry) * qty
            else:
                pnl = 0

            balance += pnl
            trades.append(pnl)

    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    return balance, win_rate, len(trades)

if st.sidebar.checkbox("回测"):
    bal, wr, n = backtest(df)
    st.metric("回测资金", f"{bal:.2f}")
    st.metric("胜率", f"{wr:.1f}%")
    st.metric("交易数", n)

# -------------------------
# UI提示
# -------------------------
st.subheader("📡 信号")
if signal:
    st.success(f"进场信号：{signal}")
    st.write(f"止损：{stop_loss:.2f}")
    st.write(f"止盈：{take_profit:.2f}")
else:
    st.info("无信号")

# -------------------------
# 图表
# -------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], name="EMA_fast"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], name="EMA_slow"))
st.plotly_chart(fig, use_container_width=True)
