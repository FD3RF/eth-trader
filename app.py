"""
Streamlit 小利润策略（模拟交易 + 回测）
可跑版本：OKX 5分钟数据
资金：模拟 100 USDT
策略：趋势 + 小回调 + 有利润进场
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime, timedelta

# =========================
# 配置
# =========================
SYMBOL = "ETH-USDT-SWAP"
INITIAL_BALANCE = 100.0
ATR_MULT_STOP = 0.6
ATR_MULT_TP = 0.8

st.set_page_config(layout="wide")
st.title("📈 小利润策略（模拟 + 回测）")

# =========================
# 数据获取
# =========================
@st.cache_data(ttl=10)
def fetch_5m(limit=300):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": limit}
    r = requests.get(url, params=params)
    data = r.json().get("data", [])
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values("ts").reset_index(drop=True)

df = fetch_5m()
if df.empty:
    st.error("无法获取数据")
    st.stop()

# =========================
# 指标
# =========================
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=12)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=50)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
df.dropna(inplace=True)
latest = df.iloc[-1]
prev = df.iloc[-2]

# =========================
# 小利润信号
# =========================
def small_profit_signal(latest, prev):
    trend_up = latest["EMA_fast"] > latest["EMA_slow"] and latest["ADX"] > 20
    structure_up = latest["low"] > prev["low"]          # 低点抬高
    pullback = latest["close"] > latest["EMA_fast"]      # 回到快线以上
    distance = abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * 0.7

    if trend_up and structure_up and pullback and distance:
        body = abs(latest["close"] - latest["open"])
        lower_shadow = min(latest["close"], latest["open"]) - latest["low"]
        if lower_shadow > body * 1.2 or latest["close"] > latest["open"]:
            return "多"

    trend_down = latest["EMA_fast"] < latest["EMA_slow"] and latest["ADX"] > 20
    structure_down = latest["high"] < prev["high"]
    pullback_down = latest["close"] < latest["EMA_fast"]
    distance_down = abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * 0.7

    if trend_down and structure_down and pullback_down and distance_down:
        body = abs(latest["close"] - latest["open"])
        upper_shadow = latest["high"] - max(latest["close"], latest["open"])
        if upper_shadow > body * 1.2 or latest["close"] < latest["open"]:
            return "空"

    return None

signal = small_profit_signal(latest, prev)

# =========================
# 模拟账户
# =========================
if "balance" not in st.session_state:
    st.session_state.balance = INITIAL_BALANCE
    st.session_state.trades = []

def open_trade(direction, price, atr):
    if direction == "多":
        stop = price - atr * ATR_MULT_STOP
        tp = price + atr * ATR_MULT_TP
    else:
        stop = price + atr * ATR_MULT_STOP
        tp = price - atr * ATR_MULT_TP

    risk = st.session_state.balance * 0.02
    dist = abs(price - stop)
    qty = risk / dist if dist > 0 else 0.01
    qty = max(round(qty, 2), 0.01)

    st.session_state.trades.append({
        "time": latest["ts"],
        "dir": direction,
        "entry": price,
        "stop": stop,
        "tp": tp,
        "qty": qty,
        "status": "open"
    })

def close_positions(price):
    for t in st.session_state.trades:
        if t["status"] != "open":
            continue
        if t["dir"] == "多":
            if price <= t["stop"]:
                pnl = (t["stop"] - t["entry"]) * t["qty"]
                st.session_state.balance += pnl
                t["status"] = "stop"
            elif price >= t["tp"]:
                pnl = (t["tp"] - t["entry"]) * t["qty"]
                st.session_state.balance += pnl
                t["status"] = "tp"
        else:
            if price >= t["stop"]:
                pnl = (t["entry"] - t["stop"]) * t["qty"]
                st.session_state.balance += pnl
                t["status"] = "stop"
            elif price <= t["tp"]:
                pnl = (t["entry"] - t["tp"]) * t["qty"]
                st.session_state.balance += pnl
                t["status"] = "tp"

# 平仓检测
close_positions(latest["close"])

# 开仓
if signal:
    open_trade(signal, latest["close"], latest["ATR"])

# =========================
# 面板
# =========================
st.subheader("📊 状态")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("趋势", "多头" if latest["EMA_fast"] > latest["EMA_slow"] else "空头")
with col2:
    st.metric("ADX", f"{latest['ADX']:.1f}")
with col3:
    st.metric("信号", signal or "无")

st.metric("模拟余额", f"{st.session_state.balance:.2f} USDT")

# =========================
# 回测（简化）
# =========================
def backtest(df):
    balance = INITIAL_BALANCE
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        sig = small_profit_signal(row, prev)
        atr = row["ATR"]
        price = row["close"]

        # 开仓
        if sig == "多":
            stop = price - atr * ATR_MULT_STOP
            tp = price + atr * ATR_MULT_TP
        elif sig == "空":
            stop = price + atr * ATR_MULT_STOP
            tp = price - atr * ATR_MULT_TP
        else:
            continue

        risk = balance * 0.02
        dist = abs(price - stop)
        qty = risk / dist if dist > 0 else 0.01
        qty = max(round(qty, 2), 0.01)

        # 模拟立即平仓（5分钟后检测）
        future = df.iloc[i]
        if sig == "多":
            if future["low"] <= stop:
                pnl = (stop - price) * qty
            elif future["high"] >= tp:
                pnl = (tp - price) * qty
            else:
                pnl = 0
        else:
            if future["high"] >= stop:
                pnl = (price - stop) * qty
            elif future["low"] <= tp:
                pnl = (price - tp) * qty
            else:
                pnl = 0

        balance += pnl
        trades.append({"pnl": pnl})

    win = len([t for t in trades if t["pnl"] > 0])
    return {
        "balance": balance,
        "trades": len(trades),
        "win_rate": win / len(trades) * 100 if trades else 0
    }

if st.button("回测"):
    with st.spinner("回测中..."):
        res = backtest(df)
    st.success(f"""
    回测完成：
    - 终余额：{res['balance']:.2f}
    - 交易数：{res['trades']}
    - 胜率：{res['win_rate']:.1f}%
    """)

# =========================
# 图表
# =========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], name="EMA12"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], name="EMA50"))
st.plotly_chart(fig, use_container_width=True)
