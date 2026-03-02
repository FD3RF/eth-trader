"""
5分钟趋势合约系统（EMA+ADX+回调入场）
- 多头趋势：EMA20 > EMA60 且 ADX>25
- 入场：价格回调至EMA20附近出现反转K线
- 出场：移动止损（EMA20）或固定盈亏比
"""
import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime
import requests
import time

st.set_page_config(layout="wide")
st.title("📈 5分钟趋势合约系统（实战版）")

SYMBOL = "ETH-USDT-SWAP"

@st.cache_data(ttl=5)
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}
    try:
        r = requests.get(url, timeout=5, params=params)
        data = r.json()["data"]
    except:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df

df = get_data()
if df.empty:
    st.stop()

# 指标
df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], 14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
df = df.dropna().reset_index(drop=True)

latest = df.iloc[-1]
prev = df.iloc[-2]

# 趋势判断
trend = None
if latest["EMA20"] > latest["EMA60"] and latest["ADX"] > 25:
    trend = "多"
elif latest["EMA20"] < latest["EMA60"] and latest["ADX"] > 25:
    trend = "空"

# 回调入场信号
signal = None
if trend == "多":
    # 价格回踩EMA20附近（±0.2%）且前一根K线收盘在EMA20下方（制造恐慌）
    if abs(latest["close"] - latest["EMA20"]) / latest["EMA20"] < 0.002 and prev["close"] < prev["EMA20"]:
        # K线确认：下影线较长或阳线
        body = abs(latest["close"] - latest["open"])
        lower_shadow = min(latest["close"], latest["open"]) - latest["low"]
        if lower_shadow > body * 1.5 or latest["close"] > latest["open"]:
            signal = "多"
elif trend == "空":
    if abs(latest["close"] - latest["EMA20"]) / latest["EMA20"] < 0.002 and prev["close"] > prev["EMA20"]:
        body = abs(latest["close"] - latest["open"])
        upper_shadow = latest["high"] - max(latest["close"], latest["open"])
        if upper_shadow > body * 1.5 or latest["close"] < latest["open"]:
            signal = "空"

# 图表
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], line=dict(color="blue"), name="EMA20"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], line=dict(color="orange"), name="EMA60"))
st.plotly_chart(fig, use_container_width=True)

st.write(f"趋势: {trend} | ADX: {latest['ADX']:.1f} | 最新价: {latest['close']:.2f}")
if signal:
    st.success(f"信号: {signal} (回调入场)")
else:
    st.info("无信号")
