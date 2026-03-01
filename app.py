import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

# ===============================
# 配置
# ===============================
SYMBOL = "ETHUSDT"
INTERVAL_5M = "5m"
INTERVAL_4H = "4h"
LIMIT = 300
REFRESH_SECONDS = 10

st.set_page_config(layout="wide")

# ===============================
# 数据获取
# ===============================
def fetch_binance(symbol, interval, limit):
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbb","tbq","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open","high","low","close","volume"]] = \
        df[["open","high","low","close","volume"]].astype(float)
    return df

# ===============================
# 指标计算
# ===============================
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def MACD(df):
    ema12 = EMA(df["close"], 12)
    ema26 = EMA(df["close"], 26)
    macd = ema12 - ema26
    signal = EMA(macd, 9)
    return macd, signal

def KDJ(df, n=9):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    return k, d, j

def BOLL(df):
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    bandwidth = (upper - lower) / mid
    return upper, mid, lower, bandwidth

# ===============================
# 主逻辑
# ===============================
st.title("ETH 5分钟专业盯盘终端")

while True:

    df5 = fetch_binance(SYMBOL, INTERVAL_5M, LIMIT)
    df4 = fetch_binance(SYMBOL, INTERVAL_4H, 100)

    # 4H 趋势
    df4["ema30"] = EMA(df4["close"], 30)
    trend_long = df4["close"].iloc[-1] > df4["ema30"].iloc[-1]
    trend = "LONG ONLY" if trend_long else "SHORT ONLY"

    # 5M 指标
    df5["macd"], df5["signal"] = MACD(df5)
    df5["k"], df5["d"], df5["j"] = KDJ(df5)
    df5["upper"], df5["mid"], df5["lower"], df5["bandwidth"] = BOLL(df5)
    df5["ema7"] = EMA(df5["close"], 7)

    volume_mean5 = df5["volume"].rolling(5).mean()

    # 最新值
    macd_now = df5["macd"].iloc[-1]
    macd_prev = df5["macd"].iloc[-2]
    signal_now = df5["signal"].iloc[-1]
    signal_prev = df5["signal"].iloc[-2]

    k_now = df5["k"].iloc[-1]
    d_now = df5["d"].iloc[-1]
    k_prev = df5["k"].iloc[-2]
    d_prev = df5["d"].iloc[-2]

    volume_valid = df5["volume"].iloc[-1] > volume_mean5.iloc[-1]

    macd_cross_up = macd_prev < signal_prev and macd_now > signal_now and macd_now > 0
    kdj_cross_up = k_prev < d_prev and k_now > d_now

    range_market = df5["bandwidth"].iloc[-1] < 0.015
    near_lower = (df5["close"].iloc[-1] - df5["lower"].iloc[-1]) / df5["close"].iloc[-1] < 0.003

    long_signal = trend_long and macd_cross_up and kdj_cross_up and volume_valid and (not range_market or near_lower)

    # ===============================
    # UI 显示
    # ===============================
    col1, col2, col3 = st.columns(3)

    col1.metric("当前价格", round(df5["close"].iloc[-1],2))
    col2.metric("4H趋势", trend)
    col3.metric("市场状态", "震荡" if range_market else "趋势")

    if long_signal:
        st.success("🔥 多头共振成立 — 等待下一根K线突破信号K线最高价")
    else:
        st.info("无有效信号")

    # K线图
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df5["time"],
        open=df5["open"],
        high=df5["high"],
        low=df5["low"],
        close=df5["close"],
        name="K"
    ))

    fig.add_trace(go.Scatter(
        x=df5["time"],
        y=df5["ema7"],
        name="EMA7"
    ))

    fig.add_trace(go.Scatter(
        x=df5["time"],
        y=df5["upper"],
        name="Upper"
    ))

    fig.add_trace(go.Scatter(
        x=df5["time"],
        y=df5["lower"],
        name="Lower"
    ))

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")

    time.sleep(REFRESH_SECONDS)
    st.rerun()
