"""
工程级小利润趋势系统（一次性完整版）
可直接运行
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
import requests
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --------------------------
# 日志
# --------------------------
log_handler = RotatingFileHandler("error.log", maxBytes=5*1024*1024, backupCount=3)
logging.basicConfig(handlers=[log_handler], level=logging.INFO)

# --------------------------
# 页面
# --------------------------
st.set_page_config(layout="wide")
st.title("📈 工程级趋势结构小利润系统")

# --------------------------
# 省电刷新
# --------------------------
power_saving = st.sidebar.checkbox("省电模式", value=False)
refresh_interval = 15000 if power_saving else 5000
st_autorefresh(interval=refresh_interval, key="refresh")

# --------------------------
# OKX 接入
# --------------------------
SYMBOL = "ETH-USDT-SWAP"
API_URL = "https://www.okx.com/api/v5/market/candles"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

# --------------------------
# 数据线程（工程级）
# --------------------------
class DataFetcher:
    def __init__(self):
        self.lock = threading.Lock()
        self._data = None
        self.fail = 0
        self.last_heartbeat = time.time()
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _fetch(self):
        params = {"instId": SYMBOL, "bar": "5m", "limit": 300}
        for attempt in range(3):
            try:
                r = requests.get(API_URL, params=params, headers=headers, timeout=8)
                if r.status_code == 200:
                    j = r.json()
                    return j.get("data")
            except Exception:
                time.sleep(2 ** attempt)
        return None

    def _run(self):
        while not self.stop_flag:
            try:
                data = self._fetch()
                if data:
                    df = pd.DataFrame(data, columns=[
                        "ts","open","high","low","close","volume",
                        "volCcy","volCcyQuote","confirm"
                    ])
                    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
                    for col in ["open","high","low","close","volume"]:
                        df[col] = df[col].astype(float)
                    df = df.sort_values("ts")
                    with self.lock:
                        self._data = df
                self.fail = 0
                self.last_heartbeat = time.time()
            except Exception:
                self.fail += 1
                logging.exception("数据线程异常")
                if self.fail > 5:
                    self.stop_flag = True
            time.sleep(5)

    def get(self):
        with self.lock:
            return self._data.copy() if self._data is not None else None

# 初始化
if "fetcher" not in st.session_state:
    st.session_state.fetcher = DataFetcher()

df = st.session_state.fetcher.get()
if df is None or df.empty:
    st.info("等待数据...")
    st.stop()

# --------------------------
# 指标计算
# --------------------------
df["EMA20"] = ta.trend.ema_indicator(df["close"], window=20)
df["EMA50"] = ta.trend.ema_indicator(df["close"], window=50)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

# --------------------------
# 结构检测（Swing）
# --------------------------
def find_swing(df, window=3):
    highs = []
    lows = []
    for i in range(window, len(df)-window):
        if df["high"].iloc[i] > df["high"].iloc[i-window:i].max() and df["high"].iloc[i] > df["high"].iloc[i+1:i+window+1].max():
            highs.append((df["ts"].iloc[i], df["high"].iloc[i]))
        if df["low"].iloc[i] < df["low"].iloc[i-window:i].min() and df["low"].iloc[i] < df["low"].iloc[i+1:i+window+1].min():
            lows.append((df["ts"].iloc[i], df["low"].iloc[i]))
    return highs, lows

highs, lows = find_swing(df)

latest = df.iloc[-1]
prev = df.iloc[-2]

# 趋势状态
trend_up = latest["EMA20"] > latest["EMA50"] and latest["ADX"] > 22
trend_down = latest["EMA20"] < latest["EMA50"] and latest["ADX"] > 22

# --------------------------
# 小利润信号（工程级）
# --------------------------
def small_profit_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    atr = latest["ATR"]

    # 趋势
    if latest["EMA20"] > latest["EMA50"] and latest["ADX"] > 22:
        trend = "多"
    elif latest["EMA20"] < latest["EMA50"] and latest["ADX"] > 22:
        trend = "空"
    else:
        return None

    # 多单回调
    if trend == "多":
        if latest["close"] > latest["EMA20"] and prev["close"] < prev["EMA20"]:
            if abs(latest["close"] - latest["EMA20"]) < atr * 0.7:
                return "多"

    # 空单回调
    if trend == "空":
        if latest["close"] < latest["EMA20"] and prev["close"] > prev["EMA20"]:
            if abs(latest["close"] - latest["EMA20"]) < atr * 0.7:
                return "空"

    return None

signal = small_profit_signal(df)

# 防抖
if signal and st.session_state.get("last_signal_ts") != latest["ts"]:
    st.session_state.last_signal_ts = latest["ts"]

# --------------------------
# 绘图
# --------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], name="EMA20"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA50"], name="EMA50"))

if highs:
    hx, hy = zip(*highs)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers",
                            marker=dict(symbol="triangle-down", size=8),
                            name="结构高点"))
if lows:
    lx, ly = zip(*lows)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers",
                            marker=dict(symbol="triangle-up", size=8),
                            name="结构低点"))

if signal:
    y = latest["high"] * 1.002 if signal == "多" else latest["low"] * 0.998
    fig.add_trace(go.Scatter(
        x=[latest["ts"]],
        y=[y],
        mode="markers+text",
        text=signal,
        marker=dict(symbol="arrow-up" if signal=="多" else "arrow-down", size=15),
        name="信号"
    ))

fig.update_layout(title="趋势结构小利润", height=650)
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 状态面板
# --------------------------
st.subheader("状态")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("趋势", "多头" if trend_up else "空头" if trend_down else "震荡")
with col2:
    st.metric("ADX", f"{latest['ADX']:.1f}")
with col3:
    st.metric("ATR", f"{latest['ATR']:.4f}")

if signal:
    st.success(f"小利润信号: {signal}")
else:
    st.info("无信号")

st.caption(f"更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
