import streamlit as st
import pandas as pd
import numpy as np
import websocket
import json
import threading
import plotly.graph_objects as go
from collections import deque
from datetime import datetime

# ===============================
# 配置
# ===============================
SYMBOL = "ethusdt"
INTERVAL = "5m"
LIMIT = 200

# ===============================
# 全局数据（队列存最近K线）
# ===============================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "signals" not in st.session_state:
    st.session_state.signals = deque(maxlen=50)

# ===============================
# WebSocket 实时K线
# ===============================
def on_message(ws, message):
    data = json.loads(message)
    if "k" in data:
        k = data["k"]
        row = {
            "time": pd.to_datetime(k["t"], unit="ms"),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"])
        }
        df = st.session_state.df
        df = pd.concat([df, pd.DataFrame([row])]).tail(LIMIT)
        st.session_state.df = df

def start_ws():
    url = f"wss://fstream.binance.com/ws/{SYMBOL}@kline_{INTERVAL}"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    ws.run_forever()

threading.Thread(target=start_ws, daemon=True).start()

# ===============================
# 指标函数
# ===============================
def EMA(s, n):
    return s.ewm(span=n).mean()

def MACD(df):
    ema12 = EMA(df["close"], 12)
    ema26 = EMA(df["close"], 26)
    macd = ema12 - ema26
    signal = EMA(macd, 9)
    return macd, signal

def ATR(df, n=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift()), abs(low - close.shift()))
    )
    return tr.rolling(n).mean()

# ===============================
# 信号生成（生存版）
# ===============================
def generate_signal(df):
    if len(df) < 30:
        return None

    df["ema7"] = EMA(df["close"], 7)
    df["macd"], df["signal"] = MACD(df)
    df["atr"] = ATR(df)

    macd_cross_up = df["macd"].iloc[-2] < df["signal"].iloc[-2] and df["macd"].iloc[-1] > df["signal"].iloc[-1]
    macd_cross_down = df["macd"].iloc[-2] > df["signal"].iloc[-2] and df["macd"].iloc[-1] < df["signal"].iloc[-1]

    price = df["close"].iloc[-1]
    ema7 = df["ema7"].iloc[-1]
    atr = df["atr"].iloc[-1]

    # 趋势条件
    long = price > ema7 and macd_cross_up
    short = price < ema7 and macd_cross_down

    if long:
        return {
            "side": "LONG",
            "entry": price,
            "stop": price - atr * 1.5,
            "tp": price + (atr * 1.5) * 2
        }

    if short:
        return {
            "side": "SHORT",
            "entry": price,
            "stop": price + atr * 1.5,
            "tp": price - (atr * 1.5) * 2
        }

    return None

# ===============================
# 风控
# ===============================
def risk_check(signal):
    if not signal:
        return "无信号"

    risk = abs(signal["entry"] - signal["stop"]) / signal["entry"] * 100

    if risk > 2:
        return "风险过高（>2%）— 不建议交易"

    return "风险可控"

# ===============================
# UI
# ===============================
st.title("5分钟生存盯盘终端")

df = st.session_state.df

if len(df) > 30:
    signal = generate_signal(df)

    col1, col2 = st.columns(2)
    col1.metric("最新价格", round(df["close"].iloc[-1], 4))
    col2.metric("K线数量", len(df))

    if signal:
        st.success(f"信号：{signal['side']}")
        st.write(f"入场：{round(signal['entry'],4)}")
        st.write(f"止损：{round(signal['stop'],4)}")
        st.write(f"止盈：{round(signal['tp'],4)}")
        st.write(f"风控：{risk_check(signal)}")
        st.session_state.signals.append(signal)
    else:
        st.info("暂无信号")

    # 可视化K线
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ))
    if "ema7" in df:
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema7"], name="EMA7"))
    st.plotly_chart(fig, use_container_width=True)

    # 最近信号历史
    st.subheader("最近信号")
    for s in list(st.session_state.signals):
        st.write(f"{s['side']} @ {round(s['entry'],4)} | SL {round(s['stop'],4)} | TP {round(s['tp'],4)}")

    st.caption(f"更新时间 {datetime.now().strftime('%H:%M:%S')}")
else:
    st.info("等待数据中...")
