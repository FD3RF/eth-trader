import streamlit as st
import pandas as pd
import numpy as np
import websocket
import json
import threading
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# 参数
# ==============================
SYMBOL = "ethusdt"
INTERVAL = "5m"
ACCOUNT_SIZE = 10000
RISK_PER_TRADE = 0.01

# ==============================
# 全局状态
# ==============================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "loss_count" not in st.session_state:
    st.session_state.loss_count = 0

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

# ==============================
# WebSocket 实时数据
# ==============================
def on_message(ws, message):
    data = json.loads(message)
    if "k" in data:
        k = data["k"]
        new_row = {
            "time": pd.to_datetime(k["t"], unit="ms"),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"])
        }
        st.session_state.df = pd.concat(
            [st.session_state.df, pd.DataFrame([new_row])]
        ).tail(300)

def start_ws():
    url = f"wss://fstream.binance.com/ws/{SYMBOL}@kline_{INTERVAL}"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    ws.run_forever()

threading.Thread(target=start_ws, daemon=True).start()

# ==============================
# 指标函数
# ==============================
def EMA(series, n):
    return series.ewm(span=n).mean()

def MACD(df):
    ema12 = EMA(df["close"], 12)
    ema26 = EMA(df["close"], 26)
    macd = ema12 - ema26
    signal = EMA(macd, 9)
    return macd, signal

def KDJ(df):
    low_min = df["low"].rolling(9).min()
    high_max = df["high"].rolling(9).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    return k, d

def BOLL(df):
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    upper = mid + 2*std
    lower = mid - 2*std
    bandwidth = (upper-lower)/mid
    return upper, lower, bandwidth

# ==============================
# 主逻辑
# ==============================
st.title("ETH 专业级实时交易终端")

df = st.session_state.df

if len(df) > 50:

    df["ema7"] = EMA(df["close"],7)
    df["macd"], df["signal"] = MACD(df)
    df["k"], df["d"] = KDJ(df)
    df["upper"], df["lower"], df["bandwidth"] = BOLL(df)

    macd_cross_up = df["macd"].iloc[-2] < df["signal"].iloc[-2] and df["macd"].iloc[-1] > df["signal"].iloc[-1]
    macd_cross_down = df["macd"].iloc[-2] > df["signal"].iloc[-2] and df["macd"].iloc[-1] < df["signal"].iloc[-1]

    kdj_up = df["k"].iloc[-1] > df["d"].iloc[-1]
    kdj_down = df["k"].iloc[-1] < df["d"].iloc[-1]

    volume_valid = df["volume"].iloc[-1] > df["volume"].rolling(5).mean().iloc[-1]

    long_signal = macd_cross_up and kdj_up and volume_valid
    short_signal = macd_cross_down and kdj_down and volume_valid

    price = df["close"].iloc[-1]

    col1,col2,col3 = st.columns(3)
    col1.metric("当前价格",round(price,2))
    col2.metric("连续亏损",st.session_state.loss_count)

    # ==========================
    # 交易建议生成
    # ==========================
    if st.session_state.loss_count >= 2:
        st.error("风控触发：暂停交易")
    else:

        if long_signal:
            sl = price * 0.991
            tp1 = price * 1.018
            tp2 = price * 1.027
            position_size = (ACCOUNT_SIZE*RISK_PER_TRADE)/(price-sl)

            st.success("多头信号触发")
            st.write(f"建议做多 @ {price}")
            st.write(f"止损: {round(sl,2)}")
            st.write(f"TP1: {round(tp1,2)} | TP2: {round(tp2,2)}")
            st.write(f"建议仓位: {round(position_size,3)} ETH")

            st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg")

            st.session_state.last_signal = "LONG"

        elif short_signal:
            sl = price * 1.009
            tp1 = price * 0.982
            tp2 = price * 0.973
            position_size = (ACCOUNT_SIZE*RISK_PER_TRADE)/(sl-price)

            st.warning("空头信号触发")
            st.write(f"建议做空 @ {price}")
            st.write(f"止损: {round(sl,2)}")
            st.write(f"TP1: {round(tp1,2)} | TP2: {round(tp2,2)}")
            st.write(f"建议仓位: {round(position_size,3)} ETH")

            st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg")

            st.session_state.last_signal = "SHORT"

    # ==========================
    # 图表
    # ==========================
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ))

    fig.add_trace(go.Scatter(x=df["time"],y=df["ema7"],name="EMA7"))

    st.plotly_chart(fig,use_container_width=True)

    st.caption(f"更新时间: {datetime.now().strftime('%H:%M:%S')}")
