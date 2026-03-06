import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pyttsx3
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# ===== 参数 =====
INSTRUMENT = "ETH/USDT:USDT"
TIMEFRAME = "5m"
LIMIT = 200
MODEL_FILE = "eth_ai.pkl"

st.set_page_config(layout="wide")
st.title("ETH 5分钟AI盯盘系统")

# 自动刷新
st_autorefresh(interval=5000, key="refresh")

# ===== OKX =====
def exchange():
    return ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })

# ===== 语音 =====
def speak(msg):
    engine = pyttsx3.init()
    engine.say(msg)
    engine.runAndWait()

# ===== 数据 =====
@st.cache_data(ttl=10)
def fetch_data():
    ex = exchange()
    bars = ex.fetch_ohlcv(INSTRUMENT, timeframe=TIMEFRAME, limit=LIMIT)

    df = pd.DataFrame(
        bars,
        columns=['ts','open','high','low','close','vol']
    )

    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    return df

# ===== 多周期 =====
@st.cache_data(ttl=20)
def fetch_tf(tf):
    ex = exchange()
    bars = ex.fetch_ohlcv(INSTRUMENT, timeframe=tf, limit=120)

    df = pd.DataFrame(
        bars,
        columns=['ts','open','high','low','close','vol']
    )

    return df

# ===== AI模型 =====
def train_model(df):

    df["ret"] = df["close"].pct_change()

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()

    df = df.dropna()

    X = df[["ret","ma5","ma10"]]
    y = (df["ret"].shift(-1) > 0).astype(int)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X[:-1], y[:-1])

    joblib.dump(model, MODEL_FILE)

    return model

def load_model(df):

    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    return train_model(df)

# ===== AI预测 =====
def ai_predict(df, model):

    df["ret"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()

    df = df.dropna()

    X = df[["ret","ma5","ma10"]]

    pred = model.predict(X)

    return pred[-1]

# ===== 关键点位 =====
def key_levels(df):

    prev_high = df["high"].rolling(20).max().iloc[-2]
    prev_low = df["low"].rolling(20).min().iloc[-2]

    return prev_high, prev_low

# ===== 量能 =====
def volume_state(df):

    avg = df["vol"].rolling(10).mean().iloc[-1]
    curr = df["vol"].iloc[-1]

    ratio = curr / avg

    if ratio > 1.8:
        return "放量"

    if ratio < 0.6:
        return "缩量"

    return "正常"

# ===== 多周期趋势 =====
def tf_trend(df):

    ma = df["close"].rolling(20).mean()

    if df["close"].iloc[-1] > ma.iloc[-1]:
        return "多"

    return "空"

# ===== 口诀识别 =====
def detect_signal(df):

    prev_high, prev_low = key_levels(df)

    curr = df.iloc[-1]
    vol_state = volume_state(df)

    signal = "观察"
    reason = ""

    # 做多
    if curr["low"] >= prev_low and vol_state == "缩量":
        signal = "准备多"
        reason = "缩量回踩，低点不破"

    if curr["close"] > prev_high and vol_state == "放量":
        signal = "做多"
        reason = "放量起涨，突破前高"

    if curr["low"] <= prev_low and vol_state == "放量":
        signal = "激进多"
        reason = "放量急跌，底部不破"

    # 做空
    if curr["high"] <= prev_high and vol_state == "缩量":
        signal = "准备空"
        reason = "缩量反弹，高点不破"

    if curr["close"] < prev_low and vol_state == "放量":
        signal = "做空"
        reason = "放量下跌，跌破前低"

    if curr["high"] >= prev_high and vol_state == "放量":
        signal = "激进空"
        reason = "放量急涨，顶部不破"

    return signal, reason

# ===== K线图 =====
def plot_chart(df):

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    ))

    return fig

# ===== 主程序 =====
df = fetch_data()

model = load_model(df)

ai = ai_predict(df, model)

signal, reason = detect_signal(df)

tf1 = tf_trend(fetch_tf("1m"))
tf5 = tf_trend(fetch_tf("5m"))
tf15 = tf_trend(fetch_tf("15m"))

price = df["close"].iloc[-1]

# ===== UI =====
col1, col2, col3, col4 = st.columns(4)

col1.metric("ETH价格", round(price,2))
col2.metric("AI预测", "上涨" if ai else "下跌")
col3.metric("5m信号", signal)
col4.metric("多周期", f"{tf1}/{tf5}/{tf15}")

st.write("原因:", reason)

# ===== 图 =====
st.plotly_chart(plot_chart(df), width="stretch")

# ===== 播报 =====
if "last" not in st.session_state:
    st.session_state.last = ""

msg = f"{signal}，{reason}"

if msg != st.session_state.last:
    speak(msg)
    st.session_state.last = msg
