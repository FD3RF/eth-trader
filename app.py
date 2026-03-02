import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import requests
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

st.set_page_config(layout="wide", page_title="ETH高频AI监控")
st.title("🚀 ETH-USDT-SWAP 5分钟高频AI监控（工业稳定版）")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "signals_history.csv"

st_autorefresh(interval=3000, key="refresh")

# =====================
# 资金设置
# =====================
st.sidebar.header("💰 100元本金模拟")
capital = st.sidebar.number_input("初始资金 (RMB)", value=100.0, min_value=50.0)
leverage = st.sidebar.slider("杠杆倍数", 5, 50, 20)
risk_percent = st.sidebar.slider("单笔风险%", 0.5, 5.0, 2.0) / 100
long_only = st.sidebar.checkbox("只做多单", value=True)

# =====================
# 数据获取
# =====================
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}
    r = requests.get(url, params=params, timeout=5)
    j = r.json()
    if j.get("code") != "0":
        return pd.DataFrame()
    df = pd.DataFrame(j["data"], columns=["ts","open","high","low","close","vol","a","b","c"])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ["open","high","low","close","vol"]:
        df[col] = pd.to_numeric(df[col])
    return df.sort_values("ts")

df = get_data()
if df.empty:
    st.error("数据获取失败")
    st.stop()

# =====================
# 指标
# =====================
df["EMA60"] = df["close"].ewm(span=60).mean()
df["RSI"] = df["close"].diff().clip(lower=0).rolling(14).mean() / (
    df["close"].diff().abs().rolling(14).mean()
) * 100
df.dropna(inplace=True)

price = df["close"].iloc[-1]

# =====================
# AI模型缓存
# =====================
@st.cache_resource
def load_model(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1,1))

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(60,1)),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler

model, scaler = load_model(df["close"].values)

def ai_predict():
    last60 = df["close"].values[-60:]
    scaled = scaler.transform(last60.reshape(-1,1))
    X = scaled.reshape(1,60,1)
    pred = model.predict(X, verbose=0)
    real = scaler.inverse_transform(pred)[0][0]
    return "多头" if real > price else "空头"

ai_trend = ai_predict()

# =====================
# 信号逻辑
# =====================
trend = 1 if price > df["EMA60"].iloc[-1] else -1

signal = None
stop = tp = rr = 0

if trend > 0 and ai_trend=="多头":
    stop = price * 0.992
    tp = price * 1.015
    rr = round((tp-price)/(price-stop),2)
    signal="多单"
elif not long_only and trend<0 and ai_trend=="空头":
    stop = price * 1.008
    tp = price * 0.985
    rr = round((price-tp)/(stop-price),2)
    signal="空单"

# =====================
# 仓位计算
# =====================
risk_amount = capital * risk_percent
stop_distance = abs(price-stop) if signal else 0
contracts = int((risk_amount/stop_distance)*leverage*0.01) if stop_distance>0 else 0

# =====================
# 图表
# =====================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], name="EMA60"))
st.plotly_chart(fig, use_container_width=True)

# =====================
# 面板
# =====================
col1,col2,col3 = st.columns(3)
col1.metric("当前价格", f"{price:.2f}")
col2.metric("AI趋势", ai_trend)
col3.metric("仓位张数", contracts)

if signal:
    st.success(f"{signal} | 入场 {price:.2f} | 止损 {stop:.2f} | 止盈 {tp:.2f} | RR {rr}")
else:
    st.warning("等待信号")

st.caption("工业级稳定版 | 已修复全部结构性错误")
