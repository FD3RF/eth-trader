import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide", page_title="ETH 高频AI交易")
st.title("🚀 高频AI交易：5分钟 + 统计期望（100元模拟）")

# =========================
# 配置
# =========================
SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "history.csv"
MODEL_FILE = "ai_model.keras"
st_autorefresh = st.autorefresh if hasattr(st, "autorefresh") else None

# =========================
# 模拟资金
# =========================
st.sidebar.header("💰 模拟账户")
capital = st.sidebar.number_input("初始资金(RMB)", value=100.0, min_value=50.0)
leverage = st.sidebar.slider("杠杆", 5, 50, 20)
risk_percent = st.sidebar.slider("单笔风险%", 0.5, 5.0, 2.0) / 100

# =========================
# 数据获取
# =========================
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        if "data" not in j:
            return pd.DataFrame()
        df = pd.DataFrame(j["data"], columns=[
            "ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("ts")
    except:
        return pd.DataFrame()

df = get_data()
if df.empty:
    st.error("数据获取失败")
    st.stop()

# =========================
# 手动指标（轻量高频）
# =========================
def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA60"] = df["close"].ewm(span=60).mean()
    df["RSI"] = 100 - (100 / (1 + (
        df["close"].diff().clip(lower=0).rolling(14).mean() /
        (-df["close"].diff().clip(upper=0)).rolling(14).mean()
    )))
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
    df["Z"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    return df.dropna()

df = add_indicators(df)
latest = df.iloc[-1]
price = latest["close"]

# =========================
# 高频信号逻辑（激进）
# =========================
trend = 1 if price > latest["EMA20"] else -1
z = latest["Z"]

# 高频条件（不苛刻）
long_cond = (
    trend > 0 and
    z < -1.0 and
    latest["RSI"] < 45
)

short_cond = (
    trend < 0 and
    z > 1.0 and
    latest["RSI"] > 55
)

signal = "多单" if long_cond else "空单" if short_cond else None

# RR模型
atr = latest["ATR"] if not pd.isna(latest["ATR"]) else price * 0.005
stop_distance = max(atr * 1.1, price * 0.004)

if signal == "多单":
    stop = price - stop_distance
    tp = price + stop_distance * 1.4
else:
    stop = price + stop_distance
    tp = price - stop_distance * 1.4

rr = round(abs((tp - price) / stop_distance), 2)

# 评分
score = 0
if abs(z) > 1.0: score += 2
if latest["RSI"] < 45 or latest["RSI"] > 55: score += 1
if rr >= 1.2: score += 2
quality = "高" if score >= 4 else "中"

# =========================
# 100元模拟仓位
# =========================
risk_amount = capital * risk_percent
contracts = int((risk_amount / stop_distance) * leverage * 0.01) if stop_distance > 0 else 0
margin_used = (price * contracts * 0.01) / leverage

# =========================
# 历史（统计期望）
# =========================
def load_history():
    try:
        return pd.read_csv(HISTORY_FILE)
    except:
        return pd.DataFrame(columns=["time","direction","entry","stop","tp","result","rr"])

history = load_history()

if signal:
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": signal,
        "entry": round(price,4),
        "stop": round(stop,4),
        "tp": round(tp,4),
        "result": "",
        "rr": rr
    }
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True).tail(5000)
    history.to_csv(HISTORY_FILE, index=False)

# 胜率统计
completed = history[history["result"].notna()]
win_rate = round((completed["result"] == "win").mean() * 100, 2) if not completed.empty else 0

# =========================
# 图表
# =========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], name="EMA20"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], name="EMA60"))
st.plotly_chart(fig, use_container_width=True)

# =========================
# 面板
# =========================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("当前价格", f"{price:.2f}")
with col2:
    st.metric("信号", signal or "等待")
with col3:
    st.metric("历史胜率", f"{win_rate}%")

if signal:
    st.success(f"""
🚀 {signal}
入场: {round(price,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {rr}
建议仓位: {contracts}张
""")
else:
    st.warning("等待高频机会...")

# 历史
st.subheader("历史信号")
st.dataframe(history.tail(15))

# 统计
st.subheader("统计")
st.write(f"胜率: {win_rate}% | 总信号: {len(history)}")

st.caption("100元模拟 | 高频策略 | 真实数据 | 激进版")
