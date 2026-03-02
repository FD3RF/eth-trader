import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="AI 高频 5m")
st.title("🚀 AI高频：5分钟分析 + TP/SL + 期望统计")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "history.csv"

# 高频刷新
st_autorefresh(interval=1000, key="refresh")

# =========================
# 模拟账户（100元）
# =========================
st.sidebar.header("模拟账户")
capital = st.sidebar.number_input("本金(RMB)", value=100.0, min_value=50.0)
leverage = st.sidebar.slider("杠杆", 5, 50, 20)
risk_percent = st.sidebar.slider("单笔风险%", 0.5, 5.0, 2.0) / 100

# =========================
# 数据（真实K线）
# =========================
def get_data():
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": SYMBOL, "bar": "5m", "limit": 300},
            timeout=5
        )
        j = r.json()
        if "data" not in j:
            return pd.DataFrame()

        df = pd.DataFrame(j["data"], columns=[
            "ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        df[["open","high","low","close","volume"]] = df[
            ["open","high","low","close","volume"]
        ].apply(pd.to_numeric, errors="coerce")
        return df.sort_values("ts")
    except:
        return pd.DataFrame()

df = get_data()
if df.empty:
    st.error("数据获取失败")
    st.stop()

# =========================
# 指标（轻量高频）
# =========================
def indicators(df):
    df = df.copy()
    df["EMA20"] = df.close.ewm(span=20).mean()
    df["EMA60"] = df.close.ewm(span=60).mean()

    gain = df.close.diff().clip(lower=0).rolling(14).mean()
    loss = (-df.close.diff().clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df["ATR"] = (df.high - df.low).rolling(14).mean()
    df["Z"] = (df.close - df.close.rolling(20).mean()) / df.close.rolling(20).std()

    return df.dropna()

df = indicators(df)
latest = df.iloc[-1]
price = latest.close

# =========================
# 高频信号逻辑（10–30单目标）
# =========================
trend = 1 if price > latest.EMA20 else -1
z = latest.Z

# 激进但不过度
long_cond = (
    trend > 0 and
    z < -1.2 and
    latest.RSI < 42
)

short_cond = (
    trend < 0 and
    z > 1.2 and
    latest.RSI > 58
)

signal = "多单" if long_cond else "空单" if short_cond else None

# TP/SL + RR
atr = latest.ATR if not pd.isna(latest.ATR) else price * 0.005
stop_distance = max(atr * 1.1, price * 0.004)

if signal == "多单":
    stop = price - stop_distance
    tp = price + stop_distance * 1.4
else:
    stop = price + stop_distance
    tp = price - stop_distance * 1.4

rr = round(abs((tp - price) / stop_distance), 2) if stop_distance > 0 else 0
quality = "高" if rr >= 1.3 and abs(z) > 1 else "中"

# =========================
# 模拟仓位（100元）
# =========================
risk_amount = capital * risk_percent
contracts = int((risk_amount / stop_distance) * leverage * 0.01) if stop_distance > 0 else 0

# =========================
# 历史与期望统计
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

completed = history[history.result.notna()]
win_rate = round((completed.result == "win").mean() * 100, 2) if not completed.empty else 0

# =========================
# 可视化（真实K线 + 点位）
# =========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.ts, open=df.open, high=df.high,
    low=df.low, close=df.close
))
fig.add_trace(go.Scatter(x=df.ts, y=df.EMA20, name="EMA20"))
fig.add_trace(go.Scatter(x=df.ts, y=df.EMA60, name="EMA60"))

# 标记进场点
if signal:
    fig.add_trace(go.Scatter(
        x=[latest.ts],
        y=[price],
        mode="markers",
        marker=dict(size=12),
        name="进场"
    ))

st.plotly_chart(fig, use_container_width=True)

# =========================
# 面板
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("价格", f"{price:.2f}")
col2.metric("信号", signal or "等待")
col3.metric("历史胜率", f"{win_rate}%")

if signal:
    st.success(f"""
{signal}
入场: {round(price,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {rr}
仓位: {contracts}张
""")
else:
    st.warning("等待高频机会")

# =========================
# 历史
# =========================
st.subheader("历史")
st.dataframe(history.tail(15))

st.subheader("统计")
st.write(f"胜率: {win_rate}% | 总信号: {len(history)}")

st.caption("100元模拟 | 高频 | 真实K线 | 期望统计")
