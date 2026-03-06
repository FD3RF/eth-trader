import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(page_title="ETH 5m AI量价策略（本地版）", layout="wide")
st.title("📊 ETH 永续 5分钟 AI 量价策略（本地数据）")

# ====================== 数据加载 ======================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ETHUSDT_5m_last_90days.csv")
        df["open_time"] = pd.to_datetime(df["open_time"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception:
        # 如果没有 CSV → 生成示例数据
        n = 200
        rng = pd.date_range("2024-01-01", periods=n, freq="5min")
        price = np.cumsum(np.random.randn(n)) + 2000
        df = pd.DataFrame({
            "open_time": rng,
            "open": price + np.random.randn(n),
            "high": price + np.random.rand(n) * 5,
            "low": price - np.random.rand(n) * 5,
            "close": price,
            "volume": np.abs(np.random.randn(n) * 1000)
        })
        return df

df = load_data()

# ====================== 策略逻辑 ======================
def signal_logic(df):
    if df is None or df.empty or len(df) < 2:
        return "observe", "数据不足"

    last = df.iloc[-1]
    prev = df.iloc[-51:-1] if len(df) > 51 else df

    recent_high = prev["high"].max()
    recent_low = prev["low"].min()
    avg_vol = prev["volume"].mean()

    vol_ratio = last["volume"] / avg_vol if avg_vol > 0 else 1
    is_shrink = vol_ratio < 0.6
    is_expand = vol_ratio > 1.8

    near_low = abs(last["low"] - recent_low) / recent_low < 0.003
    near_high = abs(last["high"] - recent_high) / recent_high < 0.003
    broke_high = last["close"] > recent_high
    broke_low = last["low"] < recent_low * 0.997

    drop_pct = (last["open"] - last["low"]) / last["open"]

    if is_expand and broke_high:
        return "buy", "放量起涨，突破前高"
    if is_expand and broke_low:
        return "sell", "放量跌破前低"
    if is_expand and near_low and drop_pct > 0.012:
        return "buy", "放量暴跌低点不破（机会）"
    if is_expand and near_high:
        return "sell", "放量急涨顶部不破（机会）"
    if is_shrink and near_low:
        return "observe", "缩量回踩低点不破"
    if is_shrink and near_high:
        return "observe", "缩量反弹高点不破"
    if is_shrink:
        return "observe", "缩量横盘，等待放量方向"

    return "observe", "量能不明"

# ====================== 图表 ======================
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["open_time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ))
    fig.update_layout(
        template="plotly_dark",
        height=600,
        title="ETH 5分钟K线（本地数据）"
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================== 主流程 ======================
signal, reason = signal_logic(df)
last = df.iloc[-1]

plot_chart(df)

st.subheader("🤖 策略播报")
st.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"信号：{signal}")
st.write(f"原因：{reason}")
st.write(f"当前价：{last['close']:.2f}")

if signal == "buy":
    st.success("📈 多单倾向")
elif signal == "sell":
    st.error("📉 空单倾向")
else:
    st.info("⏳ 观察区")

st.subheader("📋 最近数据")
st.dataframe(df.tail())
