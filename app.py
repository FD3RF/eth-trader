import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI口诀实时播报", layout="wide")
st.title("📢 AI 口诀实时播报（语音版）")

# ======= 数据 =======
def load_data():
    n = 200
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = np.cumsum(np.random.randn(n)) + 2000
    return pd.DataFrame({
        "open_time": rng,
        "open": price + np.random.randn(n),
        "high": price + np.random.rand(n) * 5,
        "low": price - np.random.rand(n) * 5,
        "close": price,
        "volume": np.abs(np.random.randn(n) * 1000)
    })

df = load_data()

# ======= 策略 =======
def signal_logic(df):
    last = df.iloc[-1]
    prev = df.iloc[-51:-1]

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
        return "放量起涨，突破前高，直接开多"
    if is_expand and broke_low:
        return "放量下跌，跌破前低，直接开空"
    if is_expand and near_low and drop_pct > 0.012:
        return "放量暴跌低点不破，这是机会"
    if is_expand and near_high:
        return "放量急涨顶部不破，这是机会"
    if is_shrink and near_low:
        return "缩量回踩，低点不破，准备动手"
    if is_shrink and near_high:
        return "缩量反弹，高点不破，准备动手"
    if is_shrink:
        return "缩量横盘，等待放量方向"

    return "量能不明"

# ======= 信号 =======
motto = signal_logic(df)
last = df.iloc[-1]

# ======= 语音播报函数（浏览器语音）=======
def speak(text):
    js = f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = 'zh-CN';
    msg.rate = 1;
    msg.pitch = 1;
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js, height=0)

# ======= UI =======
st.subheader("🤖 当前口诀")
st.success(motto)

if st.button("📢 语音播报"):
    speak(motto)

# ======= 图表 =======
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["open_time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# ======= 播报信息 =======
st.subheader("📊 播报信息")
st.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"当前价：{last['close']:.2f}")
st.write(f"状态：{motto}")
