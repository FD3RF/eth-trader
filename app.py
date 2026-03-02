"""
5分钟趋势合约系统（实战版）- 含支撑阻力线
策略：EMA20/60趋势 + ADX强度 + 回调确认 + 支撑阻力可视化
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime
import requests
import time

st.set_page_config(layout="wide")
st.title("📈 5分钟趋势合约系统（含支撑阻力）")

SYMBOL = "ETH-USDT-SWAP"

# ==========================
# 侧边栏参数
# ==========================
st.sidebar.header("⚙️ 策略参数")
adx_threshold = st.sidebar.slider("ADX趋势阈值", 20, 40, 25, help="ADX大于此值才认为有趋势")
ema_period_fast = st.sidebar.slider("快线EMA周期", 10, 30, 20)
ema_period_slow = st.sidebar.slider("慢线EMA周期", 30, 100, 60)
atr_period = st.sidebar.slider("ATR周期", 7, 30, 14)
lookback_sr = st.sidebar.slider("支撑/阻力周期（根K线）", 10, 50, 20, help="计算前高前低所用的K线数量")
risk_reward = st.sidebar.slider("盈亏比", 1.0, 3.0, 2.0, step=0.1)
risk_percent = st.sidebar.slider("单笔风险 (%)", 1.0, 5.0, 2.0, step=0.5)

# ==========================
# 数据获取
# ==========================
@st.cache_data(ttl=5)
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}
    retries = 3
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=5, params=params)
            j = r.json()
            if "data" in j:
                break
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"数据获取失败: {e}")
                return pd.DataFrame()
            time.sleep(1)
    else:
        return pd.DataFrame()

    df = pd.DataFrame(j["data"], columns=[
        "ts", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df

df = get_data()
if df.empty:
    st.stop()

# ==========================
# 指标计算
# ==========================
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_period_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_period_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)

# 支撑阻力（最近N根K线的最高/最低）
df["resistance"] = df["high"].rolling(window=lookback_sr).max()
df["support"] = df["low"].rolling(window=lookback_sr).min()

df = df.dropna().reset_index(drop=True)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==========================
# 趋势判断
# ==========================
trend = None
if latest["EMA_fast"] > latest["EMA_slow"] and latest["ADX"] > adx_threshold:
    trend = "多"
elif latest["EMA_fast"] < latest["EMA_slow"] and latest["ADX"] > adx_threshold:
    trend = "空"

# ==========================
# 回调入场信号（基于EMA20回调）
# ==========================
signal = None
if trend == "多":
    # 价格回踩EMA_fast附近（±0.2%）且前一根K线收盘在EMA_fast下方
    if abs(latest["close"] - latest["EMA_fast"]) / latest["EMA_fast"] < 0.002 and prev["close"] < prev["EMA_fast"]:
        # K线确认：下影线较长或阳线
        body = abs(latest["close"] - latest["open"])
        lower_shadow = min(latest["close"], latest["open"]) - latest["low"]
        if lower_shadow > body * 1.5 or latest["close"] > latest["open"]:
            signal = "多"
elif trend == "空":
    if abs(latest["close"] - latest["EMA_fast"]) / latest["EMA_fast"] < 0.002 and prev["close"] > prev["EMA_fast"]:
        body = abs(latest["close"] - latest["open"])
        upper_shadow = latest["high"] - max(latest["close"], latest["open"])
        if upper_shadow > body * 1.5 or latest["close"] < latest["open"]:
            signal = "空"

# ==========================
# 绘图（K线 + 均线 + 支撑阻力）
# ==========================
fig = go.Figure()

# K线
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
))

# EMA均线
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], line=dict(color="blue", width=1), name=f"EMA{ema_period_fast}"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], line=dict(color="orange", width=1), name=f"EMA{ema_period_slow}"))

# 支撑阻力线（虚线）
fig.add_trace(go.Scatter(x=df["ts"], y=df["support"], line=dict(color="green", width=1, dash="dash"), name="支撑"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["resistance"], line=dict(color="red", width=1, dash="dash"), name="阻力"))

# 标记信号点
if signal:
    fig.add_trace(go.Scatter(
        x=[latest["ts"]],
        y=[latest["close"]],
        mode="markers",
        marker=dict(symbol="star", size=15, color="yellow"),
        name="信号点"
    ))

fig.update_layout(
    title=f"{SYMBOL} 5分钟图 (支撑阻力周期={lookback_sr})",
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# 状态面板
# ==========================
st.subheader("📊 当前状态")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("趋势方向", trend if trend else "无")
with col2:
    st.metric("ADX", f"{latest['ADX']:.1f}")
with col3:
    st.metric("最新价", f"{latest['close']:.2f}")
with col4:
    st.metric("ATR", f"{latest['ATR']:.4f}")

st.write(f"**支撑** (最近{lookback_sr}根): {latest['support']:.2f}")
st.write(f"**阻力** (最近{lookback_sr}根): {latest['resistance']:.2f}")

if signal:
    st.success(f"📈 当前信号: {signal} (回调确认)")
else:
    st.info("⏳ 无信号，等待趋势机会")

# 可选的简单模拟账户提示
st.caption("注意：本系统仅展示信号，未包含自动交易模块。如需模拟交易，可参考之前的账户代码。")
