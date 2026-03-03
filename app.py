import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="小利润战神 V2600")
st.title("🛡️ ETH 小利润·趋势共振工程版")

# --------------------------
# 配置参数
# --------------------------
SYMBOL = "ETH-USDT-SWAP"
with st.sidebar:
    st.header("⚙️ 核心逻辑参数")
    adx_threshold = st.slider("ADX 强度阈值 (趋势过滤)", 15, 35, 25) # 提高阈值过滤震荡
    atr_sl_mult = 0.6  # 止损系数 (你的逻辑)
    atr_tp_mult = 0.8  # 止盈系数 (你的逻辑)
    dist_buffer = 0.5   # EMA 支撑/压力缓冲区 (ATR倍数)

# --------------------------
# 数据与多周期共振引擎
# --------------------------
def fetch_data(bar="5m", limit=300):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": bar, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        df = pd.DataFrame(r.json()["data"], columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open","high","low","close"]: df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except: return pd.DataFrame()

# 获取15分钟共振方向
df15 = fetch_data("15m", 100)
df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
tf_direction = "多" if df15.iloc[-1]["EMA20"] > df15.iloc[-2]["EMA20"] else "空"

# 获取5分钟主数据
df = fetch_data("5m", 300)
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=12)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=50)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

# 结构计算：过去3根K线的局部极值
df["low_min_3"] = df["low"].shift(1).rolling(window=3).min()
df["high_max_3"] = df["high"].shift(1).rolling(window=3).max()

# --------------------------
# 逻辑判定逻辑
# --------------------------
def check_logic(df, tf_dir, adx_th, buf):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. 假突破过滤 (长影线判定)
    body = abs(latest["close"] - latest["open"])
    is_fake = (latest["high"] - max(latest["close"], latest["open"]) > body * 1.5) or \
              (min(latest["close"], latest["open"]) - latest["low"] > body * 1.5)
    
    # 2. 趋势与强度
    trend_up = latest["EMA_fast"] > latest["EMA_slow"] and latest["ADX"] > adx_th
    trend_down = latest["EMA_fast"] < latest["EMA_slow"] and latest["ADX"] > adx_th
    
    signal = None
    # 多头逻辑：15m共振 + 5m趋势 + 低点抬高 + 回调至EMA缓冲区
    if trend_up and tf_dir == "多" and not is_fake:
        # 低点抬高：当前低点 > 过去3根K线最低点
        structure_ok = latest["low"] > latest["low_min_3"]
        # 回调结束判定：价格在 EMA 快线之上，但距离不超过 0.5 ATR
        at_support = latest["EMA_fast"] < latest["close"] < (latest["EMA_fast"] + latest["ATR"] * buf)
        if structure_ok and at_support: signal = "多"
        
    # 空头逻辑
    elif trend_down and tf_dir == "空" and not is_fake:
        structure_ok = latest["high"] < latest["high_max_3"]
        at_resistance = (latest["EMA_fast"] - latest["ATR"] * buf) < latest["close"] < latest["EMA_fast"]
        if structure_ok and at_resistance: signal = "空"
        
    return signal, latest

signal, last_row = check_logic(df, tf_direction, adx_threshold, dist_buffer)

# --------------------------
# UI 渲染与回测
# --------------------------
st.subheader("📊 实时状态面板")
c1, c2, c3, c4 = st.columns(4)
c1.metric("15M 共振方向", tf_direction, delta="对齐" if (signal == tf_direction) else None)
c2.metric("ADX 强度", f"{last_row['ADX']:.1f}")
c3.metric("ATR 波动", f"{last_row['ATR']:.2f}")
c4.metric("价格", f"{last_row['close']:.2f}")

if signal:
    sl = last_row["close"] - last_row["ATR"] * atr_sl_mult if signal == "多" else last_row["close"] + last_row["ATR"] * atr_sl_mult
    tp = last_row["close"] + last_row["ATR"] * atr_tp_mult if signal == "多" else last_row["close"] - last_row["ATR"] * atr_tp_mult
    st.success(f"🔥 发现【{signal}】信号！ 止盈目标: {tp:.2f} | 止损防御: {sl:.2f}")
else:
    st.info("⏳ 监控中：等待多周期趋势共振及回调确认...")

# 图表展示 (EMA + 信号标记)
fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="K线")])
fig.add_trace(go.Scatter(x=df['ts'], y=df['EMA_fast'], line=dict(color='yellow', width=1), name="EMA12"))
fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
