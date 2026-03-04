# -*- coding: utf-8 -*-
"""
纯K 5分钟实盘监控系统
不连接交易API，仅信号监控
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 5分钟纯K结构监控系统")

# =========================
# 参数设置
# =========================

body_threshold = 0.15
vol_period = 15
break_threshold = 0.001
lookback = 20
rr = 2.5

st.sidebar.header("策略参数")
body_threshold = st.sidebar.slider("实体比例", 0.05, 0.5, body_threshold, 0.01)
vol_period = st.sidebar.slider("成交量周期", 5, 30, vol_period)
break_threshold = st.sidebar.slider("突破幅度", 0.0005, 0.003, break_threshold, 0.0001)

# =========================
# 上传数据
# =========================

file = st.file_uploader("上传5分钟CSV数据", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]
df['time'] = pd.to_datetime(df['time'])

df.sort_values("time", inplace=True)

# =========================
# 特征计算
# =========================

df['high_max'] = df['high'].rolling(lookback).max().shift(1)
df['low_min'] = df['low'].rolling(lookback).min().shift(1)

df['body'] = abs(df['close'] - df['open'])
df['range'] = df['high'] - df['low']
df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
df['vol_ma'] = df['vol'].rolling(vol_period).mean()

df.dropna(inplace=True)

# =========================
# 最新一根K线结构判断
# =========================

latest = df.iloc[-1]

signal = 0
reason = []

if latest['close'] > latest['high_max']:
    if latest['body_ratio'] >= body_threshold:
        if latest['vol'] > latest['vol_ma']:
            if (latest['close'] - latest['high_max']) / latest['high_max'] >= break_threshold:
                signal = 1
            else:
                reason.append("突破幅度不足")
        else:
            reason.append("成交量不足")
    else:
        reason.append("实体不足")
else:
    reason.append("无突破")

if latest['close'] < latest['low_min']:
    if latest['body_ratio'] >= body_threshold:
        if latest['vol'] > latest['vol_ma']:
            if (latest['low_min'] - latest['close']) / latest['low_min'] >= break_threshold:
                signal = -1

# =========================
# 输出交易计划
# =========================

st.header("📌 当前交易计划")

if signal == 1:
    st.success("🚀 多头突破信号")
elif signal == -1:
    st.error("🔻 空头突破信号")
else:
    st.info("⏳ 无计划：等待结构")
    for r in reason:
        st.write("-", r)

# =========================
# 真实K线图
# =========================

st.header("📈 实时K线图")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name="K线"
))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=700
)

st.plotly_chart(fig, use_container_width=True)

st.success("监控运行中（不连接交易API）")
