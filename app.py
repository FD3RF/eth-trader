# -*- coding: utf-8 -*-
"""
真实K线图 + 信号
进场提示
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="K线+信号", layout="wide")
st.title("📊 真实K线 + 进场提示")

file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

# ================================
# 加载
# ================================
df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

# 字段映射
df['open'] = df['open']
df['high'] = df['high']
df['low'] = df['low']
df['close'] = df['close']
df['volume'] = df['vol']

st.write(f"数据行: {len(df)}")

# ================================
# 特征
# ================================
def build_features(df):
    df = df.copy()
    lookback = 20

    df['high_max'] = df['high'].rolling(lookback).max().shift(1)
    df['low_min'] = df['low'].rolling(lookback).min().shift(1)

    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-9)

    df.dropna(inplace=True)
    return df

df = build_features(df)

# ================================
# 信号
# ================================
df['signal'] = 0

df.loc[
    (df['close'] > df['high_max']) &
    (df['body_ratio'] > 0.35) &
    (df['close'] > df['open']),
    'signal'
] = 1

df.loc[
    (df['close'] < df['low_min']) &
    (df['body_ratio'] > 0.35) &
    (df['close'] < df['open']),
    'signal'
] = -1

# ================================
# K线图
# ================================
st.header("K线图")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name="K线"
))

# 多头信号点
buy = df[df['signal'] == 1]
fig.add_trace(go.Scatter(
    x=buy.index,
    y=buy['close'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=12, color='green'),
    name='多头'
))

# 空头信号点
sell = df[df['signal'] == -1]
fig.add_trace(go.Scatter(
    x=sell.index,
    y=sell['close'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=12, color='red'),
    name='空头'
))

fig.update_layout(
    title="K线 + 信号",
    xaxis_title="时间",
    yaxis_title="价格",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# ================================
# 最新计划
# ================================
st.header("最新交易计划")

latest = df.iloc[-1]

if latest['signal'] == 1:
    st.success("📈 多头计划：突破 + 强阳")
    st.write("""
    进场：
    - 突破前高
    - 强实体阳

    止损：
    - 结构跌破

    退出：
    - 盈利目标
    - 信号反转
    """)

elif latest['signal'] == -1:
    st.error("📉 空头计划：跌破 + 强阴")
    st.write("""
    进场：
    - 跌破前低
    - 强实体阴

    止损：
    - 结构突破

    退出：
    - 盈利目标
    - 信号反转
    """)

else:
    st.info("⏳ 无计划：等待结构")
    st.write("""
    不进场：
    - 无突破
    - 无强实体
    - 结构未确认
    """)

# ================================
# 历史统计
# ================================
st.header("历史统计")

st.write("多头次数:", (df['signal'] == 1).sum())
st.write("空头次数:", (df['signal'] == -1).sum())
st.write("无信号:", (df['signal'] == 0).sum())

st.success("K线加载完成")
