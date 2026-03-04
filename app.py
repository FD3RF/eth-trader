# -*- coding: utf-8 -*-
"""
纯K突破 + 放量策略
版本：稳定可跑
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="纯K突破策略", layout="wide")
st.title("📈 纯K突破 + 放量 策略")

# 上传
uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
if uploaded_file is None:
    st.info("上传数据")
    st.stop()

# 加载
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    return df

df = load_data(uploaded_file)
st.write(f"数据行数: {len(df)}")

# 字段检测
required = ['open','high','low','close','volume']
for c in required:
    if c not in df.columns:
        st.error(f"缺少字段: {c}")
        st.stop()

# ================================
# 参数
# ================================
st.sidebar.header("策略参数")

lookback = st.sidebar.slider("突破周期(前高/前低)", 5, 50, 20)
vol_mult = st.sidebar.slider("放量倍数", 1.0, 3.0, 1.5)
rr = st.sidebar.slider("盈亏比", 1.0, 4.0, 2.0)

# ================================
# 特征：前高前低 + 放量
# ================================
df['high_max'] = df['high'].rolling(lookback).max().shift(1)
df['low_min'] = df['low'].rolling(lookback).min().shift(1)
df['vol_ma'] = df['volume'].rolling(lookback).mean().shift(1)

# 信号
df['signal'] = 0

# 多头：突破前高 + 放量
df.loc[
    (df['close'] > df['high_max']) &
    (df['volume'] > df['vol_ma'] * vol_mult),
    'signal'
] = 1

# 空头：跌破前低 + 放量
df.loc[
    (df['close'] < df['low_min']) &
    (df['volume'] > df['vol_ma'] * vol_mult),
    'signal'
] = -1

df.dropna(inplace=True)

# ================================
# 回测
# ================================
def backtest(df):
    equity = [0]
    position = 0
    entry = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平多
        if position == 1 and row['signal'] == 0:
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 平空
        if position == -1 and row['signal'] == 0:
            pnl = entry - price
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开多
        if prev['signal'] == 1 and position == 0:
            entry = price
            position = 1

        # 开空
        if prev['signal'] == -1 and position == 0:
            entry = price
            position = -1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "total": sum(trades),
        "equity": equity
    }

# 执行
res = backtest(df)

# 展示
st.header("回测结果")
st.metric("交易", res["trades"])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("盈利", f"{res['total']:.2f}")

if res["trades"] > 0:
    st.line_chart(pd.Series(res["equity"]))

st.success("运行完毕")
