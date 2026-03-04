# -*- coding: utf-8 -*-
"""
纯K线突破多空策略（无指标）
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 纯K线突破策略")

# ===== 上传 =====
file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.info("上传数据")
    st.stop()

# ===== 加载 =====
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    return df

with st.spinner("加载中"):
    df = load_data(file)

st.success(f"数据: {len(df)} 行")

# ===== 特征：只用K线 =====
def prepare(df):
    df = df.copy()

    # 前高 / 前低
    df['high_prev'] = df['high'].shift(1).rolling(20).max()
    df['low_prev'] = df['low'].shift(1).rolling(20).min()

    # 信号
    df['signal'] = 0
    df.loc[df['close'] > df['high_prev'], 'signal'] = 1   # 突破多
    df.loc[df['close'] < df['low_prev'], 'signal'] = -1   # 跌破空

    df.dropna(inplace=True)
    return df

df = prepare(df)
st.success("特征完成（纯K线）")

# ===== 回测 =====
def backtest(df, rr=2.0):
    equity = [0]
    trades = []
    position = 0
    entry = 0
    stop = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平仓：止损或反转
        if position == 1:
            if row['low'] < stop:
                pnl = stop - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0
            elif row['signal'] == -1:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        if position == -1:
            if row['high'] > stop:
                pnl = entry - stop
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0
            elif row['signal'] == 1:
                pnl = entry - price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 开仓
        if prev['signal'] == 1 and position == 0:
            entry = price
            stop = row['low']
            position = 1

        if prev['signal'] == -1 and position == 0:
            entry = price
            stop = row['high']
            position = -1

        equity.append(equity[-1])

    return {
        "交易": len(trades),
        "胜率": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "盈利": sum(trades),
        "equity": equity
    }

# ===== 分割 =====
n = len(df)
test = df.iloc[int(n*0.6):]

res = backtest(test)

st.header("回测结果")
st.metric("交易", res["交易"])
st.metric("胜率", f"{res['胜率']*100:.2f}%")
st.metric("盈利", f"{res['盈利']:.2f}")

if res["交易"] > 0:
    st.line_chart(pd.Series(res["equity"]))
