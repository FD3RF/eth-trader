# -*- coding: utf-8 -*-
"""
K线 + 成交量 策略（实盘友好版）
思路：
- 突破前高
- 放量确认
- 回调后进场
- 固定止损 + 盈亏比
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="K线+成交量策略", layout="wide")
st.title("📈 K线 + 成交量 策略")

# 上传
uploaded = st.file_uploader("上传 CSV", type=["csv"])
if uploaded is None:
    st.info("上传数据")
    st.stop()

# 加载
@st.cache_data
def load(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    return df

df = load(uploaded)
st.write("数据行:", len(df))

# 字段兼容
def get(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None

df['open'] = get(df,'open')
df['high'] = get(df,'high')
df['low'] = get(df,'low')
df['close'] = get(df,'close')
df['volume'] = get(df,'volume','vol')

for c in ['open','high','low','close','volume']:
    if df[c] is None:
        st.error(f"缺少字段: {c}")
        st.stop()

# 策略参数
st.sidebar.header("策略参数")
lookback = st.sidebar.slider("突破周期", 5, 50, 20)
rr = st.sidebar.slider("盈亏比", 1.0, 4.0, 2.0)
vol_mult = st.sidebar.slider("放量倍数", 1.0, 3.0, 1.5)

# 特征：前高 + 放量
df['high_roll'] = df['high'].rolling(lookback).max()
df['vol_ma'] = df['volume'].rolling(lookback).mean()

# 信号
df['break'] = df['close'] > df['high_roll'].shift(1)
df['vol_confirm'] = df['volume'] > df['vol_ma'] * vol_mult
df['signal'] = (df['break'] & df['vol_confirm']).astype(int)

# 回测
def backtest(df):
    equity = [0]
    position = 0
    entry = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平仓条件
        if position == 1:
            stop = entry * 0.99
            target = entry + (entry - stop) * rr

            if row['low'] <= stop:
                pnl = stop - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif row['high'] >= target:
                pnl = target - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif row['close'] < row['high_roll'] * 0.995:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 开仓条件
        if prev['signal'] == 1 and position == 0:
            entry = price
            position = 1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "profit": sum(trades),
        "equity": equity
    }

# 执行回测
res = backtest(df)

st.header("回测结果")
st.metric("交易", res["trades"])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("盈利", f"{res['profit']:.2f}")

if res["trades"] > 0:
    st.line_chart(pd.Series(res["equity"]))

st.success("运行完成")
