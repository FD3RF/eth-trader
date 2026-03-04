# -*- coding: utf-8 -*-
"""
5分钟合约：K线突破 + 成交量 + 回调进场（稳定版）
作者：AI
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("🚀 稳定版：突破+回调+成交量")

# 上传
file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

# 加载
df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

# 字段兼容
def col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None

df['open'] = col(df,'open')
df['high'] = col(df,'high')
df['low'] = col(df,'low')
df['close'] = col(df,'close')
df['volume'] = col(df,'volume','vol')

for c in ['open','high','low','close','volume']:
    if df[c] is None:
        st.error(f"缺少字段: {c}")
        st.write(df.columns)
        st.stop()

df = df.dropna().reset_index(drop=True)

# ====== 策略参数 ======
lookback = 20                 # 突破周期
volume_mult = 2.0             # 放量倍数
pullback_ratio = 0.5          # 回调比例
stop_loss_pct = 0.01          # 单笔止损 1%
rr = 2.0                      # 盈亏比

# ====== 特征 ======
df['high_max'] = df['high'].rolling(lookback).max()
df['low_min'] = df['low'].rolling(lookback).min()
df['volume_ma'] = df['volume'].rolling(lookback).mean()

# ====== 信号 ======
df['break_up'] = df['close'] > df['high_max'].shift(1)
df['break_down'] = df['close'] < df['low_min'].shift(1)

df['volume_ok'] = df['volume'] > df['volume_ma'] * volume_mult

# 回调条件：价格回踩不超过突破幅度的一半
df['pullback_ok'] = (
    (df['close'] >= df['high_max'].shift(1) * (1 - pullback_ratio * 0.01))
)

df['long_signal'] = df['break_up'] & df['volume_ok'] & df['pullback_ok']
df['short_signal'] = df['break_down'] & df['volume_ok']

# ====== 回测 ======
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['open']

        # 平仓条件
        if position == 1:
            if row['close'] < entry * (1 - stop_loss_pct):
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0
            elif row['short_signal']:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        elif position == -1:
            if row['close'] > entry * (1 + stop_loss_pct):
                pnl = entry - price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0
            elif row['long_signal']:
                pnl = entry - price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 开仓
        if position == 0:
            if row['long_signal']:
                entry = price
                position = 1
            elif row['short_signal']:
                entry = price
                position = -1

        equity.append(equity[-1])

    return {
        "trades": trades,
        "equity": equity
    }

# ====== 执行 ======
res = backtest(df)

# 统计
trades = res['trades']
equity = res['equity']

st.header("回测结果")
st.metric("交易次数", len(trades))
st.metric("胜率", f"{sum(1 for p in trades if p>0)/len(trades)*100:.2f}%" if trades else "0")
st.metric("总盈利", f"{sum(trades):.2f}")

# 曲线
st.line_chart(pd.Series(equity))
