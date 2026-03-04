# -*- coding: utf-8 -*-
"""
稳定增强版：突破 + 回调 + 放量 + 多周期
可回测
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("🚀 稳定增强策略")

# =========================
# 上传
# =========================
file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

# =========================
# 加载
# =========================
df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

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
        st.stop()

df = df.dropna().reset_index(drop=True)

# =========================
# 参数（可调）
# =========================
lookback = 20
volume_mult = 2.0
pullback = 0.5
stop_pct = 0.01
risk_per_trade = 0.01    # 单笔风险1%
capital = 10000          # 假设资金

# =========================
# 多周期趋势
# =========================
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()

df['trend_up'] = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50'])
df['trend_down'] = (df['close'] < df['ema20']) & (df['ema20'] < df['ema50'])

# =========================
# 特征：突破 + 放量
# =========================
df['high_max'] = df['high'].rolling(lookback).max()
df['low_min'] = df['low'].rolling(lookback).min()
df['volume_ma'] = df['volume'].rolling(lookback).mean()

df['break_up'] = df['close'] > df['high_max'].shift(1)
df['break_down'] = df['close'] < df['low_min'].shift(1)

df['volume_ok'] = df['volume'] > df['volume_ma'] * volume_mult

# 回调：不跌破突破位一定比例
df['pullback_ok'] = df['close'] >= df['high_max'].shift(1) * (1 - pullback * 0.01)

# 信号
df['long_signal'] = df['break_up'] & df['volume_ok'] & df['pullback_ok'] & df['trend_up']
df['short_signal'] = df['break_down'] & df['volume_ok'] & df['trend_down']

# =========================
# 回测（含动态止损 + 仓位）
# =========================
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0
    stop = 0
    size = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['open']

        # 平多
        if position == 1:
            if price <= stop or row['short_signal']:
                pnl = (price - entry) * size
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 平空
        elif position == -1:
            if price >= stop or row['long_signal']:
                pnl = (entry - price) * size
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 开仓
        if position == 0:
            if row['long_signal']:
                entry = price
                stop = entry * (1 - stop_pct)
                risk = entry - stop
                size = (capital * risk_per_trade) / risk if risk > 0 else 0
                position = 1

            elif row['short_signal']:
                entry = price
                stop = entry * (1 + stop_pct)
                risk = stop - entry
                size = (capital * risk_per_trade) / risk if risk > 0 else 0
                position = -1

        equity.append(equity[-1])

    return trades, equity

# =========================
# 执行
# =========================
trades, equity = backtest(df)

st.header("回测结果")
st.metric("交易次数", len(trades))
st.metric("胜率", f"{sum(1 for p in trades if p>0)/len(trades)*100:.2f}%" if trades else "0")
st.metric("总盈利", f"{sum(trades):.2f}")

# 曲线
st.line_chart(pd.Series(equity))
