# -*- coding: utf-8 -*-
"""
稳定终极版：突破 + 放量 + 多周期 + 风控
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="终极优化策略", layout="wide")
st.title("🚀 终极优化策略")

# 上传
file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

# 加载
df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]
st.write(f"数据行数: {len(df)}")
st.write("字段:", df.columns.tolist())

# 字段映射
df['open'] = df['open']
df['high'] = df['high']
df['low'] = df['low']
df['close'] = df['close']
df['volume'] = df['vol']

# 多周期趋势
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()

df['trend_up'] = df['ema20'] > df['ema50']
df['trend_down'] = df['ema20'] < df['ema50']

# 特征：突破 + 放量
lookback = 20
vol_mult = 1.5
stop_pct = 0.01

df['high_max'] = df['high'].rolling(lookback).max().shift(1)
df['low_min'] = df['low'].rolling(lookback).min().shift(1)
df['vol_ma'] = df['volume'].rolling(lookback).mean().shift(1)

df.dropna(inplace=True)

# 信号
df['signal'] = 0
df.loc[
    (df['close'] > df['high_max']) &
    (df['volume'] > df['vol_ma'] * vol_mult) &
    df['trend_up'],
    'signal'
] = 1

df.loc[
    (df['close'] < df['low_min']) &
    (df['volume'] > df['vol_ma'] * vol_mult) &
    df['trend_down'],
    'signal'
] = -1

# 回测
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0
    stop = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['open']

        # 平多
        if position == 1:
            if price <= stop or row['signal'] == 0:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 平空
        if position == -1:
            if price >= stop or row['signal'] == 0:
                pnl = entry - price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 开仓
        if position == 0:
            if row['signal'] == 1:
                entry = price
                stop = entry * (1 - stop_pct)
                position = 1

            if row['signal'] == -1:
                entry = price
                stop = entry * (1 + stop_pct)
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
