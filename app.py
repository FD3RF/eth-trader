# -*- coding: utf-8 -*-
"""
真实回测版本：允许亏损，成交假设更合理
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="真实回测", layout="wide")
st.title("🚀 真实回测版本")

file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

# 字段
df['open'] = df['open']
df['high'] = df['high']
df['low'] = df['low']
df['close'] = df['close']
df['volume'] = df['vol']

st.write(f"数据: {len(df)} 行")

# 多周期趋势
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['trend_up'] = df['ema20'] > df['ema50']
df['trend_down'] = df['ema20'] < df['ema50']

# 突破 + 放量
lookback = 20
vol_mult = 1.5
stop_pct = 0.01
rr = 2.0

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

# 回测（真实）
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0
    stop = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['open']
        high = row['high']
        low = row['low']

        # 平多（止损 / 止盈 / 信号反转）
        if position == 1:
            stop_loss = entry * (1 - stop_pct)
            take_profit = entry + (entry - stop_loss) * rr

            if low <= stop_loss:
                pnl = stop_loss - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif high >= take_profit:
                pnl = take_profit - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif row['signal'] == -1:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 平空
        if position == -1:
            stop_loss = entry * (1 + stop_pct)
            take_profit = entry - (stop_loss - entry) * rr

            if high >= stop_loss:
                pnl = entry - stop_loss
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif low <= take_profit:
                pnl = entry - take_profit
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif row['signal'] == 1:
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

            elif row['signal'] == -1:
                entry = price
                stop = entry * (1 + stop_pct)
                position = -1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "total": sum(trades),
        "equity": equity,
        "avg": np.mean(trades) if trades else 0,
        "max_loss": min(trades) if trades else 0,
        "max_profit": max(trades) if trades else 0
    }

res = backtest(df)

st.header("回测结果")
st.metric("交易", res["trades"])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("盈利", f"{res['total']:.2f}")

st.write("平均单笔:", res["avg"])
st.write("最大盈利:", res["max_profit"])
st.write("最大亏损:", res["max_loss"])

if res["trades"] > 0:
    st.line_chart(pd.Series(res["equity"]))

st.success("运行完毕")
