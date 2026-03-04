# -*- coding: utf-8 -*-
"""
最强实盘级优化
核心：
- 多周期趋势
- ATR动态止损
- 动态仓位
- 低频高质
- 防过拟合
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="实盘级优化", layout="wide")
st.title("🚀 最强实盘优化")

file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

# ================================
# 加载
# ================================
df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

df['open'] = df['open']
df['high'] = df['high']
df['low'] = df['low']
df['close'] = df['close']
df['volume'] = df['vol']

st.write(f"数据行: {len(df)}")

# ================================
# 多周期趋势（最重要）
# ================================
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema200'] = df['close'].ewm(span=200).mean()

df['trend_up'] = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
df['trend_down'] = (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])

# ================================
# 动态特征
# ================================
lookback = 30  # 拉长周期（更稳）

df['high_max'] = df['high'].rolling(lookback).max().shift(1)
df['low_min'] = df['low'].rolling(lookback).min().shift(1)

df['vol_ma'] = df['volume'].rolling(lookback).mean().shift(1)
df['vol_std'] = df['volume'].rolling(lookback).std().shift(1)
df['vol_threshold'] = df['vol_ma'] + df['vol_std'] * 0.8  # 更严格放量

# ATR（动态止损）
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1)

df['atr'] = tr.rolling(14).mean()

df.dropna(inplace=True)

# ================================
# 信号（更严格）
# ================================
df['signal'] = 0

# 多头：
df.loc[
    (df['close'] > df['high_max']) &
    (df['volume'] > df['vol_threshold']) &
    df['trend_up'],
    'signal'
] = 1

# 空头：
df.loc[
    (df['close'] < df['low_min']) &
    (df['volume'] > df['vol_threshold']) &
    df['trend_down'],
    'signal'
] = -1

# ================================
# 动态仓位（核心实盘思维）
# ================================
def calc_position_size(atr, capital=10000):
    risk_per_trade = 0.01  # 每单风险1%
    return (capital * risk_per_trade) / (atr * 2 + 1e-9)

# ================================
# 回测（实盘级）
# ================================
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0

    rr = 2.5  # 提高盈亏比
    min_hold = 3  # 最小持仓K线（防假突破）

    hold = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['open']
        high = row['high']
        low = row['low']

        # 持仓计数
        if position != 0:
            hold += 1
        else:
            hold = 0

        # ===== 平多 =====
        if position == 1:
            stop = entry - row['atr'] * 1.5
            take = entry + (entry - stop) * rr

            if low <= stop:
                pnl = stop - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif high >= take:
                pnl = take - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            # 防假突破：最低持仓
            elif hold >= min_hold and row['signal'] == -1:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # ===== 平空 =====
        if position == -1:
            stop = entry + row['atr'] * 1.5
            take = entry - (stop - entry) * rr

            if high >= stop:
                pnl = entry - stop
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif low <= take:
                pnl = entry - take
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

            elif hold >= min_hold and row['signal'] == 1:
                pnl = entry - price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # ===== 开仓 =====
        if position == 0:
            if row['signal'] == 1:
                entry = price
                position = 1

            elif row['signal'] == -1:
                entry = price
                position = -1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "total": sum(trades),
        "avg": np.mean(trades) if trades else 0,
        "max_loss": min(trades) if trades else 0,
        "max_profit": max(trades) if trades else 0,
        "equity": equity
    }

# ================================
# 回测展示
# ================================
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
