# -*- coding: utf-8 -*-
"""
实战级 5分钟多空策略
核心：
- K线突破
- 成交量放量
- 多周期趋势
- ATR动态止损
- 风控
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="实战策略", layout="wide")
st.title("🚀 实战级多空策略（更稳）")

file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

# =========================
# 加载数据（兼容字段）
# =========================
df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

# 字段映射
df['open'] = df['open']
df['high'] = df['high']
df['low'] = df['low']
df['close'] = df['close']
df['volume'] = df['vol']

st.write(f"数据行: {len(df)}")

# =========================
# 多周期趋势
# =========================
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema200'] = df['close'].ewm(span=200).mean()

df['trend_up'] = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
df['trend_down'] = (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])

# =========================
# 动态特征
# =========================
lookback = 20

df['high_max'] = df['high'].rolling(lookback).max().shift(1)
df['low_min'] = df['low'].rolling(lookback).min().shift(1)

df['vol_ma'] = df['volume'].rolling(lookback).mean().shift(1)
df['vol_std'] = df['volume'].rolling(lookback).std().shift(1)
df['vol_threshold'] = df['vol_ma'] + df['vol_std'] * 0.5

# ATR（动态止损）
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1)

df['atr'] = tr.rolling(14).mean()

df.dropna(inplace=True)

# =========================
# 信号
# =========================
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

# =========================
# 回测（真实止损止盈）
# =========================
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0

    stop_pct = 0.01  # 基础止损 1%
    rr = 2.0         # 盈亏比

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']
        high = row['high']
        low = row['low']

        # ========== 平多 ==========
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

        # ========== 平空 ==========
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

        # ========== 开仓 ==========
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

# =========================
# 回测展示
# =========================
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
