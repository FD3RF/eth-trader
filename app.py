# -*- coding: utf-8 -*-
"""
纯K战术：前高/前低突破 + K线实体
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="纯K战术", layout="wide")
st.title("🚀 纯 K 战术（价格行为）")

file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

# =========================
# 加载
# =========================
df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

df['open'] = df['open']
df['high'] = df['high']
df['low'] = df['low']
df['close'] = df['close']
df['volume'] = df['vol']

st.write(f"数据行: {len(df)}")

# =========================
# 特征：前高/前低
# =========================
lookback = 20

df['high_max'] = df['high'].rolling(lookback).max().shift(1)
df['low_min'] = df['low'].rolling(lookback).min().shift(1)

# K线实体（强势）
df['body'] = (df['close'] - df['open']).abs()
df['range'] = df['high'] - df['low']
df['body_ratio'] = df['body'] / (df['range'] + 1e-9)

# 实体必须够大（避免十字）
strong_body = df['body_ratio'] > 0.4

df.dropna(inplace=True)

# =========================
# 信号
# =========================
df['signal'] = 0

# 多头：
df.loc[
    (df['close'] > df['high_max']) &
    strong_body &
    (df['close'] > df['open']),
    'signal'
] = 1

# 空头：
df.loc[
    (df['close'] < df['low_min']) &
    strong_body &
    (df['close'] < df['open']),
    'signal'
] = -1

# =========================
# 回测
# =========================
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0

    rr = 2.0
    min_hold = 2
    hold = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['open']
        high = row['high']
        low = row['low']

        if position != 0:
            hold += 1
        else:
            hold = 0

        # ===== 平多 =====
        if position == 1:
            stop = entry - (row['range'] * 0.5)
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

            elif hold >= min_hold and row['signal'] == -1:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # ===== 平空 =====
        if position == -1:
            stop = entry + (row['range'] * 0.5)
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

# =========================
# 展示
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
