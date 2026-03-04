# -*- coding: utf-8 -*-
"""
强化滚动验证（更大样本）
纯K结构
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="强化滚动验证", layout="wide")
st.title("🚀 强化滚动验证（更大样本）")

file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]

df['open'] = df['open']
df['high'] = df['high']
df['low'] = df['low']
df['close'] = df['close']
df['volume'] = df['vol']

st.write(f"数据行: {len(df)}")

# ================================
# 纯K特征
# ================================
def build_features(df):
    df = df.copy()
    lookback = 20

    df['high_max'] = df['high'].rolling(lookback).max().shift(1)
    df['low_min'] = df['low'].rolling(lookback).min().shift(1)

    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-9)

    df.dropna(inplace=True)
    return df

df = build_features(df)

# ================================
# 回测函数
# ================================
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0

    rr = 2.5
    min_hold = 3
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

        strong_body = row['body_ratio'] > 0.35

        # 信号
        signal = 0
        if (row['close'] > row['high_max']) and strong_body and (row['close'] > row['open']):
            signal = 1
        elif (row['close'] < row['low_min']) and strong_body and (row['close'] < row['open']):
            signal = -1

        # 平多
        if position == 1:
            stop = entry - (row['range'] * 0.5)
            take = entry + (entry - stop) * rr

            if low <= stop:
                trades.append(stop - entry)
                equity.append(equity[-1] + (stop - entry))
                position = 0

            elif high >= take:
                trades.append(take - entry)
                equity.append(equity[-1] + (take - entry))
                position = 0

            elif hold >= min_hold and signal == -1:
                trades.append(price - entry)
                equity.append(equity[-1] + (price - entry))
                position = 0

        # 平空
        if position == -1:
            stop = entry + (row['range'] * 0.5)
            take = entry - (stop - entry) * rr

            if high >= stop:
                trades.append(entry - stop)
                equity.append(equity[-1] + (entry - stop))
                position = 0

            elif low <= take:
                trades.append(entry - take)
                equity.append(equity[-1] + (entry - take))
                position = 0

            elif hold >= min_hold and signal == 1:
                trades.append(entry - price)
                equity.append(equity[-1] + (entry - price))
                position = 0

        # 开仓
        if position == 0:
            if signal == 1:
                entry = price
                position = 1
            elif signal == -1:
                entry = price
                position = -1

        equity.append(equity[-1])

    return trades, equity

# ================================
# 强滚动验证
# ================================
n = len(df)

# 滚动块：20%
fold = int(n * 0.2)
results = []

# 更多区段
for i in range(0, n - fold, fold):
    test = df.iloc[i:i+fold]
    trades, equity = backtest(test)

    if trades:
        results.append({
            "start": i,
            "end": i+fold,
            "trades": len(trades),
            "win_rate": sum(1 for p in trades if p > 0) / len(trades),
            "total": sum(trades),
            "avg": np.mean(trades),
            "max_loss": min(trades),
            "max_profit": max(trades)
        })

# ================================
# 统计
# ================================
st.header("滚动验证结果")

if results:
    df_res = pd.DataFrame(results)
    st.write(df_res)

    st.write("区段数量:", len(results))
    st.write("平均胜率:", df_res['win_rate'].mean())
    st.write("平均单笔:", df_res['avg'].mean())
    st.write("胜率波动:", df_res['win_rate'].std())
    st.write("单笔波动:", df_res['avg'].std())

    st.success("验证完成")
else:
    st.write("无结果")
