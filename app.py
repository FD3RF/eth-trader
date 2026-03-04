# -*- coding: utf-8 -*-
"""
纸交易模拟版（纯K）
用于实盘验证，不下单
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="纸交易模拟", layout="wide")
st.title("🚀 纸交易模拟（实盘验证）")

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

# =========================
# 纯K特征
# =========================
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

# =========================
# 模拟交易记录
# =========================
records = []

def simulate(df):
    position = 0
    entry = 0
    hold = 0
    rr = 2.5
    min_hold = 3

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
                records.append({"pnl": stop - entry})
                position = 0

            elif high >= take:
                records.append({"pnl": take - entry})
                position = 0

            elif hold >= min_hold and signal == -1:
                records.append({"pnl": price - entry})
                position = 0

        # 平空
        if position == -1:
            stop = entry + (row['range'] * 0.5)
            take = entry - (stop - entry) * rr

            if high >= stop:
                records.append({"pnl": entry - stop})
                position = 0

            elif low <= take:
                records.append({"pnl": entry - take})
                position = 0

            elif hold >= min_hold and signal == 1:
                records.append({"pnl": entry - price})
                position = 0

        # 开仓
        if position == 0:
            if signal == 1:
                entry = price
                position = 1
            elif signal == -1:
                entry = price
                position = -1

    return records

# =========================
# 执行模拟
# =========================
records = simulate(df)

# =========================
# 统计
# =========================
st.header("模拟结果")

if records:
    df_rec = pd.DataFrame(records)
    st.write(df_rec)

    st.metric("交易数", len(df_rec))
    st.metric("胜率", f"{(df_rec['pnl'] > 0).mean() * 100:.2f}%")
    st.metric("总盈利", df_rec['pnl'].sum())
    st.metric("平均单笔", df_rec['pnl'].mean())
    st.metric("最大盈利", df_rec['pnl'].max())
    st.metric("最大亏损", df_rec['pnl'].min())

    st.line_chart(df_rec['pnl'].cumsum())

else:
    st.write("无交易记录")

st.success("模拟完成（纸交易）")
