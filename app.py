# -*- coding: utf-8 -*-
"""
纸交易模拟版（纯K优化）
用于实盘验证，不下单
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="纸交易模拟(优化版)", layout="wide")
st.title("🚀 纸交易模拟（纯K优化）")

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
# 纯K特征构建
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
# 进场信号模块（优化版）
# =========================
def generate_signal(df, i):
    row = df.iloc[i]

    close = row['close']
    high = row['high']
    low = row['low']

    high_max = row['high_max']
    low_min = row['low_min']
    body_ratio = row['body_ratio']
    volume = row['volume']

    # 成交量均线
    vol_ma = df['volume'].rolling(20).mean().iloc[i]

    # 是否在区间中部（避免中部假突破）
    in_middle = (close > low_min * 1.05) and (close < high_max * 0.95)

    # 实体强度
    strong_body = body_ratio > 0.45

    # 成交量确认
    valid_volume = volume > vol_ma

    # 突破幅度（避免刚碰就算突破）
    break_up_strength = (close - high_max) / high_max
    break_down_strength = (low_min - close) / low_min

    # 多头突破有效
    if (close > high_max) and strong_body and valid_volume:
        if break_up_strength > 0.002 and not in_middle:
            return 1

    # 空头突破有效
    if (close < low_min) and strong_body and valid_volume:
        if break_down_strength > 0.002 and not in_middle:
            return -1

    return 0


# =========================
# 模拟交易
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

        signal = generate_signal(df, i)

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
# 统计与展示
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

st.success("模拟完成（纯K优化版）")
