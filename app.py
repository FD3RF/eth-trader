# -*- coding: utf-8 -*-
"""
纯K突破 + 放量策略（字段兼容版）
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="纯K突破策略", layout="wide")
st.title("📈 纯K突破 + 放量 策略")

# 上传
uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
if uploaded_file is None:
    st.info("上传数据")
    st.stop()

# ================================
# 加载 + 字段兼容
# ================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    return df

df = load_data(uploaded_file)
st.write(f"数据行数: {len(df)}")
st.write("字段:", df.columns.tolist())

# ================================
# 字段映射（自动）
# ================================
def get_col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None

df['open'] = get_col(df, 'open')
df['high'] = get_col(df, 'high')
df['low'] = get_col(df, 'low')
df['close'] = get_col(df, 'close')

# 成交量字段可能是：
df['volume'] = get_col(df, 'volume', 'vol', 'volum', 'amount', 'qty')

for c in ['open','high','low','close']:
    if df[c] is None:
        st.error(f"缺少字段: {c}")
        st.stop()

# 成交量非必须（如果没有，后续放量逻辑自动关闭）
has_volume = df['volume'] is not None

# ================================
# 参数
# ================================
st.sidebar.header("策略参数")

lookback = st.sidebar.slider("突破周期(前高/前低)", 5, 50, 20)
vol_mult = st.sidebar.slider("放量倍数", 1.0, 3.0, 1.5) if has_volume else None
rr = st.sidebar.slider("盈亏比", 1.0, 4.0, 2.0)

# ================================
# 特征：前高前低 + 放量（可选）
# ================================
df['high_max'] = df['high'].rolling(lookback).max().shift(1)
df['low_min'] = df['low'].rolling(lookback).min().shift(1)

if has_volume:
    df['vol_ma'] = df['volume'].rolling(lookback).mean().shift(1)

# 信号
df['signal'] = 0

# 多头：突破前高
if has_volume:
    df.loc[
        (df['close'] > df['high_max']) &
        (df['volume'] > df['vol_ma'] * vol_mult),
        'signal'
    ] = 1
else:
    df.loc[df['close'] > df['high_max'], 'signal'] = 1

# 空头：跌破前低
if has_volume:
    df.loc[
        (df['close'] < df['low_min']) &
        (df['volume'] > df['vol_ma'] * vol_mult),
        'signal'
    ] = -1
else:
    df.loc[df['close'] < df['low_min'], 'signal'] = -1

df.dropna(inplace=True)

# ================================
# 回测（简化）
# ================================
def backtest(df):
    equity = [0]
    position = 0
    entry = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平仓
        if position == 1 and row['signal'] == 0:
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        if position == -1 and row['signal'] == 0:
            pnl = entry - price
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开仓
        if prev['signal'] == 1 and position == 0:
            entry = price
            position = 1

        if prev['signal'] == -1 and position == 0:
            entry = price
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
