# -*- coding: utf-8 -*-
"""
进场交易计划提示
基于纯K回测逻辑
只提示，不交易
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="进场计划", layout="wide")
st.title("🚀 进场交易计划提示")

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

# ================================
# 特征
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
# 信号
# ================================
df['signal'] = 0

df.loc[
    (df['close'] > df['high_max']) &
    (df['body_ratio'] > 0.35) &
    (df['close'] > df['open']),
    'signal'
] = 1

df.loc[
    (df['close'] < df['low_min']) &
    (df['body_ratio'] > 0.35) &
    (df['close'] < df['open']),
    'signal'
] = -1

# ================================
# 最新计划
# ================================
st.header("最新交易计划")

latest = df.iloc[-1]

if latest['signal'] == 1:
    st.success("📈 多头计划：突破 + 强阳")
    st.write("""
    进场条件：
    - 突破前高
    - 强实体阳线
    - 结构顺势

    止损：
    - 跌破结构
    - 强反转

    退出：
    - 盈利目标
    - 信号反转
    """)

elif latest['signal'] == -1:
    st.error("📉 空头计划：跌破 + 强阴")
    st.write("""
    进场条件：
    - 跌破前低
    - 强实体阴线
    - 结构顺势

    止损：
    - 突破结构
    - 强反转

    退出：
    - 盈利目标
    - 信号反转
    """)

else:
    st.info("⏳ 无计划：等待结构")
    st.write("""
    不进场原因：
    - 无突破
    - 无强实体
    - 结构未确认
    """)

# ================================
# 历史统计
# ================================
st.header("历史统计")

st.write("多头次数:", (df['signal'] == 1).sum())
st.write("空头次数:", (df['signal'] == -1).sum())
st.write("无信号:", (df['signal'] == 0).sum())

st.line_chart(df['close'])
st.line_chart(df['signal'])

st.success("计划生成完成")
