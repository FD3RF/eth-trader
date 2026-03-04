# -*- coding: utf-8 -*-
"""
5分钟 单K线突破 + 成交量 回测系统
逻辑：
1. 当前 high > 前一根 high
2. 当前收阳
3. 成交量 > 20均量
止损 = 前一根 low
止盈 = 风险 × 2
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 单K线突破 + 成交量 回测")

# 上传CSV
uploaded = st.file_uploader("上传 CSV", type=["csv"])
if uploaded is None:
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    df.sort_index(inplace=True)
    return df

df = load_data(uploaded)

required = ['open','high','low','close']
for col in required:
    if col not in df.columns:
        st.error(f"缺少字段: {col}")
        st.stop()

if 'volume' not in df.columns:
    if 'vol' in df.columns:
        df['volume'] = df['vol']
    else:
        st.error("缺少 volume 字段")
        st.stop()

st.success(f"数据行数: {len(df)}")

# ===== 参数 =====
rr = st.slider("盈亏比 (Risk Reward)", 1.0, 5.0, 2.0, 0.5)
vol_mult = st.slider("放量倍数", 1.0, 3.0, 1.2, 0.1)

# ===== 特征 =====
df['vol_ma20'] = df['volume'].rolling(20).mean()

trades = []
equity = [0]
position = False

for i in range(21, len(df)-1):

    prev = df.iloc[i-1]
    curr = df.iloc[i]
    next_candle = df.iloc[i+1]

    # 条件
    breakout = curr['high'] > prev['high']
    bullish = curr['close'] > curr['open']
    vol_ok = curr['volume'] > curr['vol_ma20'] * vol_mult

    if not position and breakout and bullish and vol_ok:

        entry = next_candle['open']
        stop = prev['low']
        risk = entry - stop

        if risk <= 0:
            continue

        target = entry + risk * rr
        position = True

        # 持仓管理
        for j in range(i+1, len(df)):

            candle = df.iloc[j]

            # 止损
            if candle['low'] <= stop:
                pnl = stop - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = False
                break

            # 止盈
            if candle['high'] >= target:
                pnl = target - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = False
                break

# ===== 结果 =====
if len(trades) == 0:
    st.warning("无交易")
    st.stop()

win_rate = sum(1 for t in trades if t > 0) / len(trades)
profit = sum(trades)

st.header("回测结果")
st.metric("交易次数", len(trades))
st.metric("胜率", f"{win_rate*100:.2f}%")
st.metric("总盈利", f"{profit:.2f}")

st.line_chart(pd.Series(equity))

# 统计信息
st.subheader("统计细节")
st.write("平均盈利:", np.mean(trades))
st.write("最大盈利:", np.max(trades))
st.write("最大亏损:", np.min(trades))
