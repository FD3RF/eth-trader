# -*- coding: utf-8 -*-
"""
结构突破 + 回踩确认 回测系统
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 突破回踩确认 回测")

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

st.success(f"数据行数: {len(df)}")

rr = st.slider("盈亏比", 1.0, 4.0, 2.0, 0.5)

trades = []
equity = [0]

for i in range(5, len(df)-5):

    prev = df.iloc[i-1]
    curr = df.iloc[i]

    # 发生突破
    breakout = curr['high'] > prev['high']

    if breakout:

        breakout_level = prev['high']

        # 等后面3根回踩
        for j in range(i+1, i+4):

            if j >= len(df)-1:
                break

            candle = df.iloc[j]

            # 回踩接近前高
            pullback = abs(candle['low'] - breakout_level) / breakout_level < 0.002

            # 回踩不破前低
            valid = candle['low'] > prev['low']

            bullish = candle['close'] > candle['open']

            if pullback and valid and bullish:

                entry = df.iloc[j+1]['open']
                stop = candle['low']
                risk = entry - stop

                if risk <= 0:
                    break

                target = entry + risk * rr

                # 持仓
                for k in range(j+1, len(df)):

                    future = df.iloc[k]

                    if future['low'] <= stop:
                        pnl = stop - entry
                        trades.append(pnl)
                        equity.append(equity[-1] + pnl)
                        break

                    if future['high'] >= target:
                        pnl = target - entry
                        trades.append(pnl)
                        equity.append(equity[-1] + pnl)
                        break

                break

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

st.write("平均盈利:", np.mean(trades))
st.write("最大盈利:", np.max(trades))
st.write("最大亏损:", np.min(trades))
