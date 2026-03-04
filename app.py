# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 趋势过滤 + 突破 回测")

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
    df['volume'] = df['vol']

st.success(f"数据行数: {len(df)}")

# 参数
rr = st.slider("盈亏比", 1.0, 4.0, 2.0, 0.5)

# 趋势
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema200'] = df['close'].ewm(span=200).mean()

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    )
)
df['atr'] = df['tr'].rolling(14).mean()

# 成交量
df['vol_ma20'] = df['volume'].rolling(20).mean()

trades = []
equity = [0]

for i in range(200, len(df)-1):

    prev = df.iloc[i-1]
    curr = df.iloc[i]

    trend = curr['ema50'] > curr['ema200']

    breakout = curr['high'] > prev['high']
    bullish = curr['close'] > curr['open']
    vol_ok = curr['volume'] > curr['vol_ma20']

    if trend and breakout and bullish and vol_ok:

        entry = df.iloc[i+1]['open']
        stop = entry - curr['atr'] * 1.5
        target = entry + (entry - stop) * rr

        for j in range(i+1, len(df)):

            future = df.iloc[j]

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
