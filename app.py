# -*- coding: utf-8 -*-
"""
5分钟 趋势强化 + 突破 + ATR风控 回测系统
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 强化趋势突破回测系统")

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
rr = st.slider("盈亏比", 1.5, 4.0, 2.0, 0.5)
fee_rate = st.slider("单边手续费 (%)", 0.0, 0.1, 0.04) / 100
atr_mult = st.slider("ATR止损倍数", 1.0, 3.0, 1.5, 0.1)

# ===== 指标计算 =====
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema200'] = df['close'].ewm(span=200).mean()

# EMA斜率（趋势是否加速）
df['ema_slope'] = df['ema50'] - df['ema50'].shift(5)

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

    # ===== 趋势过滤 =====
    trend = (
        curr['ema50'] > curr['ema200'] and
        curr['ema_slope'] > 0
    )

    breakout = curr['high'] > prev['high']
    bullish = curr['close'] > curr['open']
    vol_ok = curr['volume'] > curr['vol_ma20']

    if trend and breakout and bullish and vol_ok:

        entry = df.iloc[i+1]['open']

        stop = entry - curr['atr'] * atr_mult
        risk = entry - stop
        target = entry + risk * rr

        for j in range(i+1, len(df)):

            future = df.iloc[j]

            # 止损
            if future['low'] <= stop:
                pnl = stop - entry
                pnl -= entry * fee_rate * 2
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                break

            # 止盈
            if future['high'] >= target:
                pnl = target - entry
                pnl -= entry * fee_rate * 2
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

st.subheader("统计信息")
st.write("平均盈利:", np.mean(trades))
st.write("最大盈利:", np.max(trades))
st.write("最大亏损:", np.min(trades))
