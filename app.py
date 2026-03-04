# -*- coding: utf-8 -*-
"""
K线 + 成交量 多空策略（稳定回测版）
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 K线 + 成交量 多空策略")

# ====== 上传 ======
file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.info("上传数据")
    st.stop()

# ====== 加载 ======
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    # 成交量字段兼容
    if 'volume' in df.columns:
        df['volume'] = df['volume']
    elif 'vol' in df.columns:
        df['volume'] = df['vol']
    elif 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    else:
        st.error("缺少成交量字段：volume / vol / tick_volume")
        st.write(df.columns.tolist())
        st.stop()

    return df

with st.spinner("加载中"):
    df = load_data(file)

st.success(f"数据行数: {len(df)}")

# ====== 特征 ======
def prepare(df):
    df = df.copy()

    # 均线（趋势）
    df['ma20'] = df['close'].rolling(20).mean()

    # K线方向
    df['k_up'] = (df['close'] > df['close'].shift(1)).astype(int)

    # 成交量方向
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_up'] = (df['volume'] > df['vol_ma']).astype(int)

    # 信号：K上涨 + 放量 + 在均线上
    df['signal'] = (
        (df['k_up'] == 1) &
        (df['vol_up'] == 1) &
        (df['close'] > df['ma20'])
    ).astype(int)

    df.dropna(inplace=True)
    return df

df = prepare(df)
st.success("特征完成")

# ====== 回测 ======
def backtest(df):
    equity = [0]
    trades = []
    position = 0
    entry = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 止损：跌破均线
        if position == 1:
            if row['close'] < row['ma20']:
                pnl = price - entry
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0

        # 开仓
        if prev['signal'] == 1 and position == 0:
            entry = price
            position = 1

        equity.append(equity[-1])

    return {
        "交易次数": len(trades),
        "胜率": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "盈利": sum(trades),
        "equity": equity
    }

# 分割
n = len(df)
test = df.iloc[int(n*0.6):]

res = backtest(test)

st.header("回测结果")
st.metric("交易", res["交易次数"])
st.metric("胜率", f"{res['胜率']*100:.2f}%")
st.metric("盈利", f"{res['盈利']:.2f}")

if res["交易次数"] > 0:
    st.line_chart(pd.Series(res["equity"]))
