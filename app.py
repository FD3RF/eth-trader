# -*- coding: utf-8 -*-
"""
K线 + 成交量 + 突破策略（极简可跑版）
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 K线 + 成交量 策略")

# ======================
# 上传
# ======================
file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.info("上传文件")
    st.stop()

# ======================
# 加载兼容字段
# ======================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    # 时间
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # 成交量字段兼容
    if 'volume' in df.columns:
        df['volume'] = df['volume']
    elif 'vol' in df.columns:
        df['volume'] = df['vol']
    elif 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    elif 'quote_volume' in df.columns:
        df['volume'] = df['quote_volume']
    else:
        st.error("缺少成交量字段：volume / vol / tick_volume / quote_volume")
        st.write(df.columns.tolist())
        st.stop()

    return df

df = load_data(file)
st.success(f"数据：{len(df)} 行")

# 必须字段
for c in ['open','high','low','close','volume']:
    if c not in df.columns:
        st.error(f"缺少字段: {c}")
        st.stop()

# ======================
# 特征：K线突破 + 放量
# ======================
def build(df):
    df = df.copy()

    # 单K线突破：当前收盘 > 前一根高点
    df['break_high'] = df['close'] > df['high'].shift(1)

    # 放量：当前量 > 最近20均量 * 倍数
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] > df['vol_ma20'] * 1.5

    # 多头信号
    df['signal'] = (df['break_high'] & df['vol_spike']).astype(int)

    df.dropna(inplace=True)
    return df

df = build(df)
st.write("特征完成")

# ======================
# 分割
# ======================
n = len(df)
train = df.iloc[:int(n*0.6)]
test = df.iloc[int(n*0.6):]

# ======================
# 回测
# ======================
def backtest(df):
    equity = [0]
    trades = []

    position = 0
    entry = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平仓条件：跌破均线或信号消失
        if position == 1 and (row['close'] < row['close'].rolling(20).mean().iloc[i] or prev['signal'] == 0):
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开仓：前一根突破+放量
        if prev['signal'] == 1 and position == 0:
            entry = price
            position = 1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "total": sum(trades),
        "equity": equity
    }

res = backtest(test)

# ======================
# 输出
# ======================
st.header("回测")
st.metric("交易次数", res['trades'])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("盈利", f"{res['total']:.2f}")

if res['trades'] > 0:
    st.line_chart(pd.Series(res['equity']))

st.success("完成")
