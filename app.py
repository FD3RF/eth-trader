# -*- coding: utf-8 -*-
"""
K线 + 交易量 策略（极简可交易版）
作者：AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="K线量价策略", layout="wide")
st.title("📈 K线 + 交易量 策略")

# 上传
uploaded = st.file_uploader("上传 CSV", type=["csv"])
if uploaded is None:
    st.info("上传文件")
    st.stop()

# 加载
@st.cache_data
def load(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    # 时间
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    return df

df = load(uploaded)
st.success(f"数据：{len(df)} 行")

# 字段
def get(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None

df['open'] = get(df,'open')
df['high'] = get(df,'high')
df['low'] = get(df,'low')
df['close'] = get(df,'close')
df['volume'] = get(df,'volume','vol','tick_volume','quote_volume')

for c in ['open','high','low','close','volume']:
    if df[c] is None:
        st.error(f"缺少字段：{c}")
        st.write(df.columns)
        st.stop()

# ===== 特征：K线结构 + 量 =====
def make_features(df):
    df = df.copy()

    # K线实体
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_shadow'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_shadow'] = df[['open','close']].min(axis=1) - df['low']

    # 量能
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']

    # 结构
    df['bull_k'] = (df['close'] > df['open']) & (df['body'] > df['upper_shadow'])
    df['bear_k'] = (df['close'] < df['open']) & (df['body'] > df['lower_shadow'])

    # 未来标签：5根K线是否上涨
    future = df['close'].pct_change(5).shift(-5)
    df['target'] = (future > 0).astype(int)

    df.dropna(inplace=True)
    return df

with st.spinner("特征生成"):
    df = make_features(df)

st.success("特征完成")

# 分割
n = len(df)
train = df.iloc[:int(n*0.6)]
val = df.iloc[int(n*0.6):int(n*0.8)]
test = df.iloc[int(n*0.8):]

st.info(f"训练 {len(train)} | 验证 {len(val)} | 测试 {len(test)}")

features = ['body','upper_shadow','lower_shadow','vol_ratio','bull_k','bear_k']

# 模型
def train_model(train, val):
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']

    if y_train.nunique() < 2:
        st.error("标签只有一类")
        st.stop()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    st.write("验证准确率:", accuracy_score(y_val, pred))

    return model

with st.spinner("训练模型"):
    model = train_model(train, val)

# 回测
def backtest(df, probs, th):
    df = df.copy()
    df['prob'] = probs
    df['signal'] = (df['prob'] >= th).astype(int)

    equity = [0]
    position = 0
    entry = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平仓：趋势反转
        if position == 1 and row['close'] < row['open']:
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开仓：前一根强阳 + 信号
        if prev['bull_k'] and prev['vol_ratio'] > 1.2 and prev['signal'] == 1:
            entry = price
            position = 1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "profit": sum(trades),
        "equity": equity
    }

# 验证回测
val_probs = model.predict_proba(val[features])[:,1]
th = np.quantile(val_probs, 0.7)

res = backtest(val, val_probs, th)

st.header("回测结果")
st.metric("交易", res['trades'])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("盈利", f"{res['profit']:.2f}")

if res['trades'] > 0:
    st.line_chart(pd.Series(res['equity']))

st.success("完成")
