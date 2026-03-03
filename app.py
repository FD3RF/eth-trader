# -*- coding: utf-8 -*-
"""
终极量化策略 - 实战优化版
版本：18.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="量化系统", layout="wide")
st.title("🚀 实战级未来收益策略")

# =========================
# 上传
# =========================
file = st.file_uploader("上传 CSV", type=["csv"])
if file is None:
    st.info("上传数据")
    st.stop()

# =========================
# 加载
# =========================
@st.cache_data
def load(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    return df

df = load(file)
st.success(f"数据：{len(df)} 行")

# =========================
# 字段兼容
# =========================
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
        st.error(f"缺少字段: {c}")
        st.write(df.columns.tolist())
        st.stop()

# =========================
# 特征与标签（实战级）
# =========================
def build(df):
    df = df.copy()

    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['trend_align'] = (df['close'] > df['ema50']).astype(int)

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)

    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    # ===== 标签：未来5根K线上涨 > 0.2% =====
    future = df['close'].pct_change(5).shift(-5)
    df['target'] = (future > 0.002).astype(int)

    df.dropna(inplace=True)
    return df

df = build(df)
st.success("特征完成")

# =========================
# 分割
# =========================
def split(df):
    n = len(df)
    return (
        df.iloc[:int(n*0.6)],
        df.iloc[int(n*0.8):],
        df.iloc[int(n*0.6):int(n*0.8)]
    )

train, test, val = split(df)

st.info(f"训练 {len(train)} | 验证 {len(val)} | 测试 {len(test)}")

feat = ['trend_align','rsi','volume_ratio']

# =========================
# 模型
# =========================
def train(train, val):
    X_train = train[feat]
    y_train = train['target']
    X_val = val[feat]
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
        eval_metric='logloss',
        early_stopping_rounds=30
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred = model.predict(X_val)
    st.write("验证准确率:", accuracy_score(y_val, pred))
    st.write("精确率:", precision_score(y_val, pred, zero_division=0))
    st.write("召回率:", recall_score(y_val, pred, zero_division=0))
    st.write("F1:", f1_score(y_val, pred, zero_division=0))

    return model

model = train(train, val)

# =========================
# 回测（可交易）
# =========================
def backtest(df, probs, th):
    df = df.copy()
    df['prob'] = probs
    df['signal'] = (df['prob'] >= th) & (df['trend_align'] == 1)

    equity = [0]
    trades = []
    position = 0
    entry = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平仓
        if position == 1 and (row['close'] < row['ema20'] or not row['signal']):
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开仓
        if prev['signal'] and position == 0:
            entry = price
            position = 1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "total": sum(trades),
        "equity": equity
    }

# =========================
# 验证与阈值
# =========================
val_probs = model.predict_proba(val[feat])[:,1]

# 自适应阈值：85百分位
th = np.quantile(val_probs, 0.85)

res = backtest(val, val_probs, th)

st.header("回测")
st.metric("交易", res['trades'])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("盈利", f"{res['total']:.2f}")

if res['trades'] > 0:
    st.line_chart(pd.Series(res['equity']))

st.success("完成")
