# -*- coding: utf-8 -*-
"""
5分钟合约多空策略 - 稳定可运行版
作者：AI Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="5分钟合约策略", layout="wide")
st.title("🚀 5分钟合约多空策略")

# ============= 上传 =============
uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
if uploaded_file is None:
    st.info("上传数据")
    st.stop()

# ============= 加载 =============
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    # 时间处理
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

df = load_data(uploaded_file)
st.success(f"数据: {len(df)} 行")

# ============= 字段映射（兼容不同列名）============
def col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None

df['open'] = col(df,'open')
df['high'] = col(df,'high')
df['low'] = col(df,'low')
df['close'] = col(df,'close')
df['volume'] = col(df,'volume','vol','tick_volume','quote_volume')

for c in ['open','high','low','close']:
    if df[c] is None:
        st.error(f"缺少字段: {c}")
        st.write(df.columns.tolist())
        st.stop()

# volume 可选
if df['volume'] is None:
    df['volume'] = 1

# ============= 特征工程（轻量）============
def make_features(df):
    df = df.copy()

    # 趋势
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema_distance'] = (df['close'] - df['ema20']) / df['close']

    # 动量
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)

    # RSI简化
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)

    # 量能
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']

    df['trend_align'] = (df['close'] > df['ema20']).astype(int)

    # 标签：未来是否上涨
    future = df['close'].pct_change(5).shift(-5)
    df['target'] = (future > 0).astype(int)

    df.dropna(inplace=True)
    return df

df = make_features(df)
st.success("特征完成")

# ============= 分割 =============
n = len(df)
train = df.iloc[:int(n*0.6)]
val   = df.iloc[int(n*0.6):int(n*0.8)]
test  = df.iloc[int(n*0.8):]

st.info(f"训练 {len(train)} | 验证 {len(val)} | 测试 {len(test)}")

features = [
    'ema_distance','return_5','return_10','rsi',
    'volume_ratio','atr_ratio','trend_align'
]

# ============= 模型 =============
def train_model(train, val):
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']

    if y_train.nunique() < 2:
        st.error("标签只有一类")
        st.stop()

    model = xgb.XGBClassifier(
        n_estimators=300,
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

model = train_model(train, val)

# ============= 回测（多空简化）============
def backtest(df, probs, th):
    df = df.copy()
    df['prob'] = probs

    equity = [0]
    position = 0
    entry = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['open']
        signal = 1 if row['prob'] >= th else 0

        # 平仓条件：趋势反转
        if position == 1 and row['close'] < row['ema20']:
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开仓
        if signal == 1 and position == 0:
            entry = price
            position = 1

        equity.append(equity[-1])

    return {
        "交易": len(trades),
        "胜率": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "盈利": sum(trades),
        "equity": equity
    }

# ============= 阈值自适应 =============
val_probs = model.predict_proba(val[features])[:,1]
th = np.quantile(val_probs, 0.7)

res = backtest(test, model.predict_proba(test[features])[:,1], th)

st.header("回测")
st.metric("交易", res["交易"])
st.metric("胜率", f"{res['胜率']*100:.2f}%")
st.metric("盈利", f"{res['盈利']:.2f}")

if res["交易"] > 0:
    st.line_chart(pd.Series(res["equity"]))

st.success("运行完毕")
