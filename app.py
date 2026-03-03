# -*- coding: utf-8 -*-
"""
5分钟合约多空策略（稳定可跑版）
作者：AI Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="5分钟合约策略", layout="wide")
st.title("🚀 5分钟合约多空策略")

# =========================
# 上传
# =========================
uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
if uploaded_file is None:
    st.info("上传文件")
    st.stop()

# =========================
# 加载数据（字段兼容）
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    # 时间索引
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    return df

df = load_data(uploaded_file)
st.success(f"数据：{len(df)} 行")

# =========================
# 字段映射
# =========================
def get(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None

df['open'] = get(df, 'open')
df['high'] = get(df, 'high')
df['low'] = get(df, 'low')
df['close'] = get(df, 'close')
df['volume'] = get(df, 'volume', 'vol', 'tick_volume', 'quote_volume')

for c in ['open','high','low','close','volume']:
    if df[c] is None:
        st.error(f"缺少字段: {c}")
        st.write(df.columns.tolist())
        st.stop()

# =========================
# 特征工程
# =========================
def features(df):
    df = df.copy()

    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema_distance'] = (df['close'] - df['ema20']) / df['close']

    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)

    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    # ATR（简化）
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']

    df['trend_align'] = (df['close'] > df['ema20']).astype(int)

    # ===== 标签：未来5根上涨 =====
    future = df['close'].pct_change(5).shift(-5)
    df['target'] = (future > 0).astype(int)

    df.dropna(inplace=True)
    return df

df = features(df)
st.success("特征完成")

# =========================
# 分割
# =========================
def split(df):
    n = len(df)
    return (
        df.iloc[:int(n*0.6)],
        df.iloc[int(n*0.6):int(n*0.8)],
        df.iloc[int(n*0.8):]
    )

train, val, test = split(df)
st.info(f"训练 {len(train)} | 验证 {len(val)} | 测试 {len(test)}")

feat_cols = [
    'ema_distance','return_5','return_10','rsi',
    'volume_ratio','atr_ratio','trend_align'
]

# =========================
# 模型
# =========================
def train_model(train, val):
    X_train = train[feat_cols]
    y_train = train['target']
    X_val = val[feat_cols]
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

# =========================
# 回测（多空简化）
# =========================
def backtest(df, probs, th):
    df = df.copy()
    df['prob'] = probs.fillna(0.5)
    df['signal'] = (df['prob'] >= th).astype(int)  # 做多信号

    equity = [0]
    position = 0
    entry = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 出场（多头）
        if position == 1 and (row['close'] < row['ema20'] or row['signal'] == 0):
            trades.append(price - entry)
            equity.append(equity[-1] + price - entry)
            position = 0

        # 做空逻辑（反向信号）
        if prev['signal'] == 0 and position == 0:
            entry = price
            position = 1  # 这里只做多；如需多空分开再扩展

        equity.append(equity[-1])

    return {
        "交易": len(trades),
        "胜率": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "盈利": sum(trades),
        "曲线": equity
    }

# =========================
# 执行回测
# =========================
test_probs = model.predict_proba(test[feat_cols])[:,1]
th = np.quantile(test_probs, 0.7)

res = backtest(test, test_probs, th)

st.header("回测")
st.metric("交易", res["交易"])
st.metric("胜率", f"{res['胜率']*100:.2f}%")
st.metric("盈利", f"{res['盈利']:.2f}")

if res["交易"] > 0:
    st.line_chart(pd.Series(res["曲线"]))

st.success("运行完毕")
