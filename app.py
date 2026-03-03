# -*- coding: utf-8 -*-
"""
未来收益预测量化系统（实战可跑版）
作者：AI Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

st.set_page_config(page_title="量化策略", layout="wide")
st.title("🚀 未来收益预测量化系统")

# ===================== 上传 =====================
uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
if uploaded_file is None:
    st.info("请上传 CSV")
    st.stop()

# ===================== 加载 =====================
@st.cache_data
def load(file):
    df = pd.read_csv(file)

    # 字段替换（防缺失）
    if 'volume' not in df.columns and 'vol' in df.columns:
        df['volume'] = df['vol']
    if 'volume' not in df.columns:
        df['volume'] = 1.0

    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df[['open','high','low','close','volume']]

df = load(uploaded_file)
st.success(f"数据：{len(df)} 行")

# ===================== 特征 =====================
@st.cache_data
def features(df):
    df = df.copy()

    # 基础指标
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)

    # RSI（简化）
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - 100 / (1 + rs)

    # EMA距离
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_distance'] = (df['close'] - df['ema20']) / df['close']

    # Volume比
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1e-9)

    # 趋势一致
    df['trend_align'] = (df['close'] > df['ema20']).astype(int)

    # 标签：未来5根是否上涨 > 0
    future = df['close'].pct_change(5).shift(-5)
    df['target'] = (future > 0).astype(int)

    df.dropna(inplace=True)
    return df

df = features(df)
st.success("特征完成")

# ===================== 分割 =====================
def split(df):
    n = len(df)
    train = df.iloc[:int(n*0.6)].copy()
    val   = df.iloc[int(n*0.6):int(n*0.8)].copy()
    test  = df.iloc[int(n*0.8):].copy()
    return train, val, test

train_df, val_df, test_df = split(df)
st.info(f"训练 {len(train_df)} | 验证 {len(val_df)} | 测试 {len(test_df)}")

feat = ['trend_align','rsi','volume_ratio','ema_distance','return_5','return_10']

# ===================== 训练 =====================
def train_model(train_df, val_df):
    X_train = train_df[feat]
    y_train = train_df['target']

    X_val = val_df[feat]
    y_val = val_df['target']

    if y_train.nunique() < 2:
        st.error("标签只有一类，无法训练")
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

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    pred = model.predict(X_val)
    st.write("验证准确率:", accuracy_score(y_val, pred))
    st.write("精确率:", precision_score(y_val, pred, zero_division=0))
    st.write("召回率:", recall_score(y_val, pred, zero_division=0))
    st.write("F1:", f1_score(y_val, pred, zero_division=0))

    return model

model = train_model(train_df, val_df)

# ===================== 回测 =====================
def backtest(df, model):
    probs = model.predict_proba(df[feat])[:,1]
    df = df.copy()
    df['prob'] = probs

    # 信号
    df['signal'] = 0
    df.loc[df['prob'] >= 0.55, 'signal'] = 1
    df.loc[df['prob'] <= 0.45, 'signal'] = -1

    equity = [0]
    position = 0
    entry = 0
    trades = []
    wins = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        open_price = row['open']
        close = row['close']

        # 出场
        if position != 0:
            pnl = (close - entry) * position
            equity.append(equity[-1] + pnl)
            trades.append(pnl)
            if pnl > 0: wins += 1
            position = 0

        # 入场
        if row['signal'] == 1:
            position = 1
            entry = open_price
        elif row['signal'] == -1:
            position = -1
            entry = open_price

    total = sum(trades)
    win_rate = wins / len(trades) if trades else 0

    return {
        "交易": len(trades),
        "胜率": win_rate,
        "盈利": total
    }

st.write("回测结果")
st.json(backtest(test_df, model))

st.success("运行完毕")
