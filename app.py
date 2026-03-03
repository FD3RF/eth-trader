# -*- coding: utf-8 -*-
"""
ETH 5分钟未来收益回归系统
实盘一致性版本 v20.0
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


# =============================
# 参数
# =============================
LOOKAHEAD = 5
TOP_Q = 0.95
BOT_Q = 0.05
N_SPLITS = 5
FEE = 0.0006


# =============================
# 特征工程
# =============================
def create_features(df):

    df['ret1'] = df['close'].pct_change()
    df['ret3'] = df['close'].pct_change(3)
    df['ret5'] = df['close'].pct_change(5)

    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['oc_range'] = (df['close'] - df['open']) / df['open']

    df['vol_mean'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_mean']

    df['future_return'] = (
        df['close'].shift(-LOOKAHEAD) / df['close'] - 1
    )

    df.dropna(inplace=True)

    features = [
        'ret1','ret3','ret5',
        'hl_range','oc_range',
        'vol_ratio'
    ]

    return df, features


# =============================
# 回测（固定持仓周期）
# =============================
def backtest(df, preds):

    equity = np.zeros(len(df))
    position = 0
    entry_index = None
    entry_price = 0
    trades = []

    long_th = np.quantile(preds, TOP_Q)
    short_th = np.quantile(preds, BOT_Q)

    for i in range(1, len(df)):

        equity[i] = equity[i-1]

        price = df['close'].iloc[i]

        # 平仓
        if position != 0 and i - entry_index >= LOOKAHEAD:

            pnl = (
                (price - entry_price) / entry_price
                * position
                - FEE
            )

            equity[i] += pnl
            trades.append(pnl)
            position = 0

        # 开仓
        if position == 0:

            if preds[i] >= long_th:
                position = 1
                entry_price = price
                entry_index = i

            elif preds[i] <= short_th:
                position = -1
                entry_price = price
                entry_index = i

    # 统计
    if len(trades) == 0:
        return 0, 0, 0, equity

    total_return = equity[-1]
    max_dd = np.max(np.maximum.accumulate(equity) - equity)
    calmar = total_return / (max_dd + 1e-9)
    winrate = np.mean(np.array(trades) > 0)

    return calmar, winrate, len(trades), equity


# =============================
# Walk Forward
# =============================
def walk_forward(df, features):

    fold_size = len(df) // (N_SPLITS + 1)

    calmars = []

    for fold in range(N_SPLITS):

        train_end = fold_size * (fold + 1)
        val_end = fold_size * (fold + 2)

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]

        X_train = train[features]
        y_train = train['future_return']

        X_val = val[features]
        y_val = val['future_return']

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        calmar, winrate, trades, _ = backtest(val, preds)

        if trades > 10:
            calmars.append(calmar)

    return np.mean(calmars)


# =============================
# 主流程
# =============================
def run_system(df):

    df, features = create_features(df)

    wf_score = walk_forward(df, features)

    # 最终训练
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['future_return']

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)

    preds = model.predict(X)

    calmar, winrate, trades, equity = backtest(df, preds)

    return {
        "walk_forward_score": wf_score,
        "calmar": calmar,
        "winrate": winrate,
        "trades": trades,
        "equity": equity
    }
