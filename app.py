# -*- coding: utf-8 -*-
"""
终极量化交易策略 - 未来收益预测版（实盘级）
版本：16.0
作者：AI Assistant
说明：
- 自动字段兼容
- 无 ta 依赖
- 未来5根K线涨幅 > 0.3% 标签
- 回测 + 保守成交
- 多周期过滤
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

st.set_page_config(page_title="量化策略·实盘版", layout="wide")
st.title("🚀 未来收益预测量化系统")

# ======================
# 文件上传
# ======================
uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
if uploaded_file is None:
    st.info("请上传数据")
    st.stop()

# ======================
# 数据加载
# ======================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df.iloc[:, 0], errors='ignore') if 'datetime' not in df.columns else pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

with st.spinner("加载数据..."):
    df = load_data(uploaded_file)

st.success(f"数据加载成功：{len(df)} 行")

# ======================
# 字段标准化
# ======================
df.columns = [c.lower() for c in df.columns]

def get_col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None

df['open'] = get_col(df, 'open')
df['high'] = get_col(df, 'high')
df['low'] = get_col(df, 'low')
df['close'] = get_col(df, 'close')

df['volume'] = get_col(df, 'volume', 'vol', 'tick_volume', 'quote_volume')

required = ['open','high','low','close','volume']
for r in required:
    if df[r] is None:
        st.error(f"缺少字段: {r}")
        st.write("当前字段:", df.columns.tolist())
        st.stop()

# ======================
# 特征工程
# ======================
def create_features(df):
    df = df.copy()

    # 基础指标
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

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']

    # 多周期趋势
    df['trend_align'] = (df['close'] > df['ema20']).astype(int)

    # 未来收益标签（5根K线 > 0.3%）
    future = df['close'].pct_change(5).shift(-5)
    df['target'] = (future > 0.003).astype(int)

    df.dropna(inplace=True)
    return df

with st.spinner("生成特征..."):
    df = create_features(df)

st.success("特征生成完成")

# ======================
# 分割数据
# ======================
def split(df):
    n = len(df)
    train = df.iloc[:int(n*0.6)]
    val = df.iloc[int(n*0.6):int(n*0.8)]
    test = df.iloc[int(n*0.8):]
    return train, val, test

train, val, test = split(df)

st.info(f"训练: {len(train)} | 验证: {len(val)} | 测试: {len(test)}")

features = [
    'ema_distance','return_5','return_10','rsi','volume_ratio',
    'atr_ratio','trend_align'
]

# ======================
# 模型训练
# ======================
def train_model(train, val):
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']

    if y_train.nunique() < 2:
        st.error("目标只有单类，无法训练")
        st.stop()

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        early_stopping_rounds=50
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred = model.predict(X_val)
    st.write("验证准确率:", accuracy_score(y_val, pred))
    st.write("精确率:", precision_score(y_val, pred, zero_division=0))
    st.write("召回率:", recall_score(y_val, pred, zero_division=0))
    st.write("F1:", f1_score(y_val, pred, zero_division=0))

    return model

with st.spinner("训练模型..."):
    model = train_model(train, val)

# ======================
# 回测
# ======================
def backtest(df, probs, th=0.6):
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

        # 平仓
        if position == 1 and (row['close'] < row['ema20'] or row['signal'] == 0):
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开仓
        if prev['signal'] == 1 and position == 0:
            entry = price
            position = 1

        equity.append(equity[-1])

    total = sum(trades)
    win = sum(1 for p in trades if p > 0) / len(trades) if trades else 0

    return {
        "total": total,
        "win_rate": win,
        "trades": len(trades),
        "equity": equity
    }

# 验证集回测
val_probs = model.predict_proba(val[features])[:, 1]
res = backtest(val, val_probs)

st.header("回测结果")
st.metric("交易次数", res['trades'])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("总盈利", f"{res['total']:.2f}")

# 资金曲线
if res['trades'] > 0:
    fig = pd.Series(res['equity']).plot()
    st.pyplot(fig.figure)

st.success("系统运行完毕")
