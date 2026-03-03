# -*- coding: utf-8 -*-
"""
5分钟合约多空策略（稳定版）
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")
st.title("🚀 5分钟合约多空策略")

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
        st.stop()

# =========================
# 特征工程
# =========================
def features(df):
    df = df.copy()

    # 多周期趋势
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['trend_up'] = (df['ema20'] > df['ema50'])

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)

    # ADX（趋势强度）
    def adx(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling(period).mean()

    df['adx'] = adx(df)

    # 标签：未来5根是否上涨
    future = df['close'].pct_change(5).shift(-5)
    df['target'] = (future > 0).astype(int)

    df.dropna(inplace=True)
    return df

with st.spinner("特征"):
    df = features(df)

st.success("特征完成")

# =========================
# 分割
# =========================
n = len(df)
train = df.iloc[:int(n*0.6)]
test = df.iloc[int(n*0.6):]

st.info(f"训练 {len(train)} | 测试 {len(test)}")

feat_cols = ['ema20','ema50','rsi','adx','trend_up']

# =========================
# 模型
# =========================
def train_model(train):
    X = train[feat_cols]
    y = train['target']

    if y.nunique() < 2:
        st.error("标签单一")
        st.stop()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X, y)
    pred = model.predict(X)
    st.write("训练准确率:", accuracy_score(y, pred))
    return model

with st.spinner("训练"):
    model = train_model(train)

# =========================
# 回测（多空）
# =========================
def backtest(df, probs, th):
    df = df.copy()
    df['prob'] = pd.Series(probs, index=df.index).fillna(0.5)

    # 多空信号
    df['long'] = (df['prob'] >= th) & (df['trend_up']) & (df['adx'] > 18)
    df['short'] = (df['prob'] < (1-th)) & (~df['trend_up']) & (df['adx'] > 18)

    equity = [0]
    position = 0  # 1多 -1空
    entry = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['open']

        # 平仓
        if position == 1 and (row['close'] < row['ema20']):
            pnl = price - entry
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        if position == -1 and (row['close'] > row['ema20']):
            pnl = entry - price
            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            position = 0

        # 开多
        if prev['long'] and position == 0:
            entry = price
            position = 1

        # 开空
        if prev['short'] and position == 0:
            entry = price
            position = -1

        equity.append(equity[-1])

    return {
        "trades": len(trades),
        "win_rate": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "total": sum(trades),
        "equity": equity
    }

# =========================
# 执行
# =========================
probs = model.predict_proba(test[feat_cols])[:,1]

# 保守阈值
th = np.quantile(probs, 0.8)

res = backtest(test, probs, th)

st.header("回测")
st.metric("交易", res['trades'])
st.metric("胜率", f"{res['win_rate']*100:.2f}%")
st.metric("盈利", f"{res['total']:.2f}")

if res['trades'] > 0:
    st.line_chart(pd.Series(res['equity']))

st.success("完成")
