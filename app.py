# -*- coding: utf-8 -*-
"""
终极量化交易策略 - 多空阈值分离优化版 (Streamlit Cloud 适配)
作者：AI Assistant
版本：4.1
说明：上传CSV文件，自动优化多空阈值，显示测试集结果。
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ====================== 页面设置 ======================
st.set_page_config(page_title="终极量化策略·多空优化", layout="wide")
st.title("🚀 终极量化交易策略 - 多空阈值分离优化版")
st.markdown("上传您的 **ETHUSDT_5m_last_90days.csv** 文件，系统将自动优化多空阈值并给出测试集结果。")

# ====================== 侧边栏参数 ======================
st.sidebar.header("⚙️ 交易成本设置")
fee_rate = st.sidebar.number_input("双向手续费率 (例如0.0004)", value=0.0004, format="%.4f")
slippage = st.sidebar.number_input("滑点 (USDT)", value=0.5, step=0.1)

# ====================== 文件上传 ======================
uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])
if uploaded_file is None:
    st.info("请先上传文件")
    st.stop()

# ====================== 数据加载 ======================
with st.spinner("加载数据中..."):
    df = pd.read_csv(uploaded_file)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={'vol':'volume'}, inplace=True)
    df = df[['open','high','low','close','volume']]

st.success(f"✅ 数据加载成功！总K线数: {len(df)}")

# ====================== 特征工程 ======================
def create_features(df):
    df = df.copy()
    # 基础价格特征
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # 成交量特征
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # 技术指标
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_cross'] = (df['ema10'] - df['ema20']) / df['close']
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)
    
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = 2 * df['bb_std'] / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_mid']) / (2 * df['bb_std'])
    
    # 滞后特征
    for lag in [1,2,3]:
        df[f'close_lag{lag}'] = df['close'].shift(lag)
        df[f'volume_lag{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
        df[f'atr_lag{lag}'] = df['atr'].shift(lag)
    
    # 滚动统计
    for window in [5,10,20]:
        df[f'close_max_{window}'] = df['close'].rolling(window).max()
        df[f'close_min_{window}'] = df['close'].rolling(window).min()
        df[f'close_std_{window}'] = df['close'].rolling(window).std()
    
    # 目标变量：预测未来5根K线是否上涨
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    return df

with st.spinner("生成特征中..."):
    df_feat = create_features(df)
st.success("✅ 特征生成完成！")

# ====================== 数据分割 ======================
def split_data(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test

train, val, test = split_data(df_feat)
st.info(f"训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")

# ====================== 特征列 ======================
features = [col for col in df_feat.columns if col not in ['open','high','low','close','volume','target']]

# ====================== 训练XGBoost ======================
import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_xgboost(train, val, features):
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )
    
    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    st.write(f"验证集准确率: {acc:.3f}")
    return model

with st.spinner("训练XGBoost模型中..."):
    model = train_xgboost(train, val, features)

# ====================== 回测函数 ======================
def backtest(df, model, features, th_long, th_short, fee_rate, slippage, atr_mult, rr):
    df = df.copy()
    df['prob'] = model.predict_proba(df[features])[:, 1]
    df['signal'] = 0
    df.loc[df['prob'] >= th_long, 'signal'] = 1
    df.loc[df['prob'] <= th_short, 'signal'] = -1
    
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    equity = [0.0]
    trades = []
    wins = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        open_price = row['open']
        high = row['high']
        low = row['low']
        atr = row['atr']
        
        if position == 1:
            stop = entry_price - atr_mult * entry_atr
            take = entry_price + rr * atr_mult * entry_atr
            exit_price = None
            if low <= stop:
                exit_price = stop
            elif high >= take:
                exit_price = take
            if exit_price is not None:
                pnl = (exit_price - entry_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
                position = 0
        
        elif position == -1:
            stop = entry_price + atr_mult * entry_atr
            take = entry_price - rr * atr_mult * entry_atr
            exit_price = None
            if high >= stop:
                exit_price = stop
            elif low <= take:
                exit_price = take
            if exit_price is not None:
                pnl = (entry_price - exit_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
                position = 0
        
        if prev['signal'] == 1 and position != 1:
            if position == -1:
                pnl = (entry_price - open_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
            position = 1
            entry_price = open_price
            entry_atr = atr
        
        elif prev['signal'] == -1 and position != -1:
            if position == 1:
                pnl = (open_price - entry_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
            position = -1
            entry_price = open_price
            entry_atr = atr
    
    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            pnl = (last_price - entry_price) * (1 - fee_rate * 2) - slippage
        else:
            pnl = (entry_price - last_price) * (1 - fee_rate * 2) - slippage
        equity.append(equity[-1] + pnl)
        trades.append(pnl)
        if pnl > 0: wins += 1
    
    total_pnl = sum(trades)
    win_rate = wins / len(trades) if trades else 0
    max_equity = np.maximum.accumulate(equity)
    drawdown = max_equity - equity
    max_dd = np.max(drawdown)
    sharpe = np.mean(trades) / np.std(trades) * np.sqrt(365*24*60/5) if len(trades)>1 and np.std(trades)!=0 else 0
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': len(trades)
    }

# ====================== 多空阈值联合优化 ======================
@st.cache_data
def optimize_params(val_df, model, features, fee_rate, slippage):
    best_sharpe = -999
    best_params = None
    best_result = None
    
    long_ths = np.arange(0.5, 0.8, 0.05)
    short_ths = np.arange(0.2, 0.5, 0.05)
    atr_mults = [1.0, 1.5, 2.0]
    rrs = [1.5, 2.0, 2.5, 3.0]
    
    total = len(long_ths) * len(short_ths) * len(atr_mults) * len(rrs)
    progress_bar = st.progress(0, text="优化参数中...")
    count = 0
    
    for th_l in long_ths:
        for th_s in short_ths:
            for atr in atr_mults:
                for rr in rrs:
                    res = backtest(val_df, model, features, th_l, th_s, fee_rate, slippage, atr, rr)
                    if res['sharpe'] > best_sharpe and res['trades'] >= 20:
                        best_sharpe = res['sharpe']
                        best_params = (th_l, th_s, atr, rr)
                        best_result = res
                    count += 1
                    progress_bar.progress(count / total)
    
    progress_bar.empty()
    return best_params, best_result

st.write("正在验证集上优化多空阈值及交易参数...")
best_params, val_res = optimize_params(val, model, features, fee_rate, slippage)

if best_params is None:
    st.error("未找到符合条件的参数组合，请调整参数范围或检查数据。")
    st.stop()

th_long, th_short, atr_mult, rr = best_params

st.success("✅ 验证集优化完成！")
col1, col2, col3, col4 = st.columns(4)
col1.metric("做多阈值", f"{th_long:.2f}")
col2.metric("做空阈值", f"{th_short:.2f}")
col3.metric("ATR倍数", f"{atr_mult}")
col4.metric("盈亏比", f"{rr}")

st.write("验证集最优结果：")
st.json(val_res)

# ====================== 测试集验证 ======================
st.write("在测试集上验证...")
test_res = backtest(test, model, features, th_long, th_short, fee_rate, slippage, atr_mult, rr)

st.header("📊 测试集最终结果")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("交易次数", test_res['trades'])
col2.metric("胜率", f"{test_res['win_rate']*100:.1f}%")
col3.metric("总盈利", f"{test_res['total_pnl']:.2f} USDT")
col4.metric("最大回撤", f"{test_res['max_dd']:.2f} USDT")
col5.metric("夏普比率", f"{test_res['sharpe']:.2f}")

if test_res['sharpe'] > 1.0:
    st.success("✨ 测试结果优秀！可以考虑进一步优化或实盘模拟。")
elif test_res['sharpe'] > 0:
    st.info("📊 测试结果为正收益，但稳定性一般。可尝试调整特征或参数范围。")
else:
    st.warning("⚠️ 测试结果为负，策略可能无效。请检查特征或逻辑。")
