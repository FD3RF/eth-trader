# -*- coding: utf-8 -*-
"""
终极量化交易策略 - 多空阈值分离优化版 (改进版)
说明：优化目标、特征、模型训练，提升策略稳健性。
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

st.set_page_config(page_title="量化策略·改进版", layout="wide")
st.title("🚀 改进版量化策略 - 多空阈值优化 (稳健版)")

# ====================== 侧边栏参数 ======================
st.sidebar.header("⚙️ 交易成本")
fee_rate = st.sidebar.number_input("双向手续费率", value=0.0004, format="%.4f")
slippage = st.sidebar.number_input("滑点 (USDT)", value=0.5, step=0.1)

st.sidebar.header("🎛️ 优化参数范围")
long_th_min = st.sidebar.slider("做多阈值最小值", 0.5, 0.7, 0.5, 0.05)
long_th_max = st.sidebar.slider("做多阈值最大值", 0.55, 0.8, 0.75, 0.05)
short_th_min = st.sidebar.slider("做空阈值最小值", 0.2, 0.4, 0.2, 0.05)
short_th_max = st.sidebar.slider("做空阈值最大值", 0.25, 0.5, 0.45, 0.05)

# ====================== 文件上传 ======================
uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])
if uploaded_file is None:
    st.info("请先上传文件")
    st.stop()

# ====================== 数据加载 ======================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={'vol':'volume'}, inplace=True)
    return df[['open','high','low','close','volume']]

with st.spinner("加载数据中..."):
    df = load_data(uploaded_file)
st.success(f"✅ 数据加载成功！总K线数: {len(df)}")

# ====================== 特征工程（增强版） ======================
@st.cache_data
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
    
    # EMA
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_cross'] = (df['ema10'] - df['ema20']) / df['close']
    df['trend'] = np.where(df['close'] > df['ema20'], 1, -1)  # 趋势过滤器
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # 布林带
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = 2 * df['bb_std'] / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_mid']) / (2 * df['bb_std'])
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['return_1']) * df['volume']).fillna(0).cumsum()
    df['obv_ma5'] = df['obv'].rolling(5).mean()
    df['obv_ratio'] = df['obv'] / df['obv_ma5']
    
    # 时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # 滞后特征（减少数量，只保留重要的）
    for lag in [1,2]:
        df[f'close_lag{lag}'] = df['close'].shift(lag)
        df[f'volume_lag{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
    
    # 滚动统计（减少窗口）
    for window in [5,20]:
        df[f'close_max_{window}'] = df['close'].rolling(window).max()
        df[f'close_min_{window}'] = df['close'].rolling(window).min()
        df[f'close_std_{window}'] = df['close'].rolling(window).std()
    
    # 目标变量：预测未来1根K线的涨跌幅（回归）
    df['target'] = df['close'].pct_change(1).shift(-1)  # 下一根K线收益率
    
    df.dropna(inplace=True)
    return df

with st.spinner("生成特征中..."):
    df_feat = create_features(df)
st.success("✅ 特征生成完成！")

# ====================== 数据分割（时间顺序） ======================
def split_data(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test

train, val, test = split_data(df_feat)
st.info(f"📊 训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")

# ====================== 特征列 ======================
features = [col for col in df_feat.columns if col not in ['open','high','low','close','volume','target','trend']]
# 趋势过滤器单独使用

# ====================== 训练XGBoost回归模型（带早停） ======================
def train_xgboost(train, val, features):
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 输出验证集RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    st.write(f"✅ 验证集 RMSE: {rmse:.6f}")
    
    return model

with st.spinner("训练XGBoost回归模型中..."):
    model = train_xgboost(train, val, features)

# ====================== 回测函数（增加趋势过滤） ======================
def backtest(df, model, features, th_long, th_short, fee_rate, slippage, atr_mult, rr):
    df = df.copy()
    df['pred'] = model.predict(df[features])  # 预测下一根K线收益率
    
    # 生成信号（结合趋势过滤）
    df['signal'] = 0
    # 做多条件：预测收益率 > th_long 且 价格在EMA20之上
    df.loc[(df['pred'] >= th_long) & (df['close'] > df['ema20']), 'signal'] = 1
    # 做空条件：预测收益率 < th_short 且 价格在EMA20之下
    df.loc[(df['pred'] <= th_short) & (df['close'] < df['ema20']), 'signal'] = -1
    
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_time = None
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
        current_time = row.name
        
        # 现有持仓管理
        if position == 1:
            stop = entry_price - atr_mult * entry_atr
            take = entry_price + rr * atr_mult * entry_atr
            exit_price = None
            if low <= stop:
                exit_price = stop
            elif high >= take:
                exit_price = take
            # 强制平仓：持仓超过10根K线
            elif entry_time is not None and (current_time - entry_time).seconds > 3000:  # 5分钟*10=3000秒
                exit_price = open_price
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
            elif entry_time is not None and (current_time - entry_time).seconds > 3000:
                exit_price = open_price
            if exit_price is not None:
                pnl = (entry_price - exit_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
                position = 0
        
        # 新信号处理
        if prev['signal'] == 1 and position != 1:
            if position == -1:
                pnl = (entry_price - open_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
            position = 1
            entry_price = open_price
            entry_atr = atr
            entry_time = current_time
        
        elif prev['signal'] == -1 and position != -1:
            if position == 1:
                pnl = (open_price - entry_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
            position = -1
            entry_price = open_price
            entry_atr = atr
            entry_time = current_time
    
    # 最后平仓
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
    
    # 计算Calmar比率
    if max_dd != 0:
        calmar = total_pnl / max_dd
    else:
        calmar = 0
    
    # 夏普比率（假设年化252*24*60/5分钟）
    sharpe = np.mean(trades) / np.std(trades) * np.sqrt(365*24*60/5) if len(trades)>1 and np.std(trades)!=0 else 0
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'trades': len(trades),
        'equity': equity
    }

# ====================== 多空阈值联合优化（使用Calmar比率） ======================
@st.cache_data
def optimize_params(val_df, _model, features, fee_rate, slippage, long_range, short_range):
    best_calmar = -999
    best_params = None
    best_result = None
    
    # 网格步长加大，减少过拟合
    long_ths = np.arange(long_range[0], long_range[1] + 0.01, 0.1)
    short_ths = np.arange(short_range[0], short_range[1] + 0.01, 0.1)
    atr_mults = [1.5, 2.0]  # 减少选项
    rrs = [2.0, 2.5]
    
    total = len(long_ths) * len(short_ths) * len(atr_mults) * len(rrs)
    progress_bar = st.progress(0, text="优化参数中...")
    count = 0
    
    for th_l in long_ths:
        for th_s in short_ths:
            for atr in atr_mults:
                for rr in rrs:
                    res = backtest(val_df, _model, features, th_l, th_s, fee_rate, slippage, atr, rr)
                    # 要求至少20笔交易，且Calmar比率最优
                    if res['trades'] >= 20 and res['calmar'] > best_calmar:
                        best_calmar = res['calmar']
                        best_params = (th_l, th_s, atr, rr)
                        best_result = res
                    count += 1
                    progress_bar.progress(count / total)
    
    progress_bar.empty()
    return best_params, best_result

long_range = (long_th_min, long_th_max)
short_range = (short_th_min, short_th_max)

st.write("🔄 正在验证集上优化多空阈值及交易参数...")
best_params, val_res = optimize_params(val, model, features, fee_rate, slippage, long_range, short_range)

if best_params is None:
    st.error("❌ 未找到符合条件的参数组合，请调整参数范围或检查数据。")
    st.stop()

th_long, th_short, atr_mult, rr = best_params

st.success("✅ 验证集优化完成！")
col1, col2, col3, col4 = st.columns(4)
col1.metric("做多阈值", f"{th_long:.2f}")
col2.metric("做空阈值", f"{th_short:.2f}")
col3.metric("ATR倍数", f"{atr_mult}")
col4.metric("盈亏比", f"{rr}")

st.write("📈 验证集最优结果：")
st.json(val_res)

# ====================== 测试集验证 ======================
st.write("🔄 在测试集上验证...")
test_res = backtest(test, model, features, th_long, th_short, fee_rate, slippage, atr_mult, rr)

st.header("🎯 测试集最终结果")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("交易次数", test_res['trades'])
col2.metric("胜率", f"{test_res['win_rate']*100:.1f}%")
col3.metric("总盈利", f"{test_res['total_pnl']:.2f} USDT")
col4.metric("最大回撤", f"{test_res['max_dd']:.2f} USDT")
col5.metric("Calmar比率", f"{test_res['calmar']:.2f}")

# ====================== 资金曲线图 ======================
if test_res['trades'] > 0:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test.index[:len(test_res['equity'])],
        y=test_res['equity'],
        mode='lines',
        name='资金曲线',
        line=dict(color='#00ff88', width=2)
    ))
    fig.update_layout(
        title="测试集资金曲线",
        xaxis_title="时间",
        yaxis_title="累积盈亏 (USDT)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================== 结论 ======================
if test_res['calmar'] > 1.0:
    st.success("✨ 测试结果优秀！可以考虑进一步优化或实盘模拟。")
elif test_res['calmar'] > 0.5:
    st.info("📊 测试结果为正收益，但稳定性一般。可尝试调整特征或参数范围。")
else:
    st.warning("⚠️ 测试结果为负，策略可能无效。请检查特征或逻辑。")
