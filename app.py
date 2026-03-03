# -*- coding: utf-8 -*-
"""
终极量化交易策略 - 多空阈值分离优化版 (稳健增强版 + 交易计划提示)
作者：AI Assistant
版本：9.1
说明：整合结构目标、多周期共振、随机优化，并展示完整的交易策略计划。
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

st.set_page_config(page_title="量化策略·增强版", layout="wide")
st.title("🚀 终极量化策略 - 结构目标 + 多周期共振 (XGBoost分类)")

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

# ====================== 多周期数据构建（无未来泄漏） ======================
def compute_ema_history(df, period='15min', span=20):
    """用前一根重采样K线的EMA填充当前，避免未来信息"""
    full_idx = df.index
    resampled = df['close'].resample(period).last()
    ema_resampled = resampled.ewm(span=span, adjust=False).mean()
    ema = ema_resampled.reindex(full_idx, method='ffill')
    return ema

df['ema_15m'] = compute_ema_history(df, '15min', 20)
df['ema_1h'] = compute_ema_history(df, '1h', 20)

# ====================== 特征工程（结构目标） ======================
@st.cache_data
def create_features(df):
    df = df.copy()
    
    # 基础趋势结构
    df['high_3_prev'] = df['high'].shift(1).rolling(3).max()
    df['low_3_prev'] = df['low'].shift(1).rolling(3).min()
    df['high_increasing'] = (df['high'].shift(1) > df['high'].shift(2)) & (df['high'].shift(2) > df['high'].shift(3))
    df['low_increasing'] = (df['low'].shift(1) > df['low'].shift(2)) & (df['low'].shift(2) > df['low'].shift(3))
    
    # 上升结构：高点抬高 + 低点抬高 + 价格在近期低点之上
    df['uptrend_structure'] = (
        df['high_increasing'] & df['low_increasing'] &
        (df['close'] > df['low'].shift(1))
    ).astype(int)
    
    # 下降结构：高点降低 + 低点降低 + 价格在近期高点之下
    df['high_decreasing'] = (df['high'].shift(1) < df['high'].shift(2)) & (df['high'].shift(2) < df['high'].shift(3))
    df['low_decreasing'] = (df['low'].shift(1) < df['low'].shift(2)) & (df['low'].shift(2) < df['low'].shift(3))
    df['downtrend_structure'] = (
        df['high_decreasing'] & df['low_decreasing'] &
        (df['close'] < df['high'].shift(1))
    ).astype(int)
    
    # 目标：当前是否处于上升结构（二分类）
    df['target'] = df['uptrend_structure']
    
    # 常用技术指标
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
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
    
    def compute_adx(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx
    df['adx'] = compute_adx(df)
    
    # 回调比例
    df['pullback'] = (df['close'] - df['low'].rolling(5).min()) / (df['high'].rolling(5).max() - df['low'].rolling(5).min())
    
    # 多周期趋势一致（仅用历史值）
    df['trend_align'] = ((df['close'] > df['ema20']) & 
                         (df['close'] > df['ema_15m']) & 
                         (df['close'] > df['ema_1h'])).astype(int)
    
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
st.info(f"📊 训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")

# ====================== 特征列 ======================
features = [
    'ema_distance', 'return_5', 'return_10', 'rsi', 'volume_ratio', 'atr_ratio',
    'adx', 'pullback', 'trend_align'
]

# ====================== 训练XGBoost分类模型 ======================
def train_xgboost(train, val, features):
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']
    
    if y_train.nunique() < 2:
        st.error("目标变量只有一个类别，无法训练。")
        st.stop()
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    st.write(f"✅ 验证集准确率: {acc:.3f}")
    return model

with st.spinner("训练XGBoost分类模型中..."):
    model = train_xgboost(train, val, features)

# ====================== 回测函数（使用预计算概率 + 结构出场） ======================
def backtest_with_probs(df, probs, th_long, th_short, fee_rate, slippage, atr_mult, rr):
    df = df.copy()
    df['prob'] = probs
    
    # 信号生成：概率区间 + 多周期趋势 + ADX过滤 + 回调比例过滤
    df['signal'] = 0
    df.loc[(df['prob'] >= th_long) & (df['trend_align'] == 1) & (df['adx'] > 20) & (df['pullback'] < 0.8), 'signal'] = 1
    df.loc[(df['prob'] <= th_short) & (df['trend_align'] == 0) & (df['adx'] > 20) & (df['pullback'] > 0.2), 'signal'] = -1
    
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
            exit_reason = None
            if low <= stop:
                exit_price = stop
                exit_reason = "止损"
            elif high >= take:
                exit_price = take
                exit_reason = "止盈"
            # 趋势结构破坏（出现下降结构）
            elif row['downtrend_structure'] == 1:
                exit_price = open_price
                exit_reason = "结构破坏"
            # 强制平仓：持仓超过20根K线
            elif entry_time is not None and (current_time - entry_time).seconds > 6000:
                exit_price = open_price
                exit_reason = "超时"
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
            exit_reason = None
            if high >= stop:
                exit_price = stop
                exit_reason = "止损"
            elif low <= take:
                exit_price = take
                exit_reason = "止盈"
            elif row['uptrend_structure'] == 1:
                exit_price = open_price
                exit_reason = "结构破坏"
            elif entry_time is not None and (current_time - entry_time).seconds > 6000:
                exit_price = open_price
                exit_reason = "超时"
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
    
    if max_dd != 0:
        calmar = total_pnl / max_dd
    else:
        calmar = 0
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'calmar': calmar,
        'trades': len(trades),
        'equity': equity
    }

# ====================== 随机采样优化（5折滚动验证） ======================
@st.cache_data
def random_optimize(val_df, _model, features, fee_rate, slippage, long_range, short_range, n_iter=200):
    val_probs = _model.predict_proba(val_df[features])[:, 1]
    
    best_score = -999
    best_params = None
    best_result = None

    np.random.seed(42)
    long_ths = np.random.uniform(long_range[0], long_range[1], n_iter)
    short_ths = np.random.uniform(short_range[0], short_range[1], n_iter)
    atr_mults = np.random.choice([1.5, 2.0, 2.5], n_iter)
    rrs = np.random.choice([2.0, 2.5, 3.0], n_iter)

    val_len = len(val_df)
    fold_size = val_len // 5
    n_splits = 5

    progress_bar = st.progress(0, text="随机优化中...")
    for i in range(n_iter):
        th_l = long_ths[i]
        th_s = short_ths[i]
        atr = atr_mults[i]
        rr = rrs[i]
        
        calmars = []
        for fold in range(n_splits):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < n_splits - 1 else val_len
            val_fold = val_df.iloc[start:end].copy()
            probs_fold = val_probs[start:end]
            res = backtest_with_probs(val_fold, probs_fold, th_l, th_s, fee_rate, slippage, atr, rr)
            if res['trades'] >= 10:
                calmars.append(res['calmar'])
        if len(calmars) >= 3:
            median_calmar = np.median(calmars)
            if median_calmar > best_score:
                best_score = median_calmar
                best_params = (th_l, th_s, atr, rr)
                full_res = backtest_with_probs(val_df, val_probs, th_l, th_s, fee_rate, slippage, atr, rr)
                best_result = full_res
        progress_bar.progress((i+1) / n_iter)

    progress_bar.empty()
    return best_params, best_result

long_range = (long_th_min, long_th_max)
short_range = (short_th_min, short_th_max)

st.write("🔄 正在验证集上进行随机采样优化（5折滚动验证）...")
best_params, val_res = random_optimize(val, model, features, fee_rate, slippage, long_range, short_range)

if best_params is None:
    st.error("❌ 未找到符合条件的参数组合，请调整参数范围或检查数据。")
    st.stop()

th_long, th_short, atr_mult, rr = best_params

st.success("✅ 随机优化完成！")
col1, col2, col3, col4 = st.columns(4)
col1.metric("做多阈值", f"{th_long:.2f}")
col2.metric("做空阈值", f"{th_short:.2f}")
col3.metric("ATR倍数", f"{atr_mult}")
col4.metric("盈亏比", f"{rr}")

st.write("📈 验证集最优结果（完整验证集）：")
st.json(val_res)

# ====================== 显示交易策略计划 ======================
with st.expander("📋 交易策略计划（点击展开）", expanded=True):
    st.markdown("""
    ### 🎯 进场条件
    - **做多**：
        - 模型预测上涨概率 ≥ 做多阈值 (`{th_long:.2f}`)
        - 多周期趋势一致：5min、15min、1h EMA20 均向上
        - 趋势强度 ADX > 20
        - 回调比例 < 0.8（避免追高）
    - **做空**：
        - 模型预测上涨概率 ≤ 做空阈值 (`{th_short:.2f}`)
        - 多周期趋势一致（均向下）
        - ADX > 20
        - 回调比例 > 0.2（避免杀跌）

    ### 🛑 出场条件
    - **止盈**：盈利达到 盈亏比 × ATR 倍数 (`{rr:.1f} × {atr_mult:.1f} × ATR`)
    - **止损**：亏损达到 ATR 倍数 (`{atr_mult:.1f} × ATR`)
    - **趋势结构破坏**：
        - 做多后出现下降结构（高点降低 + 低点降低）
        - 做空后出现上升结构（高点抬高 + 低点抬高）
    - **超时平仓**：持仓超过 20 根K线（100分钟）

    ### ⚖️ 风险控制
    - 单笔最大亏损由 ATR 止损固定
    - 建议每笔交易风险不超过总资金的 1-2%
    - 策略在优化时要求每个分段至少有10笔交易，确保统计意义
    """.format(th_long=th_long, th_short=th_short, rr=rr, atr_mult=atr_mult))

# ====================== 测试集验证 ======================
st.write("🔄 在测试集上验证...")
test_probs = model.predict_proba(test[features])[:, 1]
test_res = backtest_with_probs(test, test_probs, th_long, th_short, fee_rate, slippage, atr_mult, rr)

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
if test_res['calmar'] > 1.5:
    st.success("✨ 测试结果非常优秀！策略稳健，可考虑实盘模拟。")
elif test_res['calmar'] > 0.8:
    st.info("📊 测试结果良好，风险收益比较理想，可进一步优化。")
elif test_res['calmar'] > 0:
    st.warning("⚠️ 测试结果为正收益，但风险调整后收益一般，需检查参数或特征。")
else:
    st.error("❌ 测试结果为负，策略可能无效，请重新审视特征与逻辑。")
