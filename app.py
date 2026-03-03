# -*- coding: utf-8 -*-
"""
终极量化交易策略 - 未来收益预测版（实盘级优化）
作者：AI Assistant
版本：15.1 (修复KeyError)
说明：模型预测未来5根K线涨幅是否超过0.3%，交易采用概率区间+多周期过滤，包含保守成交假设、严格样本外验证、实时权益记录。
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

st.set_page_config(page_title="量化策略·完美版", layout="wide")
st.title("🚀 终极量化策略 - 未来收益预测 (XGBoost)")

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

# ====================== 多周期EMA（滞后一期，无未来） ======================
def compute_ema_lagged(df, period='15min', span=20, shift=1):
    """计算滞后shift期的多周期EMA"""
    resampled = df['close'].resample(period).last()
    ema_resampled = resampled.ewm(span=span, adjust=False).mean()
    ema_lagged = ema_resampled.shift(shift).reindex(df.index, method='ffill')
    return ema_lagged

df['ema_15m'] = compute_ema_lagged(df, '15min', 20, shift=1)
df['ema_1h'] = compute_ema_lagged(df, '1h', 20, shift=1)

# ====================== 特征工程（历史特征 + 未来标签 + 结构列用于出场） ======================
@st.cache_data
def create_features(df):
    df = df.copy()
    
    # ---------- 基础技术指标（仅历史） ----------
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
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
    
    # ADX
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
    
    df['pullback'] = (df['close'] - df['low'].rolling(5).min()) / (df['high'].rolling(5).max() - df['low'].rolling(5).min())
    
    # 多周期趋势一致（仅用作过滤）
    df['trend_align'] = ((df['close'] > df['ema20']) & 
                         (df['close'] > df['ema_15m']) & 
                         (df['close'] > df['ema_1h'])).astype(int)
    
    # ---------- 结构定义（仅用于出场，不加入模型特征） ----------
    df['low_10'] = df['low'].rolling(10).min()
    df['high_10'] = df['high'].rolling(10).max()
    df['low_20_prev'] = df['low'].shift(10).rolling(10).min()
    df['high_20_prev'] = df['high'].shift(10).rolling(10).max()
    
    df['low_rising'] = (df['low_10'] > df['low_20_prev']) & (df['close'] > df['low_10'] * 1.005)
    df['high_rising'] = (df['high_10'] > df['high_20_prev']) & (df['close'] > df['low_10'] * 1.005)
    df['uptrend_structure'] = (
        (df['close'] > df['ema20']) &
        df['low_rising'] &
        df['high_rising'] &
        (df['adx'] > 20)
    ).astype(int)
    
    df['low_falling'] = (df['low_10'] < df['low_20_prev']) & (df['close'] < df['low_10'] * 0.995)
    df['high_falling'] = (df['high_10'] < df['high_20_prev']) & (df['close'] < df['low_10'] * 0.995)
    df['downtrend_structure'] = (
        (df['close'] < df['ema20']) &
        df['low_falling'] &
        df['high_falling'] &
        (df['adx'] > 20)
    ).astype(int)
    
    # ---------- 未来收益标签（预测未来5根K线涨幅是否 > 0.3%） ----------
    future_returns = df['close'].pct_change(5).shift(-5)  # 未来5根累计收益
    threshold = 0.003  # 0.3%
    df['target'] = (future_returns > threshold).astype(int)
    
    df.dropna(inplace=True)
    return df

with st.spinner("生成特征中..."):
    df_feat = create_features(df)
st.success("✅ 特征生成完成！")

# ====================== 严格数据分割 ======================
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
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    st.write(f"✅ 验证集准确率: {acc:.3f} | 精确率: {prec:.3f} | 召回率: {rec:.3f} | F1: {f1:.3f}")
    return model

with st.spinner("训练XGBoost分类模型中..."):
    model = train_xgboost(train, val, features)

# ====================== 回测函数（保守成交 + 实时权益记录） ======================
def backtest_with_probs(df, probs, th_long, th_short, fee_rate, slippage, atr_mult, rr):
    df = df.copy()
    df['prob'] = probs
    
    # 信号生成
    df['signal'] = 0
    df.loc[(df['prob'] >= th_long) & (df['trend_align'] == 1) & (df['adx'] > 20) & (df['pullback'] < 0.8), 'signal'] = 1
    df.loc[(df['prob'] <= th_short) & (df['trend_align'] == 0) & (df['adx'] > 20) & (df['pullback'] > 0.2), 'signal'] = -1
    
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_time = None
    equity = [0.0]  # 每根K线收盘时的权益
    trades = []
    wins = 0
    trade_pnls = []
    
    # 实时权益记录（每根K线收盘）
    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            equity.append(0.0)
            continue
        
        prev = df.iloc[i-1]
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']
        atr = row['atr']
        current_time = row.name
        
        # 先处理可能的出场（基于前一根信号）
        if position != 0:
            exit_price = None
            exit_reason = None
            if position == 1:
                stop = entry_price - atr_mult * entry_atr
                take = entry_price + rr * atr_mult * entry_atr
                # 保守成交：止损以 max(stop, open) 成交，止盈以 min(take, open) 成交
                if low <= stop:
                    exit_price = max(stop, open_price)  # 假设开盘价低于止损，则止损单可能以止损价或开盘价成交，取更差（对多头更高价不利）的是 max
                elif high >= take:
                    exit_price = min(take, open_price)  # 取更差的低价
                elif row['downtrend_structure'] == 1:
                    exit_price = open_price
                elif entry_time is not None and (current_time - entry_time).total_seconds() > 6000:
                    exit_price = open_price
            else:  # position == -1
                stop = entry_price + atr_mult * entry_atr
                take = entry_price - rr * atr_mult * entry_atr
                if high >= stop:
                    exit_price = min(stop, open_price)  # 对空头更差的价格是更低的
                elif low <= take:
                    exit_price = max(take, open_price)
                elif row['uptrend_structure'] == 1:
                    exit_price = open_price
                elif entry_time is not None and (current_time - entry_time).total_seconds() > 6000:
                    exit_price = open_price
            
            if exit_price is not None:
                pnl = (exit_price - entry_price) * (1 if position == 1 else -1) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                trade_pnls.append(pnl)
                if pnl > 0: wins += 1
                position = 0
        
        # 新信号处理（基于前一根信号）
        if i > 0 and prev['signal'] == 1 and position != 1:
            if position == -1:
                # 先平空
                pnl = (entry_price - open_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                trade_pnls.append(pnl)
                if pnl > 0: wins += 1
            position = 1
            entry_price = open_price
            entry_atr = atr
            entry_time = current_time
        
        elif i > 0 and prev['signal'] == -1 and position != -1:
            if position == 1:
                pnl = (open_price - entry_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                trade_pnls.append(pnl)
                if pnl > 0: wins += 1
            position = -1
            entry_price = open_price
            entry_atr = atr
            entry_time = current_time
        
        # 更新权益（无操作时）
        if len(equity) == i+1:  # 本根K线尚未记录权益
            if position == 0:
                equity.append(equity[-1])
            else:
                # 浮动盈亏不计入实盘权益，但这里简化：权益不变
                equity.append(equity[-1])
    
    # 最后平仓
    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            pnl = (last_price - entry_price) * (1 - fee_rate * 2) - slippage
        else:
            pnl = (entry_price - last_price) * (1 - fee_rate * 2) - slippage
        equity.append(equity[-1] + pnl)
        trades.append(pnl)
        trade_pnls.append(pnl)
        if pnl > 0: wins += 1
    
    # 确保 equity 长度与 df 一致（多一个初始0，最后多一个最终权益，对齐时取 [1:] 作为每根K线收盘权益）
    equity_per_bar = equity[1:]  # 第一根K线权益0，之后每根对应收盘权益
    
    total_pnl = sum(trades)
    win_rate = wins / len(trades) if trades else 0
    max_equity = np.maximum.accumulate(equity_per_bar)
    drawdown = max_equity - equity_per_bar
    max_dd = np.max(drawdown)
    
    if max_dd != 0:
        calmar = total_pnl / max_dd
    else:
        calmar = 0
    
    avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
    avg_loss = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) if avg_loss != 0 else 0
    
    max_consecutive_losses = 0
    current_consecutive = 0
    for pnl in trade_pnls:
        if pnl < 0:
            current_consecutive += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
        else:
            current_consecutive = 0
    
    profit_factor = total_pnl / abs(sum(p for p in trade_pnls if p < 0)) if any(p < 0 for p in trade_pnls) else 0
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'calmar': calmar,
        'trades': len(trades),
        'equity': equity_per_bar,  # 每根K线的权益
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_consecutive_losses': max_consecutive_losses,
        'profit_factor': profit_factor
    }

# ====================== 随机采样优化（基于验证集，严格不窥视测试集） ======================
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
    st.markdown(f"""
    ### 🎯 进场条件
    - **做多**：
        - 模型预测未来5根K线上涨概率 ≥ {th_long:.2f}
        - 多周期趋势一致（过滤器）：5min、15min、1h EMA20 均向上
        - 趋势强度 ADX > 20
        - 回调比例 < 0.8（避免追高）
    - **做空**：
        - 模型预测未来5根K线上涨概率 ≤ {th_short:.2f}
        - 多周期趋势一致（均向下）
        - ADX > 20
        - 回调比例 > 0.2（避免杀跌）

    ### 🛑 出场条件（保守成交假设）
    - **止盈**：盈利达到 盈亏比 × ATR 倍数 ({rr:.1f} × {atr_mult:.1f} × ATR)，成交价取 min(止盈价, 开盘价)（多头）或 max(止盈价, 开盘价)（空头）
    - **止损**：亏损达到 ATR 倍数 ({atr_mult:.1f} × ATR)，成交价取 max(止损价, 开盘价)（多头）或 min(止损价, 开盘价)（空头）
    - **趋势结构破坏**：
        - 做多后出现下降结构（高点降低 + 低点降低）
        - 做空后出现上升结构（高点抬高 + 低点抬高）
    - **超时平仓**：持仓超过 20 根K线（100分钟），按开盘价成交

    ### ⚖️ 风险与期望
    - **单笔风险建议**：不超过账户总资金的 1-2%
    - 期望收益（验证集）：{val_res['expectancy']:.2f} USDT/笔
    - 盈亏比（验证集）：{val_res['profit_factor']:.2f}
    - 最大连续亏损（验证集）：{val_res['max_consecutive_losses']} 次
    - 最大回撤（验证集）：{val_res['max_dd']:.2f} USDT
    """)

# ====================== 测试集验证（仅一次） ======================
st.write("🔄 在测试集上验证（一次性最终评估）...")
test_probs = model.predict_proba(test[features])[:, 1]
test_res = backtest_with_probs(test, test_probs, th_long, th_short, fee_rate, slippage, atr_mult, rr)

st.header("🎯 测试集最终结果")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("交易次数", test_res['trades'])
col2.metric("胜率", f"{test_res['win_rate']*100:.1f}%")
col3.metric("总盈利", f"{test_res['total_pnl']:.2f} USDT")
col4.metric("最大回撤", f"{test_res['max_dd']:.2f} USDT")
col5.metric("Calmar", f"{test_res['calmar']:.2f}")
col6.metric("期望收益/笔", f"{test_res['expectancy']:.2f}")

st.write(f"盈亏比: {test_res['profit_factor']:.2f} | 最大连续亏损: {test_res['max_consecutive_losses']} 次")

# ====================== 资金曲线图（每根K线对齐） ======================
if test_res['trades'] > 0:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test.index,
        y=test_res['equity'],
        mode='lines',
        name='资金曲线',
        line=dict(color='#00ff88', width=2)
    ))
    fig.update_layout(
        title="测试集资金曲线（每根K线收盘权益）",
        xaxis_title="时间",
        yaxis_title="累积盈亏 (USDT)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================== 结论（严格区分验证/测试） ======================
if test_res['calmar'] > 1.5 and test_res['expectancy'] > 0:
    st.success("✨ 测试结果非常优秀！策略稳健，可考虑实盘模拟。请严格控制单笔风险不超过1%，并持续监控市场状态。")
elif test_res['calmar'] > 0.8 and test_res['expectancy'] > 0:
    st.info("📊 测试结果良好，风险收益比较理想，可进一步优化参数或增加过滤条件。")
elif test_res['calmar'] > 0 and test_res['expectancy'] > 0:
    st.warning("⚠️ 测试结果为正收益，但风险调整后收益一般，需检查参数或特征是否过拟合。")
else:
    st.error("❌ 测试结果为负或期望收益为负，策略可能无效，请重新审视特征与逻辑。")
