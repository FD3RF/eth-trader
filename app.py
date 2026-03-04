# -*- coding: utf-8 -*-
"""
纸交易模拟（优化版，无matplotlib依赖）
功能：
- 支持手续费与滑点模拟
- 计算最大回撤、夏普比率等风控指标
- 可选样本外测试（训练/测试集划分）
- 权益曲线与回撤可视化（纯Streamlit实现）
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.set_page_config(page_title="纸交易模拟(优化版)", layout="wide")
st.title("📈 纸交易模拟（优化版：手续费+回撤+样本外）")

# ==================== 侧边栏参数设置 ====================
st.sidebar.header("模拟参数设置")

# 手续费与滑点
fee_rate = st.sidebar.number_input("手续费率（双边，如0.001表示0.1%）", value=0.001, format="%.4f", step=0.0005)
slippage = st.sidebar.number_input("滑点（比例，如0.0005表示0.05%）", value=0.0005, format="%.4f", step=0.0001)
apply_costs = st.sidebar.checkbox("启用手续费与滑点", value=True)

# 样本外测试设置
enable_oos = st.sidebar.checkbox("启用样本外测试（训练/测试集划分）", value=False)
if enable_oos:
    train_ratio = st.sidebar.slider("训练集比例（剩余为测试集）", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

# 策略参数（可调）
lookback = st.sidebar.number_input("突破周期", value=20, min_value=5, max_value=100, step=1)
rr_ratio = st.sidebar.number_input("盈亏比（止盈/止损）", value=2.5, min_value=1.0, max_value=5.0, step=0.1)
min_hold = st.sidebar.number_input("最小持有K线数", value=3, min_value=1, max_value=10, step=1)
break_threshold = st.sidebar.number_input("突破阈值（比例）", value=0.002, format="%.4f", step=0.0005)

# ==================== 数据上传 ====================
uploaded_file = st.file_uploader("上传CSV文件（包含OHLCV数据）", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.lower() for c in df.columns]

# 统一列名（假设可能包含 'vol' 或 'volume'）
if 'vol' in df.columns:
    df['volume'] = df['vol']
elif 'volume' not in df.columns:
    st.error("数据中缺少成交量列（需要 'vol' 或 'volume'）")
    st.stop()

required_cols = ['open', 'high', 'low', 'close', 'volume']
if not all(col in df.columns for col in required_cols):
    st.error(f"数据必须包含以下列：{required_cols}")
    st.stop()

st.write(f"总数据行数：{len(df)}")
st.dataframe(df.head())

# ==================== 特征构建 ====================
def build_features(df, lookback):
    df = df.copy()
    df['high_max'] = df['high'].rolling(lookback).max().shift(1)
    df['low_min'] = df['low'].rolling(lookback).min().shift(1)
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df.dropna(inplace=True)
    return df

df_feat = build_features(df, lookback)

# ==================== 信号生成 ====================
def generate_signal(df, i, break_threshold):
    row = df.iloc[i]
    close = row['close']
    high_max = row['high_max']
    low_min = row['low_min']
    body_ratio = row['body_ratio']
    volume = row['volume']
    vol_ma = row['vol_ma']

    # 是否在区间中部（避免假突破）
    in_middle = (close > low_min * 1.05) and (close < high_max * 0.95)

    # 实体强度与成交量确认
    strong_body = body_ratio > 0.45
    valid_volume = volume > vol_ma

    # 突破幅度
    break_up_strength = (close - high_max) / high_max
    break_down_strength = (low_min - close) / low_min

    if (close > high_max) and strong_body and valid_volume:
        if break_up_strength > break_threshold and not in_middle:
            return 1
    if (close < low_min) and strong_body and valid_volume:
        if break_down_strength > break_threshold and not in_middle:
            return -1
    return 0

# ==================== 模拟交易（单段） ====================
def simulate(df, start_idx, end_idx, fee_rate, slippage, apply_costs, rr_ratio, min_hold, break_threshold):
    """
    在 df 的 [start_idx, end_idx) 区间上模拟交易
    返回交易记录DataFrame和权益序列
    """
    records = []          # 每笔交易的字典
    equity_curve = []     # 每根K线收盘时的权益（初始资金假设为10000）
    position = 0
    entry_price = 0
    entry_idx = 0
    cash = 10000.0        # 初始资金
    hold = 0

    # 预计算信号（避免每步重复计算）
    signals = [generate_signal(df, i, break_threshold) for i in range(len(df))]

    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        # 更新持仓时间
        if position != 0:
            hold += 1
        else:
            hold = 0

        signal = signals[i]

        # 平仓逻辑
        if position == 1:  # 多头
            stop_loss = entry_price - row['range'] * 0.5
            take_profit = entry_price + (entry_price - stop_loss) * rr_ratio
            exit_price = None

            # 止损
            if low <= stop_loss:
                exit_price = stop_loss * (1 - slippage) if apply_costs else stop_loss
            # 止盈
            elif high >= take_profit:
                exit_price = take_profit * (1 - slippage) if apply_costs else take_profit
            # 反向信号平仓
            elif hold >= min_hold and signal == -1:
                exit_price = open_price * (1 - slippage) if apply_costs else open_price

            if exit_price is not None:
                pnl = (exit_price - entry_price) * 1  # 假设1单位
                if apply_costs:
                    fee = (entry_price + exit_price) * fee_rate  # 双边手续费
                    pnl -= fee
                records.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': 'long',
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': pnl
                })
                cash += pnl
                position = 0

        elif position == -1:  # 空头
            stop_loss = entry_price + row['range'] * 0.5
            take_profit = entry_price - (stop_loss - entry_price) * rr_ratio
            exit_price = None

            if high >= stop_loss:
                exit_price = stop_loss * (1 + slippage) if apply_costs else stop_loss
            elif low <= take_profit:
                exit_price = take_profit * (1 + slippage) if apply_costs else take_profit
            elif hold >= min_hold and signal == 1:
                exit_price = open_price * (1 + slippage) if apply_costs else open_price

            if exit_price is not None:
                pnl = (entry_price - exit_price) * 1
                if apply_costs:
                    fee = (entry_price + exit_price) * fee_rate
                    pnl -= fee
                records.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': 'short',
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': pnl
                })
                cash += pnl
                position = 0

        # 开仓逻辑
        if position == 0:
            if signal == 1:
                entry_price = open_price * (1 + slippage) if apply_costs else open_price
                position = 1
                entry_idx = i
            elif signal == -1:
                entry_price = open_price * (1 - slippage) if apply_costs else open_price
                position = -1
                entry_idx = i

        # 记录权益（现金 + 持仓市值）
        if position == 1:
            market_value = cash + (close - entry_price)  # 浮动盈亏
        elif position == -1:
            market_value = cash + (entry_price - close)
        else:
            market_value = cash
        equity_curve.append(market_value)

    return pd.DataFrame(records), pd.Series(equity_curve, index=df.index[start_idx:end_idx])

# ==================== 执行模拟（考虑样本外划分） ====================
if enable_oos:
    split_point = int(len(df_feat) * train_ratio)
    train_df = df_feat.iloc[:split_point]
    test_df = df_feat.iloc[split_point:]

    st.subheader("📊 训练集结果")
    train_records, train_equity = simulate(train_df, 0, len(train_df), fee_rate, slippage,
                                            apply_costs, rr_ratio, min_hold, break_threshold)
    if len(train_records) > 0:
        st.dataframe(train_records)
    else:
        st.write("训练集无交易记录。")

    st.subheader("📊 测试集结果")
    test_records, test_equity = simulate(test_df, 0, len(test_df), fee_rate, slippage,
                                          apply_costs, rr_ratio, min_hold, break_threshold)
    if len(test_records) > 0:
        st.dataframe(test_records)
    else:
        st.write("测试集无交易记录。")
else:
    # 全数据模拟
    st.subheader("📊 全样本模拟结果")
    records, equity = simulate(df_feat, 0, len(df_feat), fee_rate, slippage,
                                apply_costs, rr_ratio, min_hold, break_threshold)

# ==================== 统计指标计算函数 ====================
def compute_stats(records, equity_series, name="策略"):
    """输入交易记录DataFrame和权益序列，返回统计指标字典"""
    if len(records) == 0:
        return {f"{name}_交易数": 0}

    stats_dict = {}

    # 基础统计
    stats_dict[f"{name}_交易数"] = len(records)
    stats_dict[f"{name}_总盈利"] = records['pnl'].sum()
    stats_dict[f"{name}_平均单笔"] = records['pnl'].mean()
    stats_dict[f"{name}_最大盈利"] = records['pnl'].max()
    stats_dict[f"{name}_最大亏损"] = records['pnl'].min()
    stats_dict[f"{name}_胜率"] = (records['pnl'] > 0).mean() * 100
    stats_dict[f"{name}_盈亏比"] = records[records['pnl'] > 0]['pnl'].mean() / abs(records[records['pnl'] < 0]['pnl'].mean()) if len(records[records['pnl'] < 0]) > 0 else np.inf

    # 连续亏损
    losses = (records['pnl'] < 0).astype(int)
    loss_streaks = (losses.groupby((losses != losses.shift()).cumsum()).sum())
    stats_dict[f"{name}_最大连续亏损次数"] = loss_streaks.max() if len(loss_streaks) > 0 else 0

    # 权益曲线相关
    equity = equity_series.values
    cumulative = np.maximum.accumulate(equity)
    drawdown = (cumulative - equity) / cumulative
    max_dd = drawdown.max() * 100
    stats_dict[f"{name}_最大回撤 (%)"] = max_dd

    # 年化收益率（假设一年252个交易日，5分钟K线需换算，这里简单按总交易日数估算）
    total_days = len(equity_series) * 5 / (60 * 24)  # 5分钟K线数转换为天数
    if total_days > 0:
        annual_return = (equity[-1] / equity[0]) ** (365 / total_days) - 1
        stats_dict[f"{name}_年化收益率 (%)"] = annual_return * 100
    else:
        stats_dict[f"{name}_年化收益率 (%)"] = 0

    # 夏普比率（年化，假设无风险利率0）
    daily_returns = pd.Series(equity).pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() != 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        stats_dict[f"{name}_夏普比率"] = sharpe
    else:
        stats_dict[f"{name}_夏普比率"] = 0

    return stats_dict

# ==================== 展示统计结果 ====================
if enable_oos:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("训练集统计")
        train_stats = compute_stats(train_records, train_equity, "训练集")
        for k, v in train_stats.items():
            if isinstance(v, float):
                st.metric(k, f"{v:.2f}")
            else:
                st.metric(k, v)
        if len(train_records) > 0:
            st.line_chart(train_records['pnl'].cumsum())

    with col2:
        st.subheader("测试集统计")
        test_stats = compute_stats(test_records, test_equity, "测试集")
        for k, v in test_stats.items():
            if isinstance(v, float):
                st.metric(k, f"{v:.2f}")
            else:
                st.metric(k, v)
        if len(test_records) > 0:
            st.line_chart(test_records['pnl'].cumsum())

    # 对比
    if len(train_records) > 0 and len(test_records) > 0:
        st.subheader("训练集 vs 测试集")
        compare_df = pd.DataFrame({
            '指标': ['交易数', '胜率(%)', '总盈利', '最大回撤(%)', '夏普比率'],
            '训练集': [train_stats['训练集_交易数'], train_stats['训练集_胜率'], train_stats['训练集_总盈利'],
                     train_stats['训练集_最大回撤 (%)'], train_stats['训练集_夏普比率']],
            '测试集': [test_stats['测试集_交易数'], test_stats['测试集_胜率'], test_stats['测试集_总盈利'],
                     test_stats['测试集_最大回撤 (%)'], test_stats['测试集_夏普比率']]
        })
        st.dataframe(compare_df)
else:
    stats = compute_stats(records, equity, "全样本")
    # 以指标卡形式展示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("交易数", stats['全样本_交易数'])
        st.metric("胜率", f"{stats['全样本_胜率']:.2f}%")
    with col2:
        st.metric("总盈利", f"{stats['全样本_总盈利']:.2f}")
        st.metric("平均单笔", f"{stats['全样本_平均单笔']:.2f}")
    with col3:
        st.metric("最大盈利", f"{stats['全样本_最大盈利']:.2f}")
        st.metric("最大亏损", f"{stats['全样本_最大亏损']:.2f}")
    with col4:
        st.metric("最大回撤", f"{stats['全样本_最大回撤 (%)']:.2f}%")
        st.metric("夏普比率", f"{stats['全样本_夏普比率']:.2f}")

    st.subheader("累计盈亏曲线")
    if len(records) > 0:
        st.line_chart(records['pnl'].cumsum())

    st.subheader("权益曲线与回撤")
    if len(equity) > 0:
        # 准备数据
        equity_df = pd.DataFrame({
            '权益': equity.values,
            '峰值': np.maximum.accumulate(equity.values),
            '回撤': (np.maximum.accumulate(equity.values) - equity.values)
        }, index=equity.index)
        # 显示权益曲线
        st.line_chart(equity_df[['权益', '峰值']])
        # 显示回撤面积图（用 area_chart 模拟）
        st.area_chart(equity_df[['回撤']])

st.success("模拟完成（优化版，无matplotlib依赖）")
