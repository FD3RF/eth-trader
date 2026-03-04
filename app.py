# -*- coding: utf-8 -*-
"""
纸交易模拟（最终版：默认参数已优化）
策略参数基于回测选定：body=0.15, vol_period=15, break=0.001
功能：
- 支持手续费与滑点模拟
- 可调参数：突破周期、盈亏比、最小持有K线、实体强度阈值、成交量均线周期、突破阈值
- 可选样本外测试（训练/测试集划分）
- 参数自动扫描：一次性测试多组参数组合，汇总对比结果（含多空分项统计）
- 自动推荐：根据您的偏好计算综合得分，显示前三名最优组合
- 全面统计指标（最大回撤、夏普比率、盈亏比、连续亏损等）
- 权益曲线与回撤可视化（纯Streamlit实现）
- 适配 CSV 文件列名：ts, open, high, low, close, vol
"""

import streamlit as st
import pandas as pd
import numpy as np
from itertools import product

st.set_page_config(page_title="纸交易模拟(最终优化版)", layout="wide")
st.title("📈 纸交易模拟（最终优化版：默认参数已优化）")

# ==================== 侧边栏参数设置（默认值已设为最优组合）====================
st.sidebar.header("📌 策略参数（单次运行）")

# 基础参数
lookback = st.sidebar.number_input("突破周期 (lookback)", value=20, min_value=5, max_value=100, step=1)
rr_ratio = st.sidebar.number_input("盈亏比 (止盈/止损)", value=2.5, min_value=1.0, max_value=5.0, step=0.1)
min_hold = st.sidebar.number_input("最小持有K线数 (min_hold)", value=3, min_value=1, max_value=10, step=1)

# 信号过滤参数（默认值已设为 0.15, 15, 0.001）
break_threshold = st.sidebar.number_input("突破阈值 (比例)", value=0.001, format="%.4f", step=0.0005,
                                          help="突破幅度要求，降低此值会增加信号数量")
body_threshold = st.sidebar.number_input("实体强度阈值 (body_threshold)", value=0.15, min_value=0.1, max_value=0.9, step=0.05,
                                          help="K线实体占波动的比例要求，降低此值会增加信号数量")
vol_ma_period = st.sidebar.number_input("成交量均线周期 (vol_ma_period)", value=15, min_value=5, max_value=100, step=1,
                                        help="成交量确认使用的均线周期，缩短此值会增加信号数量")

# 手续费与滑点
st.sidebar.header("💰 交易成本")
fee_rate = st.sidebar.number_input("手续费率（双边，如0.001表示0.1%）", value=0.001, format="%.4f", step=0.0005)
slippage = st.sidebar.number_input("滑点（比例，如0.0005表示0.05%）", value=0.0005, format="%.4f", step=0.0001)
apply_costs = st.sidebar.checkbox("启用手续费与滑点", value=True)

# 样本外测试设置
st.sidebar.header("🔬 验证方式")
enable_oos = st.sidebar.checkbox("启用样本外测试（训练/测试集划分）", value=False)
if enable_oos:
    train_ratio = st.sidebar.slider("训练集比例（剩余为测试集）", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

# ==================== 参数扫描设置 ====================
st.sidebar.header("⚙️ 参数自动扫描")
enable_scan = st.sidebar.checkbox("启用参数扫描（将覆盖单次运行参数）", value=False)

if enable_scan:
    st.sidebar.markdown("请为以下参数输入候选值（用逗号分隔）")
    # 默认候选值设为围绕最优组合的范围
    body_thresholds_scan = st.sidebar.text_input("实体强度阈值", "0.15, 0.12, 0.18")
    vol_ma_periods_scan = st.sidebar.text_input("成交量均线周期", "15, 10, 20")
    break_thresholds_scan = st.sidebar.text_input("突破阈值", "0.001, 0.0008, 0.0012")
    run_scan = st.sidebar.button("🚀 运行参数扫描")

# ==================== 数据上传 ====================
uploaded_file = st.file_uploader("上传CSV文件（包含OHLCV数据）", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.lower() for c in df.columns]

# 处理时间戳列
if 'ts' in df.columns:
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.drop('ts', axis=1, inplace=True)

# 统一成交量列名
if 'vol' in df.columns:
    df['volume'] = df['vol']
    df.drop('vol', axis=1, inplace=True)

required_cols = ['open', 'high', 'low', 'close', 'volume']
if not all(col in df.columns for col in required_cols):
    st.error(f"数据必须包含以下列：{required_cols}")
    st.stop()

st.write(f"总数据行数：{len(df)}")
st.dataframe(df.head())

# ==================== 特征构建 ====================
def build_features(df, lookback, vol_ma_period):
    df = df.copy()
    df['high_max'] = df['high'].rolling(lookback).max().shift(1)
    df['low_min'] = df['low'].rolling(lookback).min().shift(1)
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
    df['vol_ma'] = df['volume'].rolling(vol_ma_period).mean()
    df.dropna(inplace=True)
    return df

# ==================== 信号生成 ====================
def generate_signal(df, i, break_threshold, body_threshold):
    row = df.iloc[i]
    close = row['close']
    high_max = row['high_max']
    low_min = row['low_min']
    body_ratio = row['body_ratio']
    volume = row['volume']
    vol_ma = row['vol_ma']

    in_middle = (close > low_min * 1.05) and (close < high_max * 0.95)
    strong_body = body_ratio > body_threshold
    valid_volume = volume > vol_ma
    break_up_strength = (close - high_max) / high_max
    break_down_strength = (low_min - close) / low_min

    if (close > high_max) and strong_body and valid_volume:
        if break_up_strength > break_threshold and not in_middle:
            return 1
    if (close < low_min) and strong_body and valid_volume:
        if break_down_strength > break_threshold and not in_middle:
            return -1
    return 0

# ==================== 模拟交易（返回多空统计） ====================
def simulate(df, start_idx, end_idx, fee_rate, slippage, apply_costs,
             rr_ratio, min_hold, break_threshold, body_threshold):
    """
    返回: records_df, equity_series, long_trades, short_trades,
          long_wins, short_wins, long_pnl, short_pnl
    """
    records = []
    equity_curve = []
    position = 0
    entry_price = 0
    entry_idx = 0
    cash = 10000.0
    hold = 0

    # 多空统计
    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0
    long_pnl = 0.0
    short_pnl = 0.0

    signals = [generate_signal(df, i, break_threshold, body_threshold) for i in range(len(df))]

    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

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
            if low <= stop_loss:
                exit_price = stop_loss * (1 - slippage) if apply_costs else stop_loss
            elif high >= take_profit:
                exit_price = take_profit * (1 - slippage) if apply_costs else take_profit
            elif hold >= min_hold and signal == -1:
                exit_price = open_price * (1 - slippage) if apply_costs else open_price
            if exit_price is not None:
                pnl = (exit_price - entry_price) * 1
                if apply_costs:
                    fee = (entry_price + exit_price) * fee_rate
                    pnl -= fee
                records.append({'entry_idx': entry_idx, 'exit_idx': i, 'direction': 'long',
                                'entry': entry_price, 'exit': exit_price, 'pnl': pnl})
                cash += pnl
                position = 0
                # 多头统计
                long_trades += 1
                long_pnl += pnl
                if pnl > 0:
                    long_wins += 1

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
                records.append({'entry_idx': entry_idx, 'exit_idx': i, 'direction': 'short',
                                'entry': entry_price, 'exit': exit_price, 'pnl': pnl})
                cash += pnl
                position = 0
                # 空头统计
                short_trades += 1
                short_pnl += pnl
                if pnl > 0:
                    short_wins += 1

        # 开仓
        if position == 0:
            if signal == 1:
                entry_price = open_price * (1 + slippage) if apply_costs else open_price
                position = 1
                entry_idx = i
            elif signal == -1:
                entry_price = open_price * (1 - slippage) if apply_costs else open_price
                position = -1
                entry_idx = i

        # 权益
        if position == 1:
            market_value = cash + (close - entry_price)
        elif position == -1:
            market_value = cash + (entry_price - close)
        else:
            market_value = cash
        equity_curve.append(market_value)

    return (pd.DataFrame(records),
            pd.Series(equity_curve, index=df.index[start_idx:end_idx]),
            long_trades, short_trades, long_wins, short_wins, long_pnl, short_pnl)

# ==================== 统计指标（修复无穷大） ====================
def compute_stats(records, equity_series, name="策略"):
    if len(records) == 0:
        return {f"{name}_交易数": 0}
    stats = {}
    stats[f"{name}_交易数"] = len(records)
    stats[f"{name}_总盈利"] = records['pnl'].sum()
    stats[f"{name}_平均单笔"] = records['pnl'].mean()
    stats[f"{name}_最大盈利"] = records['pnl'].max()
    stats[f"{name}_最大亏损"] = records['pnl'].min()
    stats[f"{name}_胜率"] = (records['pnl'] > 0).mean() * 100
    avg_win = records[records['pnl'] > 0]['pnl'].mean()
    avg_loss = records[records['pnl'] < 0]['pnl'].mean()
    if avg_loss != 0:
        stats[f"{name}_盈亏比"] = avg_win / abs(avg_loss)
    else:
        stats[f"{name}_盈亏比"] = 10.0  # 无亏损时设为较大值，避免inf
    losses = (records['pnl'] < 0).astype(int)
    loss_streaks = (losses.groupby((losses != losses.shift()).cumsum()).sum())
    stats[f"{name}_最大连续亏损次数"] = loss_streaks.max() if len(loss_streaks) > 0 else 0
    equity = equity_series.values
    cumulative = np.maximum.accumulate(equity)
    drawdown = (cumulative - equity) / cumulative
    stats[f"{name}_最大回撤 (%)"] = drawdown.max() * 100
    total_days = len(equity_series) * 5 / (60 * 24)
    if total_days > 0:
        annual_return = (equity[-1] / equity[0]) ** (365 / total_days) - 1
        stats[f"{name}_年化收益率 (%)"] = annual_return * 100
    else:
        stats[f"{name}_年化收益率 (%)"] = 0
    daily_returns = pd.Series(equity).pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() != 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        stats[f"{name}_夏普比率"] = sharpe
    else:
        stats[f"{name}_夏普比率"] = 0
    return stats

# ==================== 执行模拟（单次运行或扫描） ====================
if enable_scan and run_scan:
    # 解析用户输入的候选值
    body_list = [float(x.strip()) for x in body_thresholds_scan.split(',')]
    vol_list = [int(x.strip()) for x in vol_ma_periods_scan.split(',')]
    break_list = [float(x.strip()) for x in break_thresholds_scan.split(',')]

    # 生成所有参数组合
    param_combinations = list(product(body_list, vol_list, break_list))
    total_combos = len(param_combinations)
    st.subheader(f"🔍 参数扫描结果（共 {total_combos} 组）")

    # 用于存储结果的列表
    scan_results = []

    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (b_th, v_period, br_th) in enumerate(param_combinations):
        status_text.text(f"正在测试组合 {idx+1}/{total_combos}：实体={b_th}, 成交量周期={v_period}, 突破阈值={br_th}")

        try:
            # 构建特征（注意：lookback 使用侧边栏固定值）
            df_feat_scan = build_features(df, lookback, v_period)
            if len(df_feat_scan) == 0:
                continue

            if enable_oos:
                split_point = int(len(df_feat_scan) * train_ratio)
                test_df = df_feat_scan.iloc[split_point:]
                test_records, test_equity, lt, short_t, lw, sw, lpnl, spnl = simulate(
                    test_df, 0, len(test_df),
                    fee_rate, slippage, apply_costs,
                    rr_ratio, min_hold, br_th, b_th
                )
                stats = compute_stats(test_records, test_equity, "测试集")
                scan_results.append({
                    'body_threshold': b_th,
                    'vol_ma_period': v_period,
                    'break_threshold': br_th,
                    '交易数': stats['测试集_交易数'],
                    '多头交易数': lt,
                    '空头交易数': short_t,
                    '多头胜率': (lw / lt * 100) if lt > 0 else 0,
                    '空头胜率': (sw / short_t * 100) if short_t > 0 else 0,
                    '多头盈利': lpnl,
                    '空头盈利': spnl,
                    '胜率': stats['测试集_胜率'],
                    '总盈利': stats['测试集_总盈利'],
                    '最大回撤': stats['测试集_最大回撤 (%)'],
                    '夏普比率': stats['测试集_夏普比率'],
                    '盈亏比': stats['测试集_盈亏比'],
                })
            else:
                records, equity, lt, short_t, lw, sw, lpnl, spnl = simulate(
                    df_feat_scan, 0, len(df_feat_scan),
                    fee_rate, slippage, apply_costs,
                    rr_ratio, min_hold, br_th, b_th
                )
                stats = compute_stats(records, equity, "全样本")
                scan_results.append({
                    'body_threshold': b_th,
                    'vol_ma_period': v_period,
                    'break_threshold': br_th,
                    '交易数': stats['全样本_交易数'],
                    '多头交易数': lt,
                    '空头交易数': short_t,
                    '多头胜率': (lw / lt * 100) if lt > 0 else 0,
                    '空头胜率': (sw / short_t * 100) if short_t > 0 else 0,
                    '多头盈利': lpnl,
                    '空头盈利': spnl,
                    '胜率': stats['全样本_胜率'],
                    '总盈利': stats['全样本_总盈利'],
                    '最大回撤': stats['全样本_最大回撤 (%)'],
                    '夏普比率': stats['全样本_夏普比率'],
                    '盈亏比': stats['全样本_盈亏比'],
                })
        except Exception as e:
            st.warning(f"组合 {b_th}, {v_period}, {br_th} 运行出错：{e}")
            scan_results.append({
                'body_threshold': b_th,
                'vol_ma_period': v_period,
                'break_threshold': br_th,
                '交易数': 0,
                '多头交易数': 0,
                '空头交易数': 0,
                '多头胜率': 0,
                '空头胜率': 0,
                '多头盈利': 0,
                '空头盈利': 0,
                '胜率': 0,
                '总盈利': 0,
                '最大回撤': 0,
                '夏普比率': 0,
                '盈亏比': 0,
            })

        progress_bar.progress((idx + 1) / total_combos)

    status_text.text("扫描完成！")

    if scan_results:
        result_df = pd.DataFrame(scan_results)
        # 替换无穷大和NaN为0，防止渲染错误
        result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)
        st.dataframe(result_df)

        # ==================== 自动推荐功能 ====================
        st.subheader("🎯 自动参数推荐")
        col1, col2 = st.columns([1, 3])
        with col1:
            recommend_strategy = st.radio(
                "推荐偏好",
                options=["平衡", "优先交易数", "优先夏普比率", "优先胜率"],
                index=0,
                help="选择后系统将根据权重计算综合得分，并显示排名前三的参数组合"
            )
        with col2:
            # 定义权重
            if recommend_strategy == "优先交易数":
                w_trades, w_sharpe, w_winrate, w_drawdown = 0.5, 0.2, 0.2, -0.1
            elif recommend_strategy == "优先夏普比率":
                w_trades, w_sharpe, w_winrate, w_drawdown = 0.2, 0.5, 0.2, -0.1
            elif recommend_strategy == "优先胜率":
                w_trades, w_sharpe, w_winrate, w_drawdown = 0.2, 0.2, 0.5, -0.1
            else:  # 平衡
                w_trades, w_sharpe, w_winrate, w_drawdown = 0.3, 0.3, 0.3, -0.1

            # 计算综合得分（归一化后加权）
            df_norm = result_df.copy()
            # 归一化正向指标（越大越好）
            for col in ['交易数', '胜率', '夏普比率']:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                df_norm[col + '_norm'] = (df_norm[col] - min_val) / (max_val - min_val + 1e-9)
            # 最大回撤取负（越小越好）
            min_dd = df_norm['最大回撤'].min()
            max_dd = df_norm['最大回撤'].max()
            df_norm['最大回撤_norm'] = 1 - (df_norm['最大回撤'] - min_dd) / (max_dd - min_dd + 1e-9)

            # 加权综合得分
            df_norm['综合得分'] = (w_trades * df_norm['交易数_norm'] +
                                  w_sharpe * df_norm['夏普比率_norm'] +
                                  w_winrate * df_norm['胜率_norm'] +
                                  w_drawdown * df_norm['最大回撤_norm'])

            # 选取前三名
            top3 = df_norm.nlargest(3, '综合得分')[
                ['body_threshold', 'vol_ma_period', 'break_threshold',
                 '交易数', '胜率', '夏普比率', '最大回撤', '综合得分']
            ]
            # 格式化显示
            st.dataframe(
                top3.style.format({
                    '胜率': '{:.2f}%',
                    '夏普比率': '{:.3f}',
                    '最大回撤': '{:.2f}%',
                    '综合得分': '{:.3f}'
                })
            )

        # 下载按钮
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载扫描结果 CSV",
            data=csv,
            file_name="parameter_scan_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("没有有效的扫描结果。")

else:
    # 单次运行（默认参数已优化）
    df_feat = build_features(df, lookback, vol_ma_period)

    if enable_oos:
        split_point = int(len(df_feat) * train_ratio)
        train_df = df_feat.iloc[:split_point]
        test_df = df_feat.iloc[split_point:]

        st.subheader("📊 训练集结果")
        train_records, train_equity, _, _, _, _, _, _ = simulate(
            train_df, 0, len(train_df),
            fee_rate, slippage, apply_costs,
            rr_ratio, min_hold, break_threshold, body_threshold
        )
        if len(train_records) > 0:
            st.dataframe(train_records)
        else:
            st.write("训练集无交易记录。")

        st.subheader("📊 测试集结果")
        test_records, test_equity, _, _, _, _, _, _ = simulate(
            test_df, 0, len(test_df),
            fee_rate, slippage, apply_costs,
            rr_ratio, min_hold, break_threshold, body_threshold
        )
        if len(test_records) > 0:
            st.dataframe(test_records)
        else:
            st.write("测试集无交易记录。")
    else:
        st.subheader("📊 全样本模拟结果")
        records, equity, _, _, _, _, _, _ = simulate(
            df_feat, 0, len(df_feat),
            fee_rate, slippage, apply_costs,
            rr_ratio, min_hold, break_threshold, body_threshold
        )

    # 统计展示
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
            equity_df = pd.DataFrame({
                '权益': equity.values,
                '峰值': np.maximum.accumulate(equity.values),
                '回撤': (np.maximum.accumulate(equity.values) - equity.values)
            }, index=equity.index)
            st.line_chart(equity_df[['权益', '峰值']])
            st.area_chart(equity_df[['回撤']])

st.success("模拟完成（最终优化版：默认参数已设为最优组合）")
