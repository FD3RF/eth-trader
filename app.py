# -*- coding: utf-8 -*-
"""
纯K线吞没形态策略（优化版）
初始资金100，只用K线（无任何指标）
信号：看涨/看跌吞没形态
止损：固定点数
止盈：止损点数的倍数
支持移动止损、双向交易、自动参数优化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("📈 纯K线吞没形态策略（优化版，初始资金100）")

# ====== 侧边栏 ======
st.sidebar.header("运行设置")
mode = st.sidebar.radio("模式", ["手动调参", "自动优化"])

if mode == "手动调参":
    stop_points = st.sidebar.number_input("止损点数", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
    tp_ratio = st.sidebar.number_input("止盈倍数 (相对于止损)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    use_trailing = st.sidebar.checkbox("移动止损", value=False)
    trail_period = st.sidebar.slider("移动止损周期", 5, 50, 10) if use_trailing else 0
    enable_short = st.sidebar.checkbox("启用做空", value=True)

# ====== 上传数据 ======
file = st.file_uploader("上传CSV (需包含 open,high,low,close,datetime)", type=["csv"])
if not file:
    st.info("请上传数据文件")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            st.error(f"缺少{col}")
            st.stop()
    return df

df_full = load_data(file)
st.success(f"数据共 {len(df_full)} 根K线")

# 分割训练/测试
split = int(len(df_full) * 0.6)
df_train = df_full.iloc[:split].copy()
df_test = df_full.iloc[split:].copy()
st.info(f"训练集 {len(df_train)} | 测试集 {len(df_test)}")

# ====== 特征工程：只识别吞没形态 ======
def prepare_features(df):
    df = df.copy()
    df['prev_open'] = df['open'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    # 看涨吞没条件（当前阳线完全覆盖前一根阴线）
    bull = (df['prev_close'] < df['prev_open']) & (df['close'] > df['open']) & \
           (df['open'] < df['prev_close']) & (df['close'] > df['prev_open'])
    # 看跌吞没条件（当前阴线完全覆盖前一根阳线）
    bear = (df['prev_close'] > df['prev_open']) & (df['close'] < df['open']) & \
           (df['open'] > df['prev_close']) & (df['close'] < df['prev_open'])

    df['long_signal'] = bull.astype(int)
    df['short_signal'] = bear.astype(int)
    df.dropna(inplace=True)
    return df

# ====== 回测核心（初始资金100，固定点数止损） ======
def backtest(df, params):
    """
    params: dict 包含 stop_points, tp_ratio, use_trailing, trail_period, enable_short
    """
    df = prepare_features(df)
    if len(df) == 0:
        return None

    initial_capital = 100.0
    equity = [initial_capital]
    trades = []                      # 每笔盈亏点数
    position = 0                      # 1多 -1空
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trail_stop = 0.0                  # 移动止损价

    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i-1]

        # ----- 平仓检查 -----
        if position != 0:
            exit_signal = False
            exit_price = cur['open']
            if position == 1:  # 多头
                if cur['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_signal = True
                elif params['tp_ratio'] > 0 and cur['high'] >= take_profit:
                    exit_price = take_profit
                    exit_signal = True
                elif params['use_trailing'] and params['trail_period'] > 0:
                    # 更新移动止损为过去N周期最低点
                    new_stop = df['low'].iloc[i-params['trail_period']:i].min()
                    trail_stop = max(trail_stop, new_stop) if trail_stop != 0 else new_stop
                    if cur['low'] <= trail_stop:
                        exit_price = trail_stop
                        exit_signal = True
                elif params['enable_short'] and prev['short_signal'] == 1:
                    exit_signal = True
            else:  # 空头
                if cur['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_signal = True
                elif params['tp_ratio'] > 0 and cur['low'] <= take_profit:
                    exit_price = take_profit
                    exit_signal = True
                elif params['use_trailing'] and params['trail_period'] > 0:
                    new_stop = df['high'].iloc[i-params['trail_period']:i].max()
                    trail_stop = min(trail_stop, new_stop) if trail_stop != 0 else new_stop
                    if cur['high'] >= trail_stop:
                        exit_price = trail_stop
                        exit_signal = True
                elif params['enable_short'] and prev['long_signal'] == 1:
                    exit_signal = True

            if exit_signal:
                pnl_points = exit_price - entry_price if position == 1 else entry_price - exit_price
                trades.append(pnl_points)
                equity.append(equity[-1] + pnl_points)
                position = 0
                continue

        # ----- 开仓检查 -----
        if position == 0:
            if prev['long_signal'] == 1:
                entry_price = cur['open']
                stop_loss = entry_price - params['stop_points']
                if params['tp_ratio'] > 0:
                    take_profit = entry_price + params['stop_points'] * params['tp_ratio']
                else:
                    take_profit = 0
                if params['use_trailing']:
                    trail_stop = stop_loss
                position = 1
                equity.append(equity[-1])
            elif params['enable_short'] and prev['short_signal'] == 1:
                entry_price = cur['open']
                stop_loss = entry_price + params['stop_points']
                if params['tp_ratio'] > 0:
                    take_profit = entry_price - params['stop_points'] * params['tp_ratio']
                else:
                    take_profit = 0
                if params['use_trailing']:
                    trail_stop = stop_loss
                position = -1
                equity.append(equity[-1])
            else:
                equity.append(equity[-1])
        else:
            equity.append(equity[-1])

    # 强制平仓
    if position != 0:
        final_price = df.iloc[-1]['close']
        if position == 1:
            pnl_points = final_price - entry_price
        else:
            pnl_points = entry_price - final_price
        trades.append(pnl_points)
        equity[-1] += pnl_points

    # 计算绩效指标
    trades_arr = np.array(trades)
    if len(trades_arr) == 0:
        return None
    win = trades_arr[trades_arr > 0]
    loss = trades_arr[trades_arr < 0]
    win_rate = len(win) / len(trades_arr)
    avg_win = win.mean() if len(win) > 0 else 0
    avg_loss = -loss.mean() if len(loss) > 0 else 0
    profit_factor = win.sum() / (-loss.sum()) if len(loss) > 0 else np.inf
    total_return = equity[-1] - initial_capital
    # 最大回撤
    eq_series = pd.Series(equity)
    peak = eq_series.expanding().max()
    drawdown = (peak - eq_series) / peak
    max_dd = drawdown.max()
    # 夏普比率（基于每根K线收益率，5分钟数据年化）
    returns = pd.Series(equity).pct_change().dropna()
    if returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24 * 12)  # 5分钟K线年化
    else:
        sharpe = 0

    return {
        'trades': trades,
        'equity': equity,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe
    }

# ====== 自动优化 ======
def optimize_params(df_train, objective='profit_factor'):
    # 扩大搜索范围，使止损更大，止盈倍数更高
    param_grid = {
        'stop_points': np.arange(10, 41, 5),        # 10,15,20,25,30,35,40
        'tp_ratio': np.arange(1.5, 4.0, 0.5),       # 1.5,2.0,2.5,3.0,3.5
        'use_trailing': [False, True],
        'trail_period': [5, 10, 15, 20],
        'enable_short': [True, False]
    }
    best_score = -np.inf
    best_params = None
    total = (len(param_grid['stop_points']) * len(param_grid['tp_ratio']) *
             2 * len(param_grid['trail_period']) * 2)
    prog = st.progress(0)
    status = st.empty()
    count = 0

    for stop in param_grid['stop_points']:
        for tp in param_grid['tp_ratio']:
            for trail in param_grid['use_trailing']:
                for tper in param_grid['trail_period']:
                    for short in param_grid['enable_short']:
                        params = {
                            'stop_points': stop,
                            'tp_ratio': tp,
                            'use_trailing': trail,
                            'trail_period': tper if trail else 0,
                            'enable_short': short
                        }
                        res = backtest(df_train, params)
                        count += 1
                        prog.progress(count / total)
                        if res and res['num_trades'] > 30:
                            # 增加过滤条件：最大回撤不能超过40%，否则跳过
                            if res['max_drawdown'] > 0.4:
                                continue
                            if objective == 'profit_factor':
                                score = res['profit_factor']
                            elif objective == 'sharpe':
                                score = res['sharpe']
                            else:
                                score = res['total_return']
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
    prog.empty()
    status.empty()
    return best_params, best_score

# ====== 执行 ======
if mode == "手动调参":
    params = {
        'stop_points': stop_points,
        'tp_ratio': tp_ratio,
        'use_trailing': use_trailing,
        'trail_period': trail_period if use_trailing else 0,
        'enable_short': enable_short
    }
    res_test = backtest(df_test, params)
    st.header("📊 测试集结果")
    if res_test:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("交易次数", res_test['num_trades'])
        col2.metric("胜率", f"{res_test['win_rate']*100:.2f}%")
        col3.metric("获利因子", f"{res_test['profit_factor']:.2f}")
        col4.metric("夏普", f"{res_test['sharpe']:.2f}")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("总收益(点)", f"{res_test['total_return']:.2f}")
        col6.metric("最大回撤", f"{res_test['max_drawdown']*100:.2f}%")
        col7.metric("平均盈利", f"{res_test['avg_win']:.2f}")
        col8.metric("平均亏损", f"{res_test['avg_loss']:.2f}")

        # 绘制权益与回撤
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05)
        eq = pd.Series(res_test['equity'])
        fig.add_trace(go.Scatter(y=eq, mode='lines', name='权益'), row=1, col=1)
        peak = eq.expanding().max()
        dd = (peak - eq) / peak
        fig.add_trace(go.Scatter(y=dd, fill='tozeroy', name='回撤'), row=2, col=1)
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("最近10笔交易")
        trades_df = pd.DataFrame({
            "序号": range(1, len(res_test['trades'])+1),
            "盈亏(点)": res_test['trades']
        }).tail(10)
        st.dataframe(trades_df)
    else:
        st.warning("无交易，请调整参数")

else:  # 自动优化
    st.header("🔍 自动参数优化")
    obj = st.sidebar.selectbox("优化目标", ["profit_factor", "sharpe", "total_return"])
    if st.sidebar.button("开始优化"):
        best_params, best_score = optimize_params(df_train, obj)
        st.success(f"最优参数 ({obj}={best_score:.4f})")
        st.json(best_params)

        res_test = backtest(df_test, best_params)
        if res_test:
            st.subheader("📈 测试集表现")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("交易次数", res_test['num_trades'])
            col2.metric("胜率", f"{res_test['win_rate']*100:.2f}%")
            col3.metric("获利因子", f"{res_test['profit_factor']:.2f}")
            col4.metric("夏普", f"{res_test['sharpe']:.2f}")
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("总收益(点)", f"{res_test['total_return']:.2f}")
            col6.metric("最大回撤", f"{res_test['max_drawdown']*100:.2f}%")
            col7.metric("平均盈利", f"{res_test['avg_win']:.2f}")
            col8.metric("平均亏损", f"{res_test['avg_loss']:.2f}")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3], vertical_spacing=0.05)
            eq = pd.Series(res_test['equity'])
            fig.add_trace(go.Scatter(y=eq, mode='lines', name='权益'), row=1, col=1)
            peak = eq.expanding().max()
            dd = (peak - eq) / peak
            fig.add_trace(go.Scatter(y=dd, fill='tozeroy', name='回撤'), row=2, col=1)
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
