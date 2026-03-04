# -*- coding: utf-8 -*-
"""
终极优化策略：K线吞没形态 + 成交量确认 + 均线趋势过滤 + 自动参数优化 + 样本内外测试
功能：自动寻参、ATR止损止盈、移动止损、双向交易、资金管理、详细绩效报告
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import brute
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("🚀 终极优化策略（自动寻参 + 样本内外验证）")

# ====== 侧边栏 ======
st.sidebar.header("运行设置")
mode = st.sidebar.radio("模式", ["手动调参", "自动优化"])

if mode == "手动调参":
    ma_period = st.sidebar.slider("均线周期", 5, 100, 20)
    vol_ma_period = st.sidebar.slider("成交量均线周期", 5, 100, 20)
    atr_period = st.sidebar.slider("ATR周期", 5, 50, 14)
    stop_mult = st.sidebar.slider("止损倍数 (ATR)", 1.0, 5.0, 2.0, 0.1)
    tp_mult = st.sidebar.slider("止盈倍数 (0=不用)", 0.0, 5.0, 2.0, 0.1)
    use_trailing = st.sidebar.checkbox("移动止损", value=False)
    trail_period = st.sidebar.slider("移动止损周期", 5, 50, 10) if use_trailing else 0
    enable_short = st.sidebar.checkbox("启用做空", value=True)

# ====== 上传数据 ======
file = st.file_uploader("上传CSV (需包含 open,high,low,close,volume,datetime)", type=["csv"])
if not file:
    st.info("请上传数据文件")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    # 尝试识别时间戳列
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
    # 成交量字段兼容
    if 'volume' in df.columns:
        df['volume'] = df['volume']
    elif 'vol' in df.columns:
        df['volume'] = df['vol']
    elif 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    else:
        st.error("缺少成交量字段")
        st.stop()
    return df

df_full = load_data(file)
st.success(f"数据共 {len(df_full)} 根K线")

# 分割训练/测试
split = int(len(df_full) * 0.6)
df_train = df_full.iloc[:split].copy()
df_test = df_full.iloc[split:].copy()
st.info(f"训练集 {len(df_train)} | 测试集 {len(df_test)}")

# ====== 特征工程 ======
def prepare_features(df, ma_p, vol_ma_p, atr_p):
    df = df.copy()
    df['ma'] = df['close'].rolling(ma_p).mean()
    df['vol_ma'] = df['volume'].rolling(vol_ma_p).mean()
    df['tr'] = np.maximum(df['high']-df['low'],
                          np.maximum(abs(df['high']-df['close'].shift(1)),
                                     abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(atr_p).mean()
    # 前一根数据
    for col in ['open','high','low','close']:
        df[f'prev_{col}'] = df[col].shift(1)
    # 吞没形态
    bull = (df['prev_close'] < df['prev_open']) & (df['close'] > df['open']) & \
           (df['open'] < df['prev_close']) & (df['close'] > df['prev_open'])
    bear = (df['prev_close'] > df['prev_open']) & (df['close'] < df['open']) & \
           (df['open'] > df['prev_close']) & (df['close'] < df['prev_open'])
    # 过滤
    price_above_ma = df['close'] > df['ma']
    price_below_ma = df['close'] < df['ma']
    vol_surge = df['volume'] > df['vol_ma']
    df['long_signal'] = (bull & price_above_ma & vol_surge).astype(int)
    df['short_signal'] = (bear & price_below_ma & vol_surge).astype(int)
    df.dropna(inplace=True)
    return df

# ====== 回测核心 ======
def backtest(df, params):
    """
    params: dict 包含 ma_period, vol_ma_period, atr_period,
                 stop_mult, tp_mult, use_trailing, trail_period, enable_short
    返回权益曲线(点数)和交易列表
    """
    df = prepare_features(df, params['ma_period'], params['vol_ma_period'], params['atr_period'])
    if len(df) == 0:
        return None

    equity = [0]           # 点数权益（初始0）
    trades = []            # 每笔盈亏点数
    position = 0           # 1多 -1空
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trail_stop = 0.0       # 移动止损价

    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i-1]

        # ----- 平仓检查 -----
        if position != 0:
            exit_signal = False
            exit_price = cur['open']      # 默认下根开盘平
            if position == 1:  # 多头
                # 止损
                if cur['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_signal = True
                # 止盈
                elif params['tp_mult'] > 0 and cur['high'] >= take_profit:
                    exit_price = take_profit
                    exit_signal = True
                # 移动止损
                elif params['use_trailing'] and params['trail_period'] > 0:
                    # 更新移动止损为过去N周期最低点
                    new_stop = df['close'].iloc[i-params['trail_period']:i].min()
                    trail_stop = max(trail_stop, new_stop) if trail_stop != 0 else new_stop
                    if cur['low'] <= trail_stop:
                        exit_price = trail_stop
                        exit_signal = True
                # 反向信号平仓
                elif params['enable_short'] and prev['short_signal'] == 1:
                    exit_signal = True
            else:  # 空头
                if cur['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_signal = True
                elif params['tp_mult'] > 0 and cur['low'] <= take_profit:
                    exit_price = take_profit
                    exit_signal = True
                elif params['use_trailing'] and params['trail_period'] > 0:
                    new_stop = df['close'].iloc[i-params['trail_period']:i].max()
                    trail_stop = min(trail_stop, new_stop) if trail_stop != 0 else new_stop
                    if cur['high'] >= trail_stop:
                        exit_price = trail_stop
                        exit_signal = True
                elif params['enable_short'] and prev['long_signal'] == 1:
                    exit_signal = True

            if exit_signal:
                pnl = exit_price - entry_price if position == 1 else entry_price - exit_price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0
                continue

        # ----- 开仓检查 -----
        if position == 0:
            if prev['long_signal'] == 1:
                entry_price = cur['open']
                atr = prev['atr']
                stop_loss = entry_price - params['stop_mult'] * atr
                if params['tp_mult'] > 0:
                    take_profit = entry_price + params['tp_mult'] * atr
                else:
                    take_profit = 0
                if params['use_trailing']:
                    trail_stop = stop_loss
                position = 1
                equity.append(equity[-1])
            elif params['enable_short'] and prev['short_signal'] == 1:
                entry_price = cur['open']
                atr = prev['atr']
                stop_loss = entry_price + params['stop_mult'] * atr
                if params['tp_mult'] > 0:
                    take_profit = entry_price - params['tp_mult'] * atr
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
            pnl = final_price - entry_price
        else:
            pnl = entry_price - final_price
        trades.append(pnl)
        equity[-1] += pnl

    # 计算指标
    trades_arr = np.array(trades)
    if len(trades_arr) == 0:
        return None
    win = trades_arr[trades_arr > 0]
    loss = trades_arr[trades_arr < 0]
    win_rate = len(win) / len(trades_arr)
    avg_win = win.mean() if len(win) > 0 else 0
    avg_loss = -loss.mean() if len(loss) > 0 else 0
    profit_factor = win.sum() / (-loss.sum()) if len(loss) > 0 else np.inf
    total_return = equity[-1]
    # 最大回撤
    eq_series = pd.Series(equity)
    drawdown = (eq_series.cummax() - eq_series) / (eq_series.cummax() + 1e-9)
    max_dd = drawdown.max()
    # 夏普（简化，用点数日收益率，假设每日一根K线，这里是5分钟，调整252为每年交易天数）
    ret_daily = pd.Series(equity).diff().dropna()
    sharpe = ret_daily.mean() / ret_daily.std() * np.sqrt(252 * 24 * 12) if ret_daily.std() != 0 else 0  # 5分钟数据年化

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
    # 搜索范围（可自行调整）
    param_grid = {
        'ma_period': range(10, 51, 5),
        'vol_ma_period': range(10, 51, 5),
        'atr_period': range(7, 22, 2),
        'stop_mult': np.arange(1.0, 4.0, 0.5),
        'tp_mult': np.arange(1.0, 4.0, 0.5),
        'use_trailing': [False, True],
        'trail_period': [5, 10, 15],
        'enable_short': [True, False]
    }
    best_score = -np.inf
    best_params = None
    total = (len(param_grid['ma_period']) * len(param_grid['vol_ma_period']) *
             len(param_grid['atr_period']) * len(param_grid['stop_mult']) *
             len(param_grid['tp_mult']) * 2 * 3 * 2)
    prog = st.progress(0)
    status = st.empty()
    count = 0

    for ma in param_grid['ma_period']:
        for vol in param_grid['vol_ma_period']:
            for atr in param_grid['atr_period']:
                for stop in param_grid['stop_mult']:
                    for tp in param_grid['tp_mult']:
                        for trail in param_grid['use_trailing']:
                            for tper in param_grid['trail_period']:
                                for short in param_grid['enable_short']:
                                    params = {
                                        'ma_period': ma,
                                        'vol_ma_period': vol,
                                        'atr_period': atr,
                                        'stop_mult': stop,
                                        'tp_mult': tp,
                                        'use_trailing': trail,
                                        'trail_period': tper if trail else 0,
                                        'enable_short': short
                                    }
                                    res = backtest(df_train, params)
                                    count += 1
                                    prog.progress(count / total)
                                    if res and res['num_trades'] > 30:
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
        'ma_period': ma_period,
        'vol_ma_period': vol_ma_period,
        'atr_period': atr_period,
        'stop_mult': stop_mult,
        'tp_mult': tp_mult,
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
        dd = (eq.cummax() - eq) / eq.cummax()
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
            dd = (eq.cummax() - eq) / eq.cummax()
            fig.add_trace(go.Scatter(y=dd, fill='tozeroy', name='回撤'), row=2, col=1)
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
