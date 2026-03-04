# -*- coding: utf-8 -*-
"""
K线吞没 + 成交量 + 均线 + 可调止损止盈 + 双向交易
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 增强版多空策略（可调参数）")

# ====== 侧边栏参数 ======
st.sidebar.header("策略参数")
ma_period = st.sidebar.slider("均线周期", 5, 100, 20)
vol_ma_period = st.sidebar.slider("成交量均线周期", 5, 100, 20)
atr_period = st.sidebar.slider("ATR周期", 5, 50, 14)
stop_mult = st.sidebar.slider("止损倍数 (ATR倍数)", 1.0, 5.0, 2.0, 0.1)
take_profit_mult = st.sidebar.slider("止盈倍数 (相对于止损)", 1.0, 5.0, 2.0, 0.1)
enable_short = st.sidebar.checkbox("启用做空", value=True)

# ====== 上传数据 ======
file = st.file_uploader("上传CSV文件（需包含 open, high, low, close, volume, datetime）", type=["csv"])
if file is None:
    st.info("请上传数据文件")
    st.stop()

# ====== 加载数据 ======
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            st.error(f"缺少必要字段: {col}")
            st.stop()

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

df = load_data(file)
st.success(f"数据加载成功，共 {len(df)} 根K线")

# ====== 特征工程 ======
def prepare_features(df, ma_period, vol_ma_period, atr_period):
    df = df.copy()

    # 均线
    df['ma'] = df['close'].rolling(ma_period).mean()
    df['vol_ma'] = df['volume'].rolling(vol_ma_period).mean()

    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(atr_period).mean()

    # 前一根K线数据
    df['prev_open'] = df['open'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    # 吞没形态
    bull_engulf = (
        (df['prev_close'] < df['prev_open']) &
        (df['close'] > df['open']) &
        (df['open'] < df['prev_close']) &
        (df['close'] > df['prev_open'])
    )
    bear_engulf = (
        (df['prev_close'] > df['prev_open']) &
        (df['close'] < df['open']) &
        (df['open'] > df['prev_close']) &
        (df['close'] < df['prev_open'])
    )

    # 过滤条件：价格相对均线、成交量放大
    price_above_ma = (df['close'] > df['ma'])
    price_below_ma = (df['close'] < df['ma'])
    vol_surge = (df['volume'] > df['vol_ma'])

    # 多空信号（基于当前K线）
    df['long_signal_raw'] = (bull_engulf & price_above_ma & vol_surge).astype(int)
    df['short_signal_raw'] = (bear_engulf & price_below_ma & vol_surge).astype(int)

    # 去掉NaN行
    df.dropna(inplace=True)
    return df

df = prepare_features(df, ma_period, vol_ma_period, atr_period)
st.success("特征工程完成")

# ====== 回测函数 ======
def backtest(df, stop_mult, take_profit_mult, enable_short):
    equity = [0]
    trades = []
    position = 0  # 1多 -1空
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ----- 平仓检查 -----
        if position != 0:
            # 对于多头
            if position == 1:
                # 止损触发
                stop_triggered = (prev['close'] < stop_loss)
                # 止盈触发
                profit_triggered = (prev['close'] > take_profit) if take_profit else False
                # 反向信号平仓（可选）
                reverse_signal = (prev['short_signal_raw'] == 1) if enable_short else False

                if stop_triggered or profit_triggered or reverse_signal:
                    exit_price = current['open']
                    pnl = exit_price - entry_price
                    trades.append(pnl)
                    equity.append(equity[-1] + pnl)
                    position = 0
                    continue

            # 对于空头
            elif position == -1:
                stop_triggered = (prev['close'] > stop_loss)  # 空头止损是价格向上突破
                profit_triggered = (prev['close'] < take_profit) if take_profit else False
                reverse_signal = (prev['long_signal_raw'] == 1)

                if stop_triggered or profit_triggered or reverse_signal:
                    exit_price = current['open']
                    pnl = entry_price - exit_price  # 空头盈利 = 入场价 - 出场价
                    trades.append(pnl)
                    equity.append(equity[-1] + pnl)
                    position = 0
                    continue

        # ----- 开仓检查 -----
        if position == 0:
            # 做多
            if prev['long_signal_raw'] == 1:
                entry_price = current['open']
                atr = prev['atr']
                stop_loss = entry_price - stop_mult * atr
                take_profit = entry_price + take_profit_mult * (entry_price - stop_loss) if take_profit_mult > 0 else 0
                position = 1
                equity.append(equity[-1])
            # 做空（如果启用）
            elif enable_short and prev['short_signal_raw'] == 1:
                entry_price = current['open']
                atr = prev['atr']
                stop_loss = entry_price + stop_mult * atr
                take_profit = entry_price - take_profit_mult * (stop_loss - entry_price) if take_profit_mult > 0 else 0
                position = -1
                equity.append(equity[-1])
            else:
                equity.append(equity[-1])
        else:
            equity.append(equity[-1])

    # 强制平仓
    if position == 1:
        final_price = df.iloc[-1]['close']
        pnl = final_price - entry_price
        trades.append(pnl)
        equity[-1] += pnl
    elif position == -1:
        final_price = df.iloc[-1]['close']
        pnl = entry_price - final_price
        trades.append(pnl)
        equity[-1] += pnl

    return {
        "交易次数": len(trades),
        "胜率": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "盈利": sum(trades),
        "equity": equity,
        "trades": trades
    }

# ====== 运行回测 ======
with st.spinner("回测进行中..."):
    result = backtest(df, stop_mult, take_profit_mult, enable_short)

# ====== 展示结果 ======
st.header("📊 回测结果")
col1, col2, col3 = st.columns(3)
col1.metric("交易次数", result["交易次数"])
col2.metric("胜率", f"{result['胜率']*100:.2f}%")
col3.metric("总盈利", f"{result['盈利']:.2f}")

if result["交易次数"] > 0:
    st.subheader("权益曲线")
    st.line_chart(pd.Series(result["equity"]))

    # 附加统计
    trades_arr = np.array(result["trades"])
    win_rate = result["胜率"]
    avg_win = trades_arr[trades_arr > 0].mean() if np.any(trades_arr > 0) else 0
    avg_loss = abs(trades_arr[trades_arr < 0].mean()) if np.any(trades_arr < 0) else 0
    profit_factor = (trades_arr[trades_arr > 0].sum()) / abs(trades_arr[trades_arr < 0].sum()) if np.any(trades_arr < 0) else np.inf

    st.subheader("详细统计")
    col4, col5, col6 = st.columns(3)
    col4.metric("平均盈利", f"{avg_win:.2f}")
    col5.metric("平均亏损", f"{avg_loss:.2f}")
    col6.metric("盈亏比", f"{(avg_win/avg_loss if avg_loss else 0):.2f}")
    st.metric("获利因子", f"{profit_factor:.2f}")

    # 交易明细
    st.subheader("交易明细（最近5笔）")
    last_n = 5
    trades_list = result["trades"][-last_n:]
    start_idx = max(1, len(result["trades"]) - last_n + 1)
    trade_indices = list(range(start_idx, len(result["trades"]) + 1))
    last_trades = pd.DataFrame({
        "交易序号": trade_indices,
        "盈亏": trades_list
    })
    st.dataframe(last_trades)
else:
    st.warning("没有产生任何交易，请调整参数")
