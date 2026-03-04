# -*- coding: utf-8 -*-
"""
K线吞没形态 + 成交量 + 均线 多空策略（修复版）
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 K线吞没 + 成交量 + 均线 策略")

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

    # 处理时间索引
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    # 确保包含必要字段
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            st.error(f"缺少必要字段: {col}")
            st.stop()

    # 成交量字段兼容
    if 'volume' in df.columns:
        df['volume'] = df['volume']
    elif 'vol' in df.columns:
        df['volume'] = df['vol']
    elif 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    else:
        st.error("缺少成交量字段：volume / vol / tick_volume")
        st.stop()

    return df

df = load_data(file)
st.success(f"数据加载成功，共 {len(df)} 根K线")

# ====== 特征工程：识别吞没形态 + 成交量 + 均线 ======
def prepare_features(df):
    df = df.copy()

    # 计算20日均线和20日均量
    df['ma20'] = df['close'].rolling(20).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()

    # 前一根K线数据
    df['prev_open'] = df['open'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    # 看涨吞没条件（当前阳线完全覆盖前一根阴线）
    bull_condition = (
        (df['prev_close'] < df['prev_open']) &          # 前一根阴线
        (df['close'] > df['open']) &                    # 当前阳线
        (df['open'] < df['prev_close']) &                # 当前开盘低于前收盘
        (df['close'] > df['prev_open'])                  # 当前收盘高于前开盘
    )
    df['bull_engulf'] = bull_condition.astype(int)

    # 看跌吞没条件（当前阴线完全覆盖前一根阳线）——用于反向平仓
    bear_condition = (
        (df['prev_close'] > df['prev_open']) &          # 前一根阳线
        (df['close'] < df['open']) &                    # 当前阴线
        (df['open'] > df['prev_close']) &                # 当前开盘高于前收盘
        (df['close'] < df['prev_open'])                  # 当前收盘低于前开盘
    )
    df['bear_engulf'] = bear_condition.astype(int)

    # 附加过滤条件：价格在均线上方 & 成交量放大
    df['price_above_ma'] = (df['close'] > df['ma20']).astype(int)
    df['vol_surge'] = (df['volume'] > df['vol_ma20']).astype(int)

    # 最终做多信号：前一根K线满足看涨吞没 + 价格在均线上 + 放量
    # 注意：我们使用前一根K线的过滤条件，因此需要将当前K线的条件shift到下一根？
    # 在回测中，我们将基于前一根K线的最终信号入场，所以这里生成一个综合信号列（基于当前K线）
    df['signal_raw'] = (
        (df['bull_engulf'] == 1) &
        (df['price_above_ma'] == 1) &
        (df['vol_surge'] == 1)
    ).astype(int)

    # 去掉前20行NaN（均线计算导致）
    df.dropna(inplace=True)
    return df

df = prepare_features(df)
st.success("特征工程完成")

# ====== 回测函数 ======
def backtest_strategy(df):
    """
    回测逻辑：
    - 做多信号：前一根K线的 signal_raw == 1，且当前无持仓，则在本根K线开盘买入
    - 止损价 = 信号K线的最低价（即前一根K线的最低价）
    - 平仓条件：
        1. 价格跌破止损价：若某根K线收盘价 < 止损价，则下一根开盘平仓
        2. 出现反向信号：若前一根K线出现看跌吞没，则本根开盘平仓
    """
    equity = [0]
    trades = []
    position = 0
    entry_price = 0.0
    stop_loss = 0.0

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ----- 平仓检查 -----
        if position == 1:
            stop_triggered = (prev['close'] < stop_loss)
            reverse_signal = (prev['bear_engulf'] == 1)

            if stop_triggered or reverse_signal:
                exit_price = current['open']
                pnl = exit_price - entry_price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0
                continue

        # ----- 开仓检查 -----
        if position == 0:
            if prev['signal_raw'] == 1:
                entry_price = current['open']
                stop_loss = prev['low']  # 信号K线的最低价
                position = 1
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

    return {
        "交易次数": len(trades),
        "胜率": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "盈利": sum(trades),
        "equity": equity,
        "trades": trades
    }

# ====== 运行回测 ======
with st.spinner("回测进行中..."):
    result = backtest_strategy(df)

# ====== 展示结果 ======
st.header("📊 回测结果")
col1, col2, col3 = st.columns(3)
col1.metric("交易次数", result["交易次数"])
col2.metric("胜率", f"{result['胜率']*100:.2f}%")
col3.metric("总盈利", f"{result['盈利']:.2f}")

if result["交易次数"] > 0:
    st.subheader("权益曲线")
    st.line_chart(pd.Series(result["equity"]))
else:
    st.warning("没有产生任何交易，请检查数据或调整过滤条件")

# ====== 交易明细 ======
if result["交易次数"] > 0:
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
