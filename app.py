# -*- coding: utf-8 -*-
"""
纯K线形态策略回测（看涨/看跌吞没）
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 纯K线形态策略回测（吞没形态）")

# ====== 上传数据 ======
file = st.file_uploader("上传CSV文件（需包含 open, high, low, close, datetime）", type=["csv"])
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
    return df

df = load_data(file)
st.success(f"数据加载成功，共 {len(df)} 根K线")

# ====== 特征工程：识别吞没形态 ======
def identify_engulfing(df):
    """识别看涨吞没和看跌吞没"""
    df = df.copy()

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

    # 看跌吞没条件（当前阴线完全覆盖前一根阳线）
    bear_condition = (
        (df['prev_close'] > df['prev_open']) &          # 前一根阳线
        (df['close'] < df['open']) &                    # 当前阴线
        (df['open'] > df['prev_close']) &                # 当前开盘高于前收盘
        (df['close'] < df['prev_open'])                  # 当前收盘低于前开盘
    )
    df['bear_engulf'] = bear_condition.astype(int)

    # 去掉前几行NaN
    df.dropna(inplace=True)
    return df

df = identify_engulfing(df)
st.success("吞没形态识别完成")

# ====== 回测函数 ======
def backtest_engulfing(df):
    """
    回测逻辑：
    - 做多信号：前一根K线出现看涨吞没，且当前无持仓，则在本根K线开盘买入
    - 止损价 = 信号K线的最低价（即前一根K线的最低价）
    - 平仓条件：
        1. 价格跌破止损价：若某根K线收盘价 < 止损价，则下一根开盘平仓
        2. 出现反向信号：若前一根K线出现看跌吞没，则本根开盘平仓
    """
    equity = [0]          # 权益曲线
    trades = []           # 每笔交易的盈亏
    position = 0          # 0-空仓，1-持有多头
    entry_price = 0.0     # 入场价
    stop_loss = 0.0       # 止损价
    entry_idx = -1        # 入场时的索引（用于定位信号K线）

    for i in range(1, len(df)):
        # 当前K线数据（用于确定入场/出场价）
        current = df.iloc[i]
        # 前一根K线数据（用于判断信号）
        prev = df.iloc[i-1]

        # ----- 平仓检查 -----
        if position == 1:
            # 条件1：止损触发（前一根K线收盘价 < 止损价）
            stop_triggered = (prev['close'] < stop_loss)
            # 条件2：反向信号（前一根K线出现看跌吞没）
            reverse_signal = (prev['bear_engulf'] == 1)

            if stop_triggered or reverse_signal:
                # 在本根K线开盘价平仓
                exit_price = current['open']
                pnl = exit_price - entry_price
                trades.append(pnl)
                equity.append(equity[-1] + pnl)
                position = 0
                # 重置止损
                stop_loss = 0.0
                entry_idx = -1
                # 注意：平仓后，本根K线不再开仓（避免同一根K线先平后开）
                # 但我们可以允许开仓（基于前一根信号），因为本根开盘已平仓，之后还可以开新仓？如果前一根有开仓信号，理论上可以在本根开盘平仓后立即开新仓，但为了简化，我们不在同一根K线同时操作。这里保持简单：先平后不立即开，下一根再处理。所以跳过本次开仓部分。
                continue

        # ----- 开仓检查（基于前一根K线的信号） -----
        if position == 0:
            if prev['bull_engulf'] == 1:
                # 入场价 = 当前K线开盘价
                entry_price = current['open']
                # 止损价 = 信号K线（即前一根K线）的最低价
                stop_loss = prev['low']
                position = 1
                entry_idx = i   # 记录入场时的索引（暂未使用）
                # 开仓时不立即计入权益，持仓期间权益不变
                equity.append(equity[-1])
            else:
                # 无信号，权益不变
                equity.append(equity[-1])
        else:
            # 已有持仓且未平仓，权益不变
            equity.append(equity[-1])

    # 最后如果还有持仓，以最后收盘价平仓（强制平仓）
    if position == 1:
        final_price = df.iloc[-1]['close']
        pnl = final_price - entry_price
        trades.append(pnl)
        equity[-1] += pnl   # 更新最后权益

    return {
        "交易次数": len(trades),
        "胜率": sum(1 for p in trades if p > 0) / len(trades) if trades else 0,
        "盈利": sum(trades),
        "equity": equity
    }

# ====== 运行回测（使用全部数据，不分割） ======
with st.spinner("回测进行中..."):
    result = backtest_engulfing(df)

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
    st.warning("没有产生任何交易，请检查数据或调整形态定义")

# ====== 显示最近几笔交易示例 ======
if result["交易次数"] > 0:
    st.subheader("交易明细（最近5笔）")
    # 简单显示最后5笔交易的盈亏
    last_trades = pd.DataFrame({
        "交易序号": range(len(result["equity"])-len(result["trades"]), len(result["equity"])),
        "盈亏": result["trades"][-5:]
    })
    st.dataframe(last_trades)
