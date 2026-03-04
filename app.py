# -*- coding: utf-8 -*-
"""
纯K 5分钟 多周期强化版
包含：
滑点 + 手续费 + ATR止损 + ADX过滤 + 实时刷新
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")
st.title("📊 5分钟纯K专业强化版")

# =========================
# 参数
# =========================

st.sidebar.header("策略参数")

body_threshold = st.sidebar.slider("实体比例", 0.05, 0.5, 0.15, 0.01)
vol_period = st.sidebar.slider("成交量周期", 5, 30, 15)
break_threshold = st.sidebar.slider("突破幅度", 0.0005, 0.003, 0.001, 0.0001)
atr_period = st.sidebar.slider("ATR周期", 5, 30, 14)
atr_mult = st.sidebar.slider("ATR倍数", 0.5, 5.0, 1.5, 0.1)
adx_threshold = st.sidebar.slider("ADX趋势过滤", 10, 40, 20)
rr = st.sidebar.slider("盈亏比", 1.0, 5.0, 2.0, 0.1)   # 注意：现在改用分批止盈，此参数不再直接用于单一目标，但保留作为参考

slippage_pct = st.sidebar.slider("滑点%", 0.0, 0.002, 0.0005, 0.0001)
fee_pct = st.sidebar.slider("手续费%", 0.0, 0.002, 0.0004, 0.0001)

risk_pct = st.sidebar.slider("单笔风险%", 0.5, 5.0, 1.0, 0.1) / 100

auto_refresh = st.sidebar.checkbox("自动刷新")
refresh_sec = st.sidebar.slider("刷新秒数", 5, 60, 15)

lookback = 20
initial_equity = 100

# =========================
# 数据加载
# =========================

file = st.file_uploader("上传5分钟CSV", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.lower() for c in df.columns]
df['time'] = pd.to_datetime(df['time'])
df.sort_values("time", inplace=True)

# =========================
# 计算指标
# =========================

def add_indicators(data):
    data['high_max'] = data['high'].rolling(lookback).max().shift(1)
    data['low_min'] = data['low'].rolling(lookback).min().shift(1)
    data['body'] = abs(data['close'] - data['open'])
    data['range'] = data['high'] - data['low']
    data['body_ratio'] = data['body'] / (data['range'] + 1e-9)
    data['vol_ma'] = data['vol'].rolling(vol_period).mean()

    # ATR
    tr = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr'] = tr.rolling(atr_period).mean()

    # ADX
    up = data['high'].diff()
    down = -data['low'].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    tr_smooth = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / tr_smooth

    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    data['adx'] = dx.rolling(14).mean()

    return data.dropna()

df = add_indicators(df)

# 30分钟确认
df_30 = df.set_index('time').resample('30T').agg({
    'open':'first','high':'max','low':'min','close':'last','vol':'sum'
}).dropna().reset_index()

df_30 = add_indicators(df_30)

# =========================
# 回测逻辑（分批止盈）
# =========================

equity = initial_equity
equity_curve = []
records = []
signals = []

position = 0                # 0-无持仓，1-多头，-1-空头
positions_list = []         # 存储当前持仓的子仓位信息，每个元素为字典

for i in range(1, len(df)):

    row = df.iloc[i]
    row30 = df_30[df_30['time'] <= row['time']].iloc[-1]

    signal = 0

    if row['adx'] > adx_threshold:

        if row['close'] > row['high_max'] and row['body_ratio'] > body_threshold:
            if row30['close'] > row30['high_max']:
                signal = 1

        if row['close'] < row['low_min'] and row['body_ratio'] > body_threshold:
            if row30['close'] < row30['low_min']:
                signal = -1

    # 开仓条件：无持仓且有信号
    if position == 0 and signal != 0:

        entry_price = row['open'] * (1 + slippage_pct * signal)
        atr = row['atr']
        stop = entry_price - atr * atr_mult if signal == 1 else entry_price + atr * atr_mult
        risk = abs(entry_price - stop)
        size = equity * risk_pct / risk

        # 三个目标价（1倍、2倍、3倍风险）
        if signal == 1:  # 多头
            tp1 = entry_price + risk * 1
            tp2 = entry_price + risk * 2
            tp3 = entry_price + risk * 3
        else:            # 空头
            tp1 = entry_price - risk * 1
            tp2 = entry_price - risk * 2
            tp3 = entry_price - risk * 3

        size_sub = size / 3.0

        # 创建子仓位记录（所有子仓位共享止损，但目标不同）
        positions_list = [
            {
                'side': signal,
                'entry': entry_price,
                'stop': stop,
                'tps': [tp1, tp2, tp3],
                'filled': [False, False, False],
                'size': size_sub
            }
        ]

        # 记录开仓信号（用于图表标注）
        signals.append((row['time'], signal, entry_price, stop, tp1))

        position = signal

    # ----------------- 持仓管理（分批止盈） -----------------
    if positions_list:
        pos = positions_list[0]      # 当前只有一组子仓位
        side = pos['side']
        entry = pos['entry']
        stop = pos['stop']
        tps = pos['tps']
        filled = pos['filled']
        size_sub = pos['size']

        # 1. 检查止损（所有剩余子仓位同时止损）
        if side == 1:  # 多头
            if row['low'] <= stop:
                for j in range(3):
                    if not filled[j]:
                        exit_price = stop * (1 - slippage_pct)
                        pnl = (exit_price - entry) * size_sub
                        pnl -= equity * fee_pct
                        equity += pnl
                        records.append({"side": "long", "pnl": pnl, "target": f"stop_{j+1}"})
                        filled[j] = True
                positions_list = []
                position = 0
        else:           # 空头
            if row['high'] >= stop:
                for j in range(3):
                    if not filled[j]:
                        exit_price = stop * (1 + slippage_pct)
                        pnl = (entry - exit_price) * size_sub
                        pnl -= equity * fee_pct
                        equity += pnl
                        records.append({"side": "short", "pnl": pnl, "target": f"stop_{j+1}"})
                        filled[j] = True
                positions_list = []
                position = 0

        # 2. 如果止损未触发，检查止盈（逐个目标）
        if positions_list:   # 仍有未平仓位
            for j in range(3):
                if not filled[j]:
                    tp = tps[j]
                    if side == 1:  # 多头
                        if row['high'] >= tp:
                            exit_price = tp * (1 - slippage_pct)
                            pnl = (exit_price - entry) * size_sub
                            pnl -= equity * fee_pct
                            equity += pnl
                            records.append({"side": "long", "pnl": pnl, "target": f"tp{j+1}"})
                            filled[j] = True
                    else:           # 空头
                        if row['low'] <= tp:
                            exit_price = tp * (1 + slippage_pct)
                            pnl = (entry - exit_price) * size_sub
                            pnl -= equity * fee_pct
                            equity += pnl
                            records.append({"side": "short", "pnl": pnl, "target": f"tp{j+1}"})
                            filled[j] = True

            # 若所有子仓位均已平仓，清空持仓
            if all(filled):
                positions_list = []
                position = 0

    # 记录资金曲线
    equity_curve.append(equity)

df['equity'] = [np.nan] + equity_curve

# =========================
# 统计
# =========================

st.header("📌 统计")

if len(records) > 0:
    rec = pd.DataFrame(records).tail(100)
    long_win = rec[rec['side']=="long"]['pnl']>0
    short_win = rec[rec['side']=="short"]['pnl']>0

    st.metric("当前资金", f"{equity:.2f}")
    st.metric("最近100胜率", f"{(rec['pnl']>0).mean()*100:.2f}%")
    st.metric("多头胜率", f"{long_win.mean()*100:.2f}%" if len(long_win)>0 else "N/A")
    st.metric("空头胜率", f"{short_win.mean()*100:.2f}%" if len(short_win)>0 else "N/A")

# =========================
# K线
# =========================

st.header("📈 K线")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
))

for t, side, entry, stop, tp1 in signals:
    fig.add_annotation(x=t, y=entry,
        text="▲" if side==1 else "▼",
        showarrow=True)

fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# =========================
# 资金曲线
# =========================

st.header("💰 资金曲线")
st.line_chart(df['equity'])

# =========================
# 自动刷新
# =========================

if auto_refresh:
    time.sleep(refresh_sec)
    st.experimental_rerun()
