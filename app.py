"""
ETH 战神 V2600 - 高胜率·盈亏优化版
核心策略：
  1. 15分钟EMA20方向确定主趋势
  2. 5分钟局部结构突破 (Higher Low / Lower High)
  3. 成交量确认 + 假突破过滤
  4. 小止损0.6ATR，小止盈0.8ATR (盈亏比1.33)
  5. 回测模块展示盈利表现
  6. 实时信号提示
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime
import time

st.set_page_config(layout="wide", page_title="战神 V2600")
st.title("🛡️ ETH 战神 V2600 (高胜率·盈亏优化版)")

SYMBOL = "ETH-USDT-SWAP"

# --------------------------
# 侧边栏参数
# --------------------------
with st.sidebar:
    st.header("⚙️ 盈利引擎参数")
    adx_threshold = st.slider("ADX 强度阈值", 20, 35, 25, help="低于此值不交易")
    ema_fast_period = st.slider("快线EMA周期", 5, 20, 8)
    ema_slow_period = st.slider("慢线EMA周期", 20, 60, 30)
    atr_period = st.slider("ATR周期", 7, 21, 10)
    swing_window = st.slider("结构点窗口", 2, 5, 3)
    volume_mult = st.slider("突破放量倍数", 1.0, 2.0, 1.3, step=0.1)
    atr_sl_mult = st.number_input("止损ATR倍数", value=0.6, step=0.1)
    atr_tp_mult = st.number_input("止盈ATR倍数", value=0.8, step=0.1)
    risk_percent = st.slider("单笔风险 %", 0.5, 2.0, 1.0, step=0.1)

    st.divider()
    st.info("💡 核心逻辑：15M趋势锁死 + 5M结构突破 + 量能确认 + 小止损小止盈")
    if st.button("🔄 强制刷新数据"):
        st.cache_data.clear()
        st.rerun()

# --------------------------
# 数据获取（缓存）
# --------------------------
@st.cache_data(ttl=5)
def get_klines(bar="5m", limit=500):
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={bar}&limit={limit}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json().get("data", [])
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for c in ["open", "high", "low", "close", "vol"]:
            df[c] = df[c].astype(float)
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

# --------------------------
# 指标计算
# --------------------------
def compute_indicators(df_5m):
    df = df_5m.copy()
    df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast_period)
    df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow_period)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)
    df["volume_ma"] = df["vol"].rolling(window=20).mean()
    # 趋势确认
    df["trend_up"] = (df["EMA_fast"] > df["EMA_slow"]) & (df["ADX"] > adx_threshold)
    df["trend_down"] = (df["EMA_fast"] < df["EMA_slow"]) & (df["ADX"] > adx_threshold)
    # 连续确认两期
    df["trend_up_streak"] = df["trend_up"].rolling(2).sum() == 2
    df["trend_down_streak"] = df["trend_down"].rolling(2).sum() == 2
    return df.dropna().reset_index(drop=True)

# --------------------------
# 结构点检测
# --------------------------
def find_swing_points(df, window=3):
    highs, lows = [], []
    for i in range(window, len(df)-window):
        if df['high'].iloc[i] == max(df['high'].iloc[i-window:i+window+1]):
            highs.append((df['ts'].iloc[i], df['high'].iloc[i]))
        if df['low'].iloc[i] == min(df['low'].iloc[i-window:i+window+1]):
            lows.append((df['ts'].iloc[i], df['low'].iloc[i]))
    return highs, lows

# --------------------------
# 假突破检测
# --------------------------
def is_fake_break(row):
    body = abs(row["close"] - row["open"])
    if body == 0:
        return True
    upper = row["high"] - max(row["close"], row["open"])
    lower = min(row["close"], row["open"]) - row["low"]
    return upper > body * 1.5 or lower > body * 1.5

# --------------------------
# 15分钟方向
# --------------------------
def get_15m_direction():
    df15 = get_klines("15m", 50)
    if len(df15) < 20:
        return "无"
    df15["EMA20"] = ta.trend.ema_indicator(df15["close"], window=20)
    df15 = df15.dropna().reset_index(drop=True)
    if len(df15) < 2:
        return "无"
    return "多" if df15["EMA20"].iloc[-1] > df15["EMA20"].iloc[-2] else "空"

# --------------------------
# 回测引擎（使用下一根K线开盘入场）
# --------------------------
def run_backtest(df, swing_highs, swing_lows):
    trades = []
    capital = 1000
    position = None
    equity_curve = [capital]

    for i in range(1, len(df)-1):  # 预留未来数据
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # 获取当前可用的结构点（不包含未来数据）
        valid_highs = [(t, p) for t, p in swing_highs if t <= row['ts']]
        valid_lows = [(t, p) for t, p in swing_lows if t <= row['ts']]
        last_high = valid_highs[-1] if valid_highs else None
        last_low = valid_lows[-1] if valid_lows else None
        prev_high = valid_highs[-2] if len(valid_highs) >= 2 else None
        prev_low = valid_lows[-2] if len(valid_lows) >= 2 else None

        bull_structure = prev_low and last_low and last_low[1] > prev_low[1]
        bear_structure = prev_high and last_high and last_high[1] < prev_high[1]

        tf15_dir = get_15m_direction()

        # 信号判断（基于当前K线收盘）
        signal = None
        if (row["trend_up_streak"] and tf15_dir == "多" and bull_structure and not is_fake_break(row)):
            if last_high and row["close"] > last_high[1]:
                body = abs(row["close"] - row["open"])
                if body > (row["high"] - row["low"]) * 0.6:
                    if (row["close"] - last_high[1]) > row["ATR"] * 0.2:
                        if row["vol"] > row["volume_ma"] * volume_mult:
                            signal = "多"
        elif (row["trend_down_streak"] and tf15_dir == "空" and bear_structure and not is_fake_break(row)):
            if last_low and row["close"] < last_low[1]:
                body = abs(row["close"] - row["open"])
                if body > (row["high"] - row["low"]) * 0.6:
                    if (last_low[1] - row["close"]) > row["ATR"] * 0.2:
                        if row["vol"] > row["volume_ma"] * volume_mult:
                            signal = "空"

        if signal and position is None:
            # 使用下一根K线的开盘价入场（更真实）
            entry_price = df.iloc[i+1]["open"]
            atr = row["ATR"]
            if signal == "多":
                sl = entry_price - atr * atr_sl_mult
                tp = entry_price + atr * atr_tp_mult
            else:
                sl = entry_price + atr * atr_sl_mult
                tp = entry_price - atr * atr_tp_mult

            risk_amount = capital * (risk_percent / 100)
            stop_dist = abs(entry_price - sl)
            qty = risk_amount / stop_dist if stop_dist > 0 else 0
            qty = round(qty / 0.01) * 0.01
            if qty < 0.01:
                qty = 0.01

            fee = entry_price * qty * 0.0005
            if capital > fee:
                capital -= fee
                position = {
                    "direction": signal,
                    "entry": entry_price,
                    "qty": qty,
                    "sl": sl,
                    "tp": tp,
                    "open_idx": i+1,
                    "open_time": df.iloc[i+1]["ts"]
                }

        # 持仓管理
        if position is not None:
            # 从入场后的K线开始检查
            for k in range(position["open_idx"]+1, min(position["open_idx"]+30, len(df))):
                cur = df.iloc[k]
                if position["direction"] == "多":
                    if cur["low"] <= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    elif cur["high"] >= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break
                else:
                    if cur["high"] >= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    elif cur["low"] <= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break
            else:
                # 30根K线未触及，以最后收盘价平仓
                exit_price = df.iloc[-1]["close"]
                reason = "时间平仓"

            pnl = (exit_price - position["entry"]) * position["qty"] if position["direction"] == "多" \
                else (position["entry"] - exit_price) * position["qty"]
            fee = exit_price * position["qty"] * 0.0005
            net_pnl = pnl - fee
            capital += net_pnl
            trades.append({
                "时间": position["open_time"],
                "方向": position["direction"],
                "入场": round(position["entry"], 2),
                "离场": round(exit_price, 2),
                "盈亏": round(net_pnl, 2),
                "原因": reason
            })
            position = None

        equity_curve.append(capital)

    # 绩效计算
    if trades:
        df_t = pd.DataFrame(trades)
        wins = len(df_t[df_t["盈亏"] > 0])
        losses = len(df_t[df_t["盈亏"] < 0])
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0
        net_profit = capital - 1000
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24*12) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown) * 100

        return trades, win_rate, net_profit, sharpe, max_dd
    else:
        return [], 0, 0, 0, 0

# --------------------------
# 主程序
# --------------------------
df_5m = get_klines("5m", 500)
if df_5m.empty:
    st.error("无法获取行情数据")
    st.stop()

df = compute_indicators(df_5m)
swing_highs, swing_lows = find_swing_points(df, window=swing_window)

# 实时信号（基于最新K线收盘）
latest = df.iloc[-1]
prev = df.iloc[-2]

valid_highs = [(t, p) for t, p in swing_highs if t <= latest['ts']]
valid_lows = [(t, p) for t, p in swing_lows if t <= latest['ts']]
last_high = valid_highs[-1] if valid_highs else None
last_low = valid_lows[-1] if valid_lows else None
prev_high = valid_highs[-2] if len(valid_highs) >= 2 else None
prev_low = valid_lows[-2] if len(valid_lows) >= 2 else None

bull_structure = prev_low and last_low and last_low[1] > prev_low[1]
bear_structure = prev_high and last_high and last_high[1] < prev_high[1]
tf15_dir = get_15m_direction()

signal = None
if (latest["trend_up_streak"] and tf15_dir == "多" and bull_structure and not is_fake_break(latest)):
    if last_high and latest["close"] > last_high[1]:
        body = abs(latest["close"] - latest["open"])
        if body > (latest["high"] - latest["low"]) * 0.6:
            if (latest["close"] - last_high[1]) > latest["ATR"] * 0.2:
                if latest["vol"] > latest["volume_ma"] * volume_mult:
                    signal = "多"
elif (latest["trend_down_streak"] and tf15_dir == "空" and bear_structure and not is_fake_break(latest)):
    if last_low and latest["close"] < last_low[1]:
        body = abs(latest["close"] - latest["open"])
        if body > (latest["high"] - latest["low"]) * 0.6:
            if (last_low[1] - latest["close"]) > latest["ATR"] * 0.2:
                if latest["vol"] > latest["volume_ma"] * volume_mult:
                    signal = "空"

# --------------------------
# 顶部面板
# --------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("15分钟方向", tf15_dir)
c2.metric("ADX", f"{latest['ADX']:.1f}")
c3.metric("结构状态", f"多:{bull_structure} 空:{bear_structure}")
if signal:
    c4.success(f"📢 实时信号: {signal}")
else:
    c4.info("无信号")

# --------------------------
# 回测结果
# --------------------------
with st.spinner("回测进行中..."):
    trades, win_rate, net_profit, sharpe, max_dd = run_backtest(df, swing_highs, swing_lows)

if trades:
    st.subheader("📊 回测绩效 (1000 USDT 初始资金)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总交易", len(trades))
    col2.metric("胜率", f"{win_rate:.1f}%")
    col3.metric("净利润", f"{net_profit:+.2f}")
    col4.metric("夏普比率", f"{sharpe:.2f}")
    col5.metric("最大回撤", f"{max_dd:.2f}%")
    st.dataframe(pd.DataFrame(trades).tail(20), use_container_width=True)
else:
    st.warning("回测期间无交易")

# --------------------------
# K线图
# --------------------------
fig = go.Figure(data=[go.Candlestick(
    x=df['ts'], open=df['open'], high=df['high'],
    low=df['low'], close=df['close'], name="ETH 5m"
)])
fig.add_trace(go.Scatter(x=df['ts'], y=df['EMA_fast'], line=dict(color='yellow', width=1), name=f"EMA{ema_fast_period}"))
fig.add_trace(go.Scatter(x=df['ts'], y=df['EMA_slow'], line=dict(color='orange', width=1), name=f"EMA{ema_slow_period}"))
# 结构点
if swing_highs:
    sh_x, sh_y = zip(*swing_highs)
    fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='结构高点'))
if swing_lows:
    sl_x, sl_y = zip(*swing_lows)
    fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='结构低点'))
# 实时信号箭头
if signal:
    y_pos = latest['high'] * 1.002 if signal == "多" else latest['low'] * 0.998
    symbol = "arrow-up" if signal == "多" else "arrow-down"
    fig.add_trace(go.Scatter(
        x=[latest['ts']], y=[y_pos],
        mode='markers+text',
        marker=dict(symbol=symbol, size=15, color='yellow'),
        text=signal,
        textposition="top center" if signal=="多" else "bottom center",
        name='实时信号'
    ))
fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 信号历史
# --------------------------
st.subheader("📜 最近信号记录")
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if signal and (not st.session_state.signal_history or st.session_state.signal_history[-1]["时间"] != latest["ts"]):
    st.session_state.signal_history.append({
        "时间": latest["ts"],
        "方向": signal,
        "价格": round(latest["close"], 2),
        "ATR": round(latest["ATR"], 2)
    })
    if len(st.session_state.signal_history) > 50:
        st.session_state.signal_history = st.session_state.signal_history[-50:]

if st.session_state.signal_history:
    hist_df = pd.DataFrame(st.session_state.signal_history)
    hist_df['时间'] = hist_df['时间'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(hist_df, use_container_width=True)
else:
    st.info("暂无历史信号")
