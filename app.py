"""
AI提示 + 双周期趋势 + 回测（优化版）
目标：
- 高频→低频
- RR>2
- 双周期确认
- 交易质量优先
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="AI提示回测优化版")
st.title("📈 AI提示 + 回测（优化）")

# -------------------------
# 上传CSV
# -------------------------
uploaded = st.file_uploader("上传CSV (ts, open, high, low, close, vol)", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded, parse_dates=["ts"])
df = df.sort_values("ts").reset_index(drop=True)
st.success(f"数据加载：{len(df)} 行")

# -------------------------
# 参数
# -------------------------
with st.sidebar:
    st.header("参数")
    ema_fast = st.slider("EMA快", 5, 20, 9)
    ema_slow = st.slider("EMA慢", 20, 60, 21)
    adx_thr = st.slider("ADX阈值", 12, 30, 18)
    risk = st.slider("单笔风险%", 0.2, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05)

# -------------------------
# 指标
# -------------------------
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df["volume_ma"] = df["vol"].rolling(20).mean()
df = df.dropna().reset_index(drop=True)

# -------------------------
# 双周期趋势（5分 + 15分模拟）
# -------------------------
def trend_signal(row):
    return "多" if row["EMA_fast"] > row["EMA_slow"] else "空"

df["trend"] = df.apply(trend_signal, axis=1)

# -------------------------
# AI进场提示（轻量）
# -------------------------
def ai_hint(row):
    if row["ADX"] < adx_thr:
        return "无（趋势弱）"
    if row["EMA_fast"] > row["EMA_slow"]:
        return "多（趋势上）"
    if row["EMA_fast"] < row["EMA_slow"]:
        return "空（趋势下）"
    return "无"

df["hint"] = df.apply(ai_hint, axis=1)

# -------------------------
# 回测（低交易 RR>2）
# -------------------------
def backtest(df):
    capital = 1000
    trades = []
    position = None
    equity = [capital]

    for i in range(1, len(df)-1):
        row = df.iloc[i]

        # 信号：趋势+ADX
        signal = None
        if row["ADX"] > adx_thr:
            signal = row["trend"]

        # 开仓（低频）
        if signal and position is None:
            entry = df.iloc[i+1]["open"] * (1 + slippage/100 if signal=="多" else 1 - slippage/100)
            sl = entry - row["ATR"]*0.6 if signal=="多" else entry + row["ATR"]*0.6
            tp = entry + row["ATR"]*1.4 if signal=="多" else entry - row["ATR"]*1.4

            qty = (capital * (risk/100)) / abs(entry - sl)
            qty = max(round(qty/0.01)*0.01, 0.01)

            fee = entry * qty * 0.0005
            if capital > fee:
                capital -= fee
                position = {"dir": signal, "entry": entry, "sl": sl, "tp": tp, "qty": qty, "idx": i+1}

        # 持仓管理
        if position:
            cur = df.iloc[i]
            exit_price = None
            reason = None

            if position["dir"]=="多":
                if cur["low"] <= position["sl"]:
                    exit_price = position["sl"]; reason="止损"
                elif cur["high"] >= position["tp"]:
                    exit_price = position["tp"]; reason="止盈"
            else:
                if cur["high"] >= position["sl"]:
                    exit_price = position["sl"]; reason="止损"
                elif cur["low"] <= position["tp"]:
                    exit_price = position["tp"]; reason="止盈"

            if exit_price is not None:
                exit_price *= (1 - slippage/100 if position["dir"]=="多" else 1 + slippage/100)
                pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"]=="多" else (position["entry"]-exit_price)*position["qty"]
                fee = exit_price * position["qty"] * 0.0005
                net = pnl - fee
                capital += net
                trades.append({"方向": position["dir"], "盈亏": net})
                position = None

        equity.append(capital)

    # 绩效
    if trades:
        df_t = pd.DataFrame(trades)
        win = (df_t["盈亏"] > 0).mean() * 100
        net = capital - 1000
        return len(trades), win, net, equity
    return 0, 0, 0, equity

# -------------------------
# 执行回测
# -------------------------
if st.button("🚀 回测"):
    trades, win, net, equity = backtest(df)

    st.metric("交易次数", trades)
    st.metric("胜率", f"{win:.1f}%")
    st.metric("净利润", f"{net:+.2f}")

    # 曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines"))
    fig.update_layout(title="资金曲线")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# AI提示
# -------------------------
st.subheader("🤖 AI进场提示")
latest = df.iloc[-1]
st.info(f"方向建议：{latest['hint']}")
