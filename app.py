"""
AI微回踩 + 趋势过滤 + 回测（优化版）
- 高频小机会 → 低回撤
- 双周期趋势
- RR=2+
- 滑点/手续费
"""

import streamlit as st
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="AI微回踩回测")
st.title("📈 AI微回踩 + 回测（优化版）")

DATA_DIR = "market_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ======================
# 数据加载
# ======================
uploaded = st.file_uploader("上传CSV (ts,open,high,low,close,vol)", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded, parse_dates=["ts"])
    st.success(f"数据：{len(df_raw)} 行")
else:
    st.warning("请上传CSV")
    st.stop()

# ======================
# 参数
# ======================
with st.sidebar:
    st.header("⚙ 参数")

    # 趋势
    ema_fast = st.slider("EMA快", 5, 20, 9)
    ema_slow = st.slider("EMA慢", 20, 60, 21)
    adx_thr = st.slider("ADX阈值", 15, 35, 18)

    # 风控
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05)

    # 回测
    run_btn = st.button("🚀 回测")

# ======================
# 指标
# ======================
df = df_raw.copy()
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df["volume_ma"] = df["vol"].rolling(20).mean()
df = df.dropna().reset_index(drop=True)

# ======================
# 结构（微回踩）
# ======================
def micro_pullback(row):
    body = abs(row["close"] - row["open"])
    if body == 0:
        return False
    upper = row["high"] - max(row["close"], row["open"])
    lower = min(row["close"], row["open"]) - row["low"]
    # 小影线回踩
    return (upper < body * 0.8) and (lower < body * 0.8)

# ======================
# 回测
# ======================
def backtest(df):
    capital = 1000.0
    position = None
    trades = []
    equity = [capital]

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        next_open = df.iloc[i+1]["open"]

        # 双周期趋势
        trend_up = row["EMA_fast"] > row["EMA_slow"] and row["ADX"] > adx_thr
        trend_down = row["EMA_fast"] < row["EMA_slow"] and row["ADX"] > adx_thr

        signal = None
        if micro_pullback(row):
            if trend_up:
                signal = "多"
            elif trend_down:
                signal = "空"

        # 开仓
        if signal and position is None:
            entry = next_open * (1 + slippage/100 if signal=="多" else 1 - slippage/100)
            sl = entry - row["ATR"] * 0.5 if signal=="多" else entry + row["ATR"] * 0.5
            tp = entry + row["ATR"] * 1.5 if signal=="多" else entry - row["ATR"] * 1.5

            risk = capital * (risk_pct / 100)
            dist = abs(entry - sl)
            if dist <= 0:
                continue
            qty = risk / dist

            fee = entry * qty * 0.0005
            if capital > fee:
                capital -= fee
                position = {
                    "dir": signal,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "qty": qty,
                    "open_idx": i+1
                }

        # 持仓
        if position:
            exit_price = None
            for k in range(position["open_idx"]+1, min(position["open_idx"]+30, len(df))):
                cur = df.iloc[k]
                if position["dir"]=="多":
                    if cur["low"] <= position["sl"]:
                        exit_price = position["sl"]; break
                    if cur["high"] >= position["tp"]:
                        exit_price = position["tp"]; break
                else:
                    if cur["high"] >= position["sl"]:
                        exit_price = position["sl"]; break
                    if cur["low"] <= position["tp"]:
                        exit_price = position["tp"]; break

            if exit_price is None:
                exit_price = df.iloc[-1]["close"]

            exit_price *= (1 - slippage/100 if position["dir"]=="多" else 1 + slippage/100)
            pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"]=="多" else (position["entry"]-exit_price)*position["qty"]
            fee = exit_price * position["qty"] * 0.0005
            net = pnl - fee
            capital += net

            trades.append({"pnl": net})
            position = None

        equity.append(capital)

    # 统计
    if trades:
        df_t = pd.DataFrame(trades)
        wins = (df_t["pnl"]>0).sum()
        total = len(df_t)
        win_rate = wins/total*100
        net = capital - 1000
        return win_rate, net, equity, total
    return 0, 0, equity, 0

# ======================
# 执行回测
# ======================
if run_btn:
    with st.spinner("回测中..."):
        wr, net, equity, cnt = backtest(df)

    st.subheader("回测结果")
    col1, col2, col3 = st.columns(3)
    col1.metric("交易", cnt)
    col2.metric("胜率", f"{wr:.1f}%")
    col3.metric("净利润", f"{net:+.2f}")

    # 曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines"))
    fig.update_layout(title="资金曲线")
    st.plotly_chart(fig, use_container_width=True)

    st.info("低胜率并不代表失败：需RR>2+，交易少而精")
