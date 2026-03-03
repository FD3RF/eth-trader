"""
AI提示 + 双向趋势 + 回测优化版
小利润策略：低风险、高质量信号
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os

st.set_page_config(layout="wide")
st.title("📈 AI提示 + 回测（优化版）")

# ----------------------
# 上传CSV
# ----------------------
uploaded = st.file_uploader("上传CSV (ts, open, high, low, close, vol)", type=["csv"])
if not uploaded:
    st.info("请上传CSV")
    st.stop()

df_raw = pd.read_csv(uploaded)
df_raw["ts"] = pd.to_datetime(df_raw["ts"])
df_raw = df_raw.sort_values("ts").reset_index(drop=True)

st.success(f"数据加载：{len(df_raw)} 行")

# ----------------------
# 参数
# ----------------------
with st.sidebar:
    st.subheader("⚙ 参数")
    ema_fast = st.slider("EMA快", 5, 20, 9)
    ema_slow = st.slider("EMA慢", 20, 60, 21)
    adx_thr = st.slider("ADX阈值", 12, 35, 18)
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)

# ----------------------
# 指标
# ----------------------
df = df_raw.copy()
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df["volume_ma"] = df["vol"].rolling(20).mean()
df = df.dropna().reset_index(drop=True)

# ----------------------
# 信号（双向）
# ----------------------
def is_fake(row):
    body = abs(row["close"] - row["open"])
    if body == 0:
        return True
    upper = row["high"] - max(row["close"], row["open"])
    lower = min(row["close"], row["open"]) - row["low"]
    return upper > body * 1.5 or lower > body * 1.5

def signal(row):
    trend_up = row["EMA_fast"] > row["EMA_slow"]
    trend_down = row["EMA_fast"] < row["EMA_slow"]

    if row["ADX"] < adx_thr:
        return None

    # 多
    if trend_up and not is_fake(row):
        return "多"
    # 空
    if trend_down and not is_fake(row):
        return "空"
    return None

# ----------------------
# 回测
# ----------------------
def backtest(df):
    capital = 1000.0
    position = None
    trades = []
    equity = [capital]

    for i in range(len(df)-1):
        row = df.iloc[i]
        sig = signal(row)

        # 开仓
        if sig and position is None:
            entry = df.iloc[i+1]["open"]
            entry *= (1 + slippage/100) if sig=="多" else (1 - slippage/100)

            sl = entry - row["ATR"]*0.5 if sig=="多" else entry + row["ATR"]*0.5
            tp = entry + row["ATR"]*1.5 if sig=="多" else entry - row["ATR"]*1.5

            risk = capital * (risk_pct/100)
            stop_dist = abs(entry - sl)
            qty = risk/stop_dist if stop_dist>0 else 0.01
            qty = max(round(qty/0.01)*0.01, 0.01)

            fee = entry * qty * 0.0005
            if capital > fee:
                capital -= fee
                position = {
                    "dir": sig,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "qty": qty,
                    "open_idx": i+1,
                    "open_time": df.iloc[i+1]["ts"]
                }

        # 平仓
        if position:
            exit_price = None
            reason = None
            for k in range(position["open_idx"]+1, min(position["open_idx"]+30, len(df))):
                cur = df.iloc[k]
                if position["dir"]=="多":
                    if cur["low"] <= position["sl"]:
                        exit_price = position["sl"]; reason="止损"; break
                    if cur["high"] >= position["tp"]:
                        exit_price = position["tp"]; reason="止盈"; break
                else:
                    if cur["high"] >= position["sl"]:
                        exit_price = position["sl"]; reason="止损"; break
                    if cur["low"] <= position["tp"]:
                        exit_price = position["tp"]; reason="止盈"; break
            if exit_price is None:
                exit_price = df.iloc[-1]["close"]; reason="时间"

            exit_price *= (1 - slippage/100) if position["dir"]=="多" else (1 + slippage/100)

            pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"]=="多" else (position["entry"] - exit_price)*position["qty"]
            fee = exit_price * position["qty"] * 0.0005
            net = pnl - fee
            capital += net

            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "盈亏": net
            })
            position = None

        equity.append(capital)

    if trades:
        df_t = pd.DataFrame(trades)
        win = (df_t["盈亏"] > 0).mean() * 100
        return trades, win, capital - 1000, equity
    return [], 0, 0, equity

# ----------------------
# 执行回测
# ----------------------
with st.spinner("回测中..."):
    trades, win, profit, equity = backtest(df)

st.subheader("回测结果")
col1, col2, col3 = st.columns(3)
col1.metric("交易次数", len(trades))
col2.metric("胜率", f"{win:.1f}%")
col3.metric("净利润", f"{profit:.2f}")

# 资金曲线
fig = go.Figure()
fig.add_trace(go.Scatter(y=equity, mode="lines"))
fig.update_layout(title="资金曲线")
st.plotly_chart(fig, use_container_width=True)

# 明细
if trades:
    st.dataframe(pd.DataFrame(trades).tail(20))
