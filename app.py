"""
高频小利润爆发模型 2.0 专业版
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🚀 高频小利润爆发模型 2.0")

# =========================
# 文件上传
# =========================
uploaded = st.file_uploader("📂 上传CSV (ts, open, high, low, close, vol)", type=["csv"])

if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)

st.success(f"文件加载成功，共 {len(df)} 行")

# =========================
# 参数
# =========================
with st.sidebar:
    st.header("⚙ 参数")
    ema_fast = st.slider("EMA快", 5, 20, 8)
    ema_slow = st.slider("EMA慢", 20, 60, 30)
    adx_thr = st.slider("ADX阈值", 15, 35, 22)
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 0.5, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)
    run_btn = st.button("运行回测")

# =========================
# 指标
# =========================
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df["RSI"] = ta.momentum.rsi(df["close"], window=14)
df = df.dropna().reset_index(drop=True)

# =========================
# 回测函数（第二轮专业优化）
# =========================
def run_backtest(df):

    initial_capital = 1000
    capital = initial_capital
    equity_curve = [capital]
    trades = []
    position = None

    consecutive_losses = 0
    pause_until = -1
    max_daily_loss_pct = 0.05
    current_day = None
    day_start_capital = capital

    for i in range(50, len(df)-1):

        row = df.iloc[i]
        prev = df.iloc[i-1]
        next_open = df.iloc[i+1]["open"]

        # 日内风控
        if current_day != row["ts"].date():
            current_day = row["ts"].date()
            day_start_capital = capital

        if (day_start_capital - capital) >= initial_capital * max_daily_loss_pct:
            equity_curve.append(capital)
            continue

        if i < pause_until:
            equity_curve.append(capital)
            continue

        # ===== 开仓逻辑 =====
        if position is None:

            atr_percent = row["ATR"] / row["close"]

            # 波动过滤
            if not (0.002 < atr_percent < 0.015):
                equity_curve.append(capital)
                continue

            dynamic_rsi = 35 if row["ADX"] > 25 else 30
            signal = None

            # 多单 二段式回补
            if (
                row["EMA_fast"] > row["EMA_slow"]
                and prev["close"] < prev["EMA_fast"] - 0.7 * prev["ATR"]
                and row["close"] > row["EMA_fast"]
                and prev["RSI"] < dynamic_rsi
            ):
                signal = "多"

            # 空单
            elif (
                row["EMA_fast"] < row["EMA_slow"]
                and prev["close"] > prev["EMA_fast"] + 0.7 * prev["ATR"]
                and row["close"] < row["EMA_fast"]
                and prev["RSI"] > 100 - dynamic_rsi
            ):
                signal = "空"

            if signal:

                entry = next_open

                # 自适应RR
                if atr_percent > 0.008:
                    sl_mult = 0.6
                    tp_mult = 0.8
                else:
                    sl_mult = 0.7
                    tp_mult = 1.0

                if signal == "多":
                    entry *= (1 + slippage/100)
                    sl = entry - row["ATR"] * sl_mult
                    tp = entry + row["ATR"] * tp_mult
                else:
                    entry *= (1 - slippage/100)
                    sl = entry + row["ATR"] * sl_mult
                    tp = entry - row["ATR"] * tp_mult

                risk_amount = capital * (risk_pct/100)
                stop_dist = abs(entry - sl)

                if stop_dist <= 0:
                    continue

                qty = risk_amount / stop_dist
                fee_open = entry * qty * 0.0005

                if capital > fee_open:
                    capital -= fee_open
                    position = {
                        "dir": signal,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "open_idx": i+1,
                        "open_time": df.iloc[i+1]["ts"]
                    }

        # ===== 持仓管理 =====
        if position:

            exit_price = None
            reason = None

            for k in range(position["open_idx"], min(position["open_idx"]+20, len(df))):
                cur = df.iloc[k]

                if position["dir"] == "多":
                    if cur["low"] <= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    if cur["high"] >= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break
                else:
                    if cur["high"] >= position["sl"]:
                        exit_price = position["sl"]
                        reason = "止损"
                        break
                    if cur["low"] <= position["tp"]:
                        exit_price = position["tp"]
                        reason = "止盈"
                        break

            if exit_price is None:
                exit_price = df.iloc[k]["close"]
                reason = "时间"

            # 滑点
            if position["dir"] == "多":
                exit_price *= (1 - slippage/100)
                pnl = (exit_price - position["entry"]) * position["qty"]
            else:
                exit_price *= (1 + slippage/100)
                pnl = (position["entry"] - exit_price) * position["qty"]

            fee_close = exit_price * position["qty"] * 0.0005
            net = pnl - fee_close
            capital += net

            if net < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

            if consecutive_losses >= 5:
                pause_until = i + 24

            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "盈亏": round(net,2),
                "原因": reason
            })

            position = None

        equity_curve.append(capital)

        if capital <= initial_capital * 0.5:
            break

    # ===== 绩效 =====
    if trades:
        df_t = pd.DataFrame(trades)
        wins = len(df_t[df_t["盈亏"]>0])
        total = len(df_t)
        win_rate = wins/total*100
        net_profit = capital - initial_capital

        returns = np.diff(equity_curve)/equity_curve[:-1]
        sharpe = np.mean(returns)/np.std(returns)*np.sqrt(365*24*12) if np.std(returns)>0 else 0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak-equity_curve)/peak
        max_dd = np.max(drawdown)*100

        return trades, win_rate, net_profit, sharpe, max_dd, equity_curve
    else:
        return [],0,0,0,0,equity_curve

# =========================
# 运行
# =========================
if run_btn:

    trades, win_rate, net_profit, sharpe, max_dd, equity_curve = run_backtest(df)

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("交易次数", len(trades))
    col2.metric("胜率", f"{win_rate:.1f}%")
    col3.metric("净利润", f"{net_profit:.2f}")
    col4.metric("夏普", f"{sharpe:.2f}")
    col5.metric("最大回撤", f"{max_dd:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"].iloc[:len(equity_curve)], y=equity_curve))
    st.plotly_chart(fig, use_container_width=True)
