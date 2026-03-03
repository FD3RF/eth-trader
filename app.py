"""
高频微回踩模型 3.0 真高频版
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("⚡ 高频微回踩模型 3.0")

uploaded = st.file_uploader("📂 上传CSV (ts, open, high, low, close, vol)", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)

st.success(f"文件加载成功，共 {len(df)} 行")

with st.sidebar:
    st.header("参数")
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 0.5, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)
    run_btn = st.button("运行回测")

# ===== 指标 =====
df["EMA9"] = ta.trend.ema_indicator(df["close"], 9)
df["EMA21"] = ta.trend.ema_indicator(df["close"], 21)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
df = df.dropna().reset_index(drop=True)

# ===== 回测 =====
def run_backtest(df):

    capital = 1000
    initial = 1000
    equity = [capital]
    trades = []
    position = None

    consecutive_loss = 0
    pause_until = -1
    current_day = None
    day_start = capital

    for i in range(30, len(df)-1):

        row = df.iloc[i]
        prev = df.iloc[i-1]
        next_open = df.iloc[i+1]["open"]

        # 日切换
        if current_day != row["ts"].date():
            current_day = row["ts"].date()
            day_start = capital

        if day_start - capital > initial * 0.06:
            equity.append(capital)
            continue

        if i < pause_until:
            equity.append(capital)
            continue

        atr = row["ATR"]
        atr_pct = atr / row["close"]

        # 波动太低不做
        if atr_pct < 0.0015:
            equity.append(capital)
            continue

        # ===== 开仓 =====
        if position is None:

            signal = None

            # 多趋势
            if row["EMA9"] > row["EMA21"]:

                # 微回踩 ≤0.3 ATR
                if (
                    prev["low"] < prev["EMA9"]
                    and abs(prev["low"] - prev["EMA9"]) <= 0.3 * atr
                    and row["close"] > row["EMA9"]
                ):
                    signal = "多"

            # 空趋势
            elif row["EMA9"] < row["EMA21"]:

                if (
                    prev["high"] > prev["EMA9"]
                    and abs(prev["high"] - prev["EMA9"]) <= 0.3 * atr
                    and row["close"] < row["EMA9"]
                ):
                    signal = "空"

            if signal:

                entry = next_open
                sl_mult = 0.6
                tp_mult = 0.8

                if signal == "多":
                    entry *= (1 + slippage/100)
                    sl = entry - atr * sl_mult
                    tp = entry + atr * tp_mult
                else:
                    entry *= (1 - slippage/100)
                    sl = entry + atr * sl_mult
                    tp = entry - atr * tp_mult

                risk_amount = capital * (risk_pct/100)
                stop_dist = abs(entry - sl)
                qty = risk_amount / stop_dist

                fee_open = entry * qty * 0.0005
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

            for k in range(position["open_idx"], min(position["open_idx"]+8, len(df))):
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
                consecutive_loss += 1
            else:
                consecutive_loss = 0

            if consecutive_loss >= 6:
                pause_until = i + 20

            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "盈亏": round(net,2),
                "原因": reason
            })

            position = None

        equity.append(capital)

        if capital < initial * 0.5:
            break

    if trades:
        df_t = pd.DataFrame(trades)
        win_rate = (df_t["盈亏"]>0).mean()*100
        net_profit = capital - initial

        returns = np.diff(equity)/equity[:-1]
        sharpe = np.mean(returns)/np.std(returns)*np.sqrt(365*24*12) if np.std(returns)>0 else 0

        peak = np.maximum.accumulate(equity)
        dd = (peak-equity)/peak
        max_dd = np.max(dd)*100

        return trades, win_rate, net_profit, sharpe, max_dd, equity
    else:
        return [],0,0,0,0,equity

# ===== 运行 =====
if run_btn:
    trades, win_rate, net_profit, sharpe, max_dd, equity = run_backtest(df)

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("交易次数", len(trades))
    col2.metric("胜率", f"{win_rate:.1f}%")
    col3.metric("净利润", f"{net_profit:.2f}")
    col4.metric("夏普", f"{sharpe:.2f}")
    col5.metric("最大回撤", f"{max_dd:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"].iloc[:len(equity)], y=equity))
    st.plotly_chart(fig, use_container_width=True)
