"""
AI提示 + 回测
- 5分钟趋势判断
- 多空提示
- 回测绩效
- 不自动交易
"""

import streamlit as st
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🤖 AI提示 + 回测")

uploaded = st.file_uploader("上传CSV (ts, open, high, low, close, vol)", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)

st.success(f"数据：{len(df)} 行")

# ===== 参数 =====
with st.sidebar:
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 0.5, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)
    run_btn = st.button("运行回测")

# ===== 指标 =====
df["EMA9"] = ta.trend.ema_indicator(df["close"], 9)
df["EMA21"] = ta.trend.ema_indicator(df["close"], 21)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], 14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
df["volume_ma"] = df["vol"].rolling(20).mean()
df = df.dropna().reset_index(drop=True)

# ===== AI提示逻辑 =====
def ai_signal(row, prev):
    """
    简单AI：趋势 + 回踩 + ADX
    返回：多 / 空 / 无
    """

    if row["ADX"] <= 15:
        return "无"

    # 多
    if row["EMA9"] > row["EMA21"]:
        dist = prev["EMA9"] - prev["low"]
        if 0.1*row["ATR"] <= dist <= 0.9*row["ATR"]:
            return "多"

    # 空
    if row["EMA9"] < row["EMA21"]:
        dist = prev["high"] - prev["EMA9"]
        if 0.1*row["ATR"] <= dist <= 0.9*row["ATR"]:
            return "空"

    return "无"


# ===== 回测 =====
def backtest(df):

    capital = 1000
    equity = [capital]
    trades = []
    position = None

    for i in range(30, len(df)-1):

        row = df.iloc[i]
        prev = df.iloc[i-1]

        signal = ai_signal(row, prev)

        # 开仓
        if signal != "无" and position is None:

            entry = df.iloc[i+1]["open"]
            entry *= (1 + slippage/100) if signal=="多" else (1 - slippage/100)

            sl = entry - row["ATR"]*0.5 if signal=="多" else entry + row["ATR"]*0.5
            tp = entry + row["ATR"]*1.2 if signal=="多" else entry - row["ATR"]*1.2

            risk = capital * (risk_pct/100)
            stop = abs(entry - sl)
            if stop <= 0:
                equity.append(capital)
                continue

            qty = risk / stop
            qty = round(qty / 0.01) * 0.01
            if qty < 0.01:
                qty = 0.01

            fee = entry * qty * 0.0005
            if capital > fee:
                capital -= fee
                position = {
                    "dir": signal,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "qty": qty,
                    "open_idx": i+1,
                    "open_time": df.iloc[i+1]["ts"]
                }

        # 持仓
        if position:

            exit_price = None
            reason = None

            for k in range(position["open_idx"], min(position["open_idx"]+10, len(df))):
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

            exit_price *= (1 - slippage/100) if position["dir"]=="多" else (1 + slippage/100)

            pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"]=="多" else (position["entry"]-exit_price)*position["qty"]
            fee = exit_price * position["qty"] * 0.0005
            net = pnl - fee
            capital += net

            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "盈亏": round(net, 2),
                "原因": reason
            })

            position = None

        equity.append(capital)

    # ===== 绩效 =====
    if trades:
        df_t = pd.DataFrame(trades)
        win_rate = (df_t["盈亏"] > 0).mean() * 100
        net_profit = capital - 1000

        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24*12) if np.std(returns) > 0 else 0

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = np.max(dd) * 100

        return trades, win_rate, net_profit, sharpe, max_dd, equity
    else:
        return [], 0, 0, 0, 0, equity


# ===== 运行 =====
if run_btn:
    with st.spinner("回测中..."):
        trades, win_rate, net_profit, sharpe, max_dd, equity = backtest(df)

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("交易次数", len(trades))
    col2.metric("胜率", f"{win_rate:.1f}%")
    col3.metric("净利润", f"{net_profit:.2f}")
    col4.metric("最大回撤", f"{max_dd:.2f}%")

    if len(equity) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"].iloc[:len(equity)], y=equity))
        st.plotly_chart(fig, use_container_width=True)

    if trades:
        st.dataframe(pd.DataFrame(trades).tail(20), use_container_width=True)
    else:
        st.info("无交易")
