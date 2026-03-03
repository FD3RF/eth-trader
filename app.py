"""
双向趋势策略（多空） + 回测
- EMA趋势
- ADX过滤
- 回踩入场
- RR控制
- 滑点/手续费
- 绩效指标
"""

import streamlit as st
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 双向趋势策略（多空）回测版")

uploaded = st.file_uploader("上传CSV (ts, open, high, low, close, vol)", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)

st.success(f"数据加载：{len(df)} 行")

with st.sidebar:
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 0.5, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)
    run_btn = st.button("回测")

# =========================
# 指标
# =========================
df["EMA9"] = ta.trend.ema_indicator(df["close"], 9)
df["EMA21"] = ta.trend.ema_indicator(df["close"], 21)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], 14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
df["volume_ma"] = df["vol"].rolling(20).mean()
df = df.dropna().reset_index(drop=True)

# =========================
# 趋势与回踩
# =========================
def trend_up(row):
    return row["EMA9"] > row["EMA21"]

def trend_down(row):
    return row["EMA9"] < row["EMA21"]

def pullback(row, prev):
    atr = row["ATR"]
    if atr <= 0:
        return False

    # 做多回踩：价格低于EMA但未深跌
    if trend_up(row):
        dist = prev["EMA9"] - prev["low"]
        return 0.15*atr <= dist <= 0.7*atr and row["close"] > row["EMA9"]

    # 做空回踩：价格高于EMA但未深涨
    if trend_down(row):
        dist = prev["high"] - prev["EMA9"]
        return 0.15*atr <= dist <= 0.7*atr and row["close"] < row["EMA9"]

    return False

# =========================
# 回测引擎
# =========================
def backtest(df):

    capital = 1000
    equity = [capital]
    trades = []
    position = None
    day_start = capital
    current_day = None
    consecutive_loss = 0
    pause_until = -1

    for i in range(30, len(df)-1):

        row = df.iloc[i]
        prev = df.iloc[i-1]

        # 日切换
        if current_day != row["ts"].date():
            current_day = row["ts"].date()
            day_start = capital

        # 日最大亏损 6%
        if day_start - capital > 1000*0.06:
            equity.append(capital)
            continue

        # 连亏暂停
        if i < pause_until:
            equity.append(capital)
            continue

        if position is None:

            signal = None

            if pullback(row, prev):

                if trend_up(row) and row["ADX"] > 18:
                    signal = "多"
                elif trend_down(row) and row["ADX"] > 18:
                    signal = "空"

            if signal:

                entry = df.iloc[i+1]["open"]
                entry *= (1 + slippage/100) if signal=="多" else (1 - slippage/100)

                sl = entry - row["ATR"]*0.6 if signal=="多" else entry + row["ATR"]*0.6
                tp = entry + row["ATR"]*1.2 if signal=="多" else entry - row["ATR"]*1.2

                risk = capital * (risk_pct/100)
                stop_dist = abs(entry - sl)
                if stop_dist <= 0:
                    equity.append(capital)
                    continue

                qty = risk / stop_dist
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

        # 持仓管理
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

            if net < 0:
                consecutive_loss += 1
            else:
                consecutive_loss = 0

            # 连亏暂停
            if consecutive_loss >= 6:
                pause_until = i + 20

            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "盈亏": round(net, 2),
                "原因": reason
            })

            position = None

        equity.append(capital)

        if capital < 500:
            break

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

# =========================
# 运行
# =========================
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
