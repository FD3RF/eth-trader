"""
AI提示 + 双向趋势 + 回测优化版
目标：
- 低交易量
- RR>=2
- 胜率虽低但期望为正
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🤖 AI进场提示 + 回测优化版")

# ==========================
# 上传CSV
# ==========================
uploaded = st.file_uploader("上传CSV (ts, open, high, low, close, vol)", type=["csv"])

if not uploaded:
    st.info("请上传数据")
    st.stop()

df = pd.read_csv(uploaded, parse_dates=["ts"])
df = df.sort_values("ts").reset_index(drop=True)

st.success(f"数据加载：{len(df)} 行")

# ==========================
# 参数
# ==========================
with st.sidebar:
    st.header("⚙ 参数")
    ema_fast = st.slider("EMA快", 5, 20, 9)
    ema_slow = st.slider("EMA慢", 20, 60, 21)
    adx_thr = st.slider("ADX阈值", 15, 35, 18)
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)

# ==========================
# 指标
# ==========================
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df = df.dropna().reset_index(drop=True)

# ==========================
# AI进场提示逻辑（低交易）
# ==========================
def ai_signal(row):
    trend_up = row["EMA_fast"] > row["EMA_slow"]
    trend_down = row["EMA_fast"] < row["EMA_slow"]
    strong = row["ADX"] > adx_thr

    body = abs(row["close"] - row["open"])
    range_ = row["high"] - row["low"]
    strong_bar = body > range_ * 0.6

    # 多
    if trend_up and strong and strong_bar:
        return "多"
    # 空
    if trend_down and strong and strong_bar:
        return "空"
    return None

# ==========================
# 回测引擎（低交易 + RR）
# ==========================
def backtest(df):
    capital = 1000.0
    trades = []
    equity = [capital]
    position = None

    for i in range(1, len(df)-1):
        row = df.iloc[i]

        # 信号
        signal = ai_signal(row)

        # 开仓
        if signal and not position:
            entry = df.iloc[i+1]["open"]
            if signal == "多":
                entry *= (1 + slippage/100)
                sl = entry - row["ATR"] * 0.5
                tp = entry + row["ATR"] * 1.5
            else:
                entry *= (1 - slippage/100)
                sl = entry + row["ATR"] * 0.5
                tp = entry - row["ATR"] * 1.5

            risk = capital * (risk_pct / 100)
            dist = abs(entry - sl)
            if dist <= 0:
                continue
            qty = risk / dist

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
            for k in range(position["open_idx"]+1, min(position["open_idx"]+30, len(df))):
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
                exit_price = df.iloc[-1]["close"]
                reason = "时间"

            # 滑点
            if position["dir"] == "多":
                exit_price *= (1 - slippage/100)
            else:
                exit_price *= (1 + slippage/100)

            pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"] == "多" \
                else (position["entry"] - exit_price) * position["qty"]

            capital += pnl
            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "盈亏": pnl,
                "原因": reason
            })
            position = None

        equity.append(capital)

    # 绩效
    if trades:
        df_t = pd.DataFrame(trades)
        win = (df_t["盈亏"] > 0).mean() * 100
        net = capital - 1000
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        mdd = np.max(dd) * 100
        return df_t, win, net, mdd, equity
    return pd.DataFrame(), 0, 0, 0, equity


# ==========================
# 运行
# ==========================
if st.button("🚀 回测"):
    with st.spinner("回测中..."):
        trades, win, net, mdd, equity = backtest(df)

    st.subheader("回测结果")
    col1, col2, col3 = st.columns(3)
    col1.metric("交易", len(trades))
    col2.metric("胜率%", f"{win:.1f}")
    col3.metric("净利润", f"{net:.2f}")

    st.metric("最大回撤%", f"{mdd:.2f}")

    # 资金曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines"))
    fig.update_layout(title="资金曲线")
    st.plotly_chart(fig, use_container_width=True)

    if not trades.empty:
        st.dataframe(trades.tail(20))
