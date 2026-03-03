"""
AI提示 + 小利润策略 + 回测（优化版）
- 多空提示
- 回测统计
- RR控制
- 滑点/手续费
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="AI提示 + 回测")
st.title("🤖 AI提示 + 回测（优化版）")

# =========================
# 上传CSV
# =========================
uploaded = st.file_uploader("上传CSV (ts, open, high, low, close, vol)", type=["csv"])

if not uploaded:
    st.info("上传CSV开始分析")
    st.stop()

df = pd.read_csv(uploaded)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)

st.success(f"数据加载：{len(df)} 行")

# =========================
# 参数
# =========================
with st.sidebar:
    st.header("⚙ 参数")
    ema_fast = st.slider("EMA快", 5, 20, 8)
    ema_slow = st.slider("EMA慢", 20, 60, 30)
    adx_thr = st.slider("ADX阈值", 12, 30, 18)
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)

# =========================
# 指标
# =========================
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df = df.dropna().reset_index(drop=True)

# =========================
# AI提示（轻量）
# =========================
def ai_signal(row):
    trend_up = row["EMA_fast"] > row["EMA_slow"]
    trend_down = row["EMA_fast"] < row["EMA_slow"]
    strong = row["ADX"] > adx_thr

    if strong and trend_up:
        return "多"
    if strong and trend_down:
        return "空"
    return "无"

df["signal"] = df.apply(ai_signal, axis=1)

latest = df.iloc[-1]
st.subheader("📢 AI提示")
st.write(f"方向：{latest['signal']}")
st.write(f"理由：EMA{'上' if latest['EMA_fast']>latest['EMA_slow'] else '下'} + ADX={latest['ADX']:.1f}")

# =========================
# 回测
# =========================
def backtest(df):
    capital = 1000.0
    trades = []
    equity = [capital]
    position = None

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        signal = row["signal"]

        # 开仓
        if signal in ("多", "空") and position is None:
            entry = df.iloc[i+1]["open"]
            if signal == "多":
                entry *= (1 + slippage/100)
                sl = entry - row["ATR"] * 0.6
                tp = entry + row["ATR"] * 1.2
            else:
                entry *= (1 - slippage/100)
                sl = entry + row["ATR"] * 0.6
                tp = entry - row["ATR"] * 1.2

            risk = capital * (risk_pct/100)
            stop_dist = abs(entry - sl)
            if stop_dist <= 0:
                continue
            qty = risk / stop_dist

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
            for k in range(position["open_idx"]+1, min(position["open_idx"]+30, len(df))):
                cur = df.iloc[k]
                if position["dir"] == "多":
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
                exit_price = df.iloc[-1]["close"]
                reason = "时间"

            # 离场滑点
            exit_price *= (1 - slippage/100) if position["dir"]=="多" else (1 + slippage/100)

            pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"]=="多" else (position["entry"]-exit_price)*position["qty"]
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

    # 统计
    if trades:
        df_t = pd.DataFrame(trades)
        win = len(df_t[df_t["盈亏"]>0])
        total = len(df_t)
        win_rate = win/total*100
        net = capital - 1000
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = dd.max()*100
        return win_rate, net, max_dd, equity, total
    return 0, 0, 0, equity, 0

# =========================
# 运行回测
# =========================
with st.spinner("回测中..."):
    win_rate, net, max_dd, equity, total = backtest(df)

st.subheader("📊 回测结果")
col1,col2,col3,col4 = st.columns(4)
col1.metric("交易次数", total)
col2.metric("胜率", f"{win_rate:.1f}%")
col3.metric("净利润", f"{net:+.2f}")
col4.metric("最大回撤", f"{max_dd:.2f}%")

# 资金曲线
fig = go.Figure()
fig.add_trace(go.Scatter(y=equity, mode="lines", name="资金"))
fig.update_layout(title="资金曲线", height=400)
st.plotly_chart(fig, use_container_width=True)

# =========================
# AI实时建议
# =========================
st.subheader("📢 实时建议")

if latest["signal"]=="多":
    st.success("建议：观察回踩小仓试多")
elif latest["signal"]=="空":
    st.warning("建议：观察回踩小仓试空")
else:
    st.info("建议：观望")

st.caption("提示：回测亏损说明参数/市场不匹配，需要调整。没有策略百分百赚钱。")
