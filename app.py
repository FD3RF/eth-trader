"""
AI提示 + 回测（双周期趋势 + 小利润思路）
可运行 Streamlit
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("🤖 AI提示 + 回测优化版")

# ==========================
# 上传数据
# ==========================
file = st.file_uploader("上传CSV (ts, open, high, low, close, vol)", type=["csv"])
if not file:
    st.stop()

df = pd.read_csv(file)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)

st.success(f"数据加载：{len(df)} 行")

# ==========================
# 参数
# ==========================
with st.sidebar:
    st.header("⚙ 参数")
    ema_fast = st.slider("EMA快", 5, 20, 8)
    ema_slow = st.slider("EMA慢", 20, 60, 30)
    adx_thr = st.slider("ADX阈值", 15, 35, 20)
    risk_pct = st.slider("单笔风险%", 0.2, 2.0, 1.0, step=0.1)
    slippage = st.number_input("滑点%", 0.0, 0.2, 0.05, step=0.01)
    st.info("AI只提示，不自动交易")

# ==========================
# 指标
# ==========================
df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
df["volume_ma"] = df["vol"].rolling(20).mean()
df = df.dropna().reset_index(drop=True)

# ==========================
# AI提示（5分钟逻辑）
# ==========================
def ai_signal(row):
    trend_up = row["EMA_fast"] > row["EMA_slow"]
    trend_down = row["EMA_fast"] < row["EMA_slow"]
    adx_ok = row["ADX"] > adx_thr

    # 做多
    if trend_up and adx_ok and row["close"] > row["EMA_fast"]:
        return "多"
    # 做空
    if trend_down and adx_ok and row["close"] < row["EMA_fast"]:
        return "空"
    return None

df["signal"] = df.apply(ai_signal, axis=1)

# ==========================
# 回测引擎（小利润 + RR）
# ==========================
def backtest(df):
    capital = 1000.0
    position = None
    trades = []
    equity = [capital]

    for i in range(len(df)-1):
        row = df.iloc[i]
        sig = row["signal"]

        # 开仓
        if sig and position is None:
            entry = df.iloc[i+1]["open"] * (1 + slippage/100 if sig=="多" else 1 - slippage/100)
            sl = entry - row["ATR"]*0.6 if sig=="多" else entry + row["ATR"]*0.6
            tp = entry + row["ATR"]*1.2 if sig=="多" else entry - row["ATR"]*1.2

            risk = capital * (risk_pct/100)
            qty = risk / abs(entry - sl)
            qty = max(round(qty, 2), 0.01)

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

        # 持仓管理
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
                exit_price = df.iloc[i]["close"]; reason="时间"

            if position["dir"]=="多":
                exit_price *= (1 - slippage/100)
            else:
                exit_price *= (1 + slippage/100)

            pnl = (exit_price - position["entry"]) * position["qty"] if position["dir"]=="多" else (position["entry"]-exit_price)*position["qty"]
            fee = exit_price * position["qty"] * 0.0005
            net = pnl - fee
            capital += net

            trades.append({
                "时间": position["open_time"],
                "方向": position["dir"],
                "盈亏": round(net, 2)
            })
            position = None

        equity.append(capital)

    # 绩效
    if trades:
        df_t = pd.DataFrame(trades)
        win = len(df_t[df_t["盈亏"]>0])
        total = len(df_t)
        win_rate = win/total*100 if total>0 else 0
        net = capital - 1000

        # 最大回撤
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = np.max(dd)*100

        return trades, win_rate, net, max_dd, equity
    return [], 0, 0, 0, equity

# ==========================
# 运行回测
# ==========================
with st.spinner("回测中..."):
    trades, win_rate, net, max_dd, equity = backtest(df)

st.subheader("📊 回测结果")
col1, col2, col3, col4 = st.columns(4)
col1.metric("交易次数", len(trades))
col2.metric("胜率", f"{win_rate:.1f}%")
col3.metric("净利润", f"{net:+.2f}")
col4.metric("最大回撤", f"{max_dd:.2f}%")

# 资金曲线
fig = go.Figure()
fig.add_trace(go.Scatter(y=equity, mode="lines"))
fig.update_layout(title="资金曲线")
st.plotly_chart(fig, use_container_width=True)

# 明细
if trades:
    st.dataframe(pd.DataFrame(trades).tail(20))
else:
    st.info("无交易")

# ==========================
# AI提示（实时）
# ==========================
st.subheader("🤖 AI进场提示（5分钟）")
latest = df.iloc[-1]
sig = ai_signal(latest)

if sig:
    st.success(f"建议方向：{sig}")
    st.write("理由：")
    st.write(f"- EMA趋势: {'上' if latest['EMA_fast']>latest['EMA_slow'] else '下'}")
    st.write(f"- ADX: {latest['ADX']:.1f}")
    st.write(f"- 回踩: {latest['close']:.2f}")
else:
    st.info("暂无明确进场")
