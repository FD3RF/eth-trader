import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import ta

st.set_page_config(layout="wide")
st.title("🔥 ETH 5分钟 AI 高频统计交易系统（工业级）")

# =========================
# 参数
# =========================
SYMBOL = "ETH-USDT-SWAP"
BAR = "5m"
LIMIT = 500

INITIAL_CAPITAL = 100
RISK_PER_TRADE = 0.02
RR = 1.4
FEE_RATE = 0.0005  # 单边手续费0.05%

# =========================
# 获取真实K线
# =========================
@st.cache_data(ttl=60)
def get_data():
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={BAR}&limit={LIMIT}"
    r = requests.get(url).json()
    data = r["data"]

    df = pd.DataFrame(data, columns=[
        "ts","open","high","low","close","vol",
        "volCcy","volCcyQuote","confirm"
    ])

    df = df.astype({
        "open":float,"high":float,"low":float,
        "close":float,"vol":float
    })

    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)

    return df

df = get_data()

# =========================
# 指标计算
# =========================
df["ema9"] = ta.trend.ema_indicator(df["close"], 9)
df["ema21"] = ta.trend.ema_indicator(df["close"], 21)
df["rsi"] = ta.momentum.rsi(df["close"], 14)

rolling_mean = df["close"].rolling(20).mean()
rolling_std = df["close"].rolling(20).std()
df["z"] = (df["close"] - rolling_mean) / rolling_std

vol_ma = df["vol"].rolling(20).mean()
df["vol_ratio"] = df["vol"] / vol_ma

# =========================
# 高频AI评分模型
# =========================
def score_row(row):
    score = 0

    if abs(row["z"]) > 1.2:
        score += 2
    elif abs(row["z"]) > 1:
        score += 1

    if row["rsi"] < 35 or row["rsi"] > 65:
        score += 2
    elif row["rsi"] < 45 or row["rsi"] > 55:
        score += 1

    if row["ema9"] > row["ema21"] or row["ema9"] < row["ema21"]:
        score += 1

    if row["vol_ratio"] > 1.2:
        score += 1

    return score

df["score"] = df.apply(score_row, axis=1)

# =========================
# 回测
# =========================
capital = INITIAL_CAPITAL
equity_curve = []
wins = 0
losses = 0
trade_log = []

for i in range(30, len(df)-2):

    row = df.iloc[i]
    next_open = df.iloc[i+1]["open"]

    if row["score"] < 4:
        equity_curve.append(capital)
        continue

    direction = None

    if row["z"] < -1.2 and row["rsi"] < 40:
        direction = "long"
    elif row["z"] > 1.2 and row["rsi"] > 60:
        direction = "short"

    if direction is None:
        equity_curve.append(capital)
        continue

    risk_amount = capital * RISK_PER_TRADE

    if direction == "long":
        sl = df.iloc[i-5:i]["low"].min()
        entry = next_open
        risk = entry - sl
        tp = entry + risk * RR
    else:
        sl = df.iloc[i-5:i]["high"].max()
        entry = next_open
        risk = sl - entry
        tp = entry - risk * RR

    if risk <= 0:
        equity_curve.append(capital)
        continue

    position_size = risk_amount / risk

    result = None

    for j in range(i+1, len(df)):
        high = df.iloc[j]["high"]
        low = df.iloc[j]["low"]

        if direction == "long":
            if low <= sl:
                result = -risk_amount
                losses += 1
                break
            if high >= tp:
                result = risk_amount * RR
                wins += 1
                break
        else:
            if high >= sl:
                result = -risk_amount
                losses += 1
                break
            if low <= tp:
                result = risk_amount * RR
                wins += 1
                break

    if result is None:
        equity_curve.append(capital)
        continue

    fee = capital * FEE_RATE * 2
    capital += result - fee

    trade_log.append({
        "time": row["ts"],
        "dir": direction,
        "score": row["score"],
        "pnl": result - fee,
        "capital": capital
    })

    equity_curve.append(capital)

# =========================
# 统计
# =========================
total_trades = wins + losses
winrate = wins / total_trades if total_trades > 0 else 0

if total_trades > 0:
    avg_win = np.mean([t["pnl"] for t in trade_log if t["pnl"] > 0])
    avg_loss = abs(np.mean([t["pnl"] for t in trade_log if t["pnl"] < 0]))
    expectancy = winrate * avg_win - (1-winrate) * avg_loss
else:
    expectancy = 0

max_drawdown = 0
peak = INITIAL_CAPITAL
for eq in equity_curve:
    peak = max(peak, eq)
    dd = (peak - eq)
    max_drawdown = max(max_drawdown, dd)

# =========================
# 显示
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("交易次数", total_trades)
col2.metric("胜率", f"{winrate*100:.2f}%")
col3.metric("真实期望值", round(expectancy,2))
col4.metric("最大回撤", round(max_drawdown,2))

# K线图
fig = go.Figure(data=[
    go.Candlestick(
        x=df["ts"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )
])

st.plotly_chart(fig, use_container_width=True)

# 资金曲线
st.subheader("资金曲线")
st.line_chart(equity_curve)

# 胜率曲线
if total_trades > 0:
    win_curve = np.cumsum([1 if t["pnl"]>0 else 0 for t in trade_log])
    trade_count_curve = np.arange(1, len(win_curve)+1)
    rolling_winrate = win_curve / trade_count_curve
    st.subheader("胜率曲线")
    st.line_chart(rolling_winrate)

# 日志
if trade_log:
    st.subheader("交易日志")
    st.dataframe(pd.DataFrame(trade_log))
