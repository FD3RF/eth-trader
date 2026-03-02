import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🔥 ETH 5分钟 AI 高频统计交易系统（工业稳定版）")

SYMBOL = "ETH-USDT-SWAP"
BAR = "5m"
LIMIT = 500

INITIAL_CAPITAL = 100
RISK_PER_TRADE = 0.02
RR = 1.4
FEE_RATE = 0.0005


# =========================
# 获取K线
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
# 手写 EMA
# =========================
df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

# =========================
# 手写 RSI
# =========================
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

# =========================
# Z-score
# =========================
rolling_mean = df["close"].rolling(20).mean()
rolling_std = df["close"].rolling(20).std()
df["z"] = (df["close"] - rolling_mean) / rolling_std

vol_ma = df["vol"].rolling(20).mean()
df["vol_ratio"] = df["vol"] / vol_ma

# =========================
# 评分
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

st.metric("交易次数", total_trades)
st.metric("胜率", f"{winrate*100:.2f}%")

st.line_chart(equity_curve)

if trade_log:
    st.dataframe(pd.DataFrame(trade_log))
