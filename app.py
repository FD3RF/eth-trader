import streamlit as st
import pandas as pd
import requests
import ta
import plotly.graph_objects as go

SYMBOL = "ETH-USDT-SWAP"
BAR = "5m"

FEE = 0.0006
SLIPPAGE = 0.0003
BASE_RISK = 0.01
REDUCED_RISK = 0.005
MAX_LEVERAGE_CAP = 5
MAX_CONSEC_LOSS = 3
PAUSE_BARS = 12

st.set_page_config(layout="wide")
st.title("工程级小利润策略 V2")

# =========================
# 数据
# =========================
@st.cache_data(ttl=30)
def get_data():
    r = requests.get(
        "https://www.okx.com/api/v5/market/candles",
        params={"instId": SYMBOL, "bar": BAR, "limit": 800},
        timeout=5
    )
    data = r.json()["data"]

    df = pd.DataFrame(data, columns=[
        "ts","open","high","low","close",
        "volume","v1","v2","confirm"
    ])

    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")

    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)

    df = df.sort_values("ts").reset_index(drop=True)
    return df


# =========================
# 指标
# =========================
def add_indicators(df):
    df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
    df["EMA50"] = ta.trend.ema_indicator(df["close"], 50)
    df["ATR"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], 14
    )
    df["ATR_mean"] = df["ATR"].rolling(20).mean()
    df["ADX"] = ta.trend.adx(
        df["high"], df["low"], df["close"], 14
    )
    return df.dropna()


# =========================
# 信号
# =========================
def signal(df, i):
    if i < 60:
        return False

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    trend = (
        row["EMA20"] > row["EMA50"]
        and (row["EMA20"] - prev["EMA20"]) > 0
        and row["ADX"] > 22
    )

    pullback = (
        prev["low"] < prev["EMA20"]
        and row["close"] > row["EMA20"]
    )

    volatility = row["ATR"] > row["ATR_mean"]

    body_ok = (
        row["close"] > row["open"]
        and abs(row["close"] - row["open"]) > row["ATR"] * 0.2
    )

    return trend and pullback and volatility and body_ok


# =========================
# 回测
# =========================
def backtest(df):

    balance = 1000
    position = None
    trades = []
    consecutive_loss = 0
    pause_until = -1

    for i in range(len(df) - 1):

        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # ======================
        # 平仓逻辑
        # ======================
        if position:

            hit_sl = row["low"] <= position["sl"]
            hit_tp = row["high"] >= position["tp"]

            if hit_sl and hit_tp:
                exit_price = position["sl"]  # 最坏情况
            elif hit_sl:
                exit_price = position["sl"]
            elif hit_tp:
                exit_price = position["tp"]
            else:
                exit_price = None

            if exit_price:

                pnl = (exit_price - position["entry"]) * position["qty"]

                fee_cost = (
                    abs(position["entry"] * position["qty"]) +
                    abs(exit_price * position["qty"])
                ) * FEE

                pnl -= fee_cost
                balance += pnl
                trades.append(pnl)

                if pnl < 0:
                    consecutive_loss += 1
                else:
                    consecutive_loss = 0

                if consecutive_loss >= MAX_CONSEC_LOSS:
                    pause_until = i + PAUSE_BARS
                    consecutive_loss = 0

                position = None

        # ======================
        # 开仓逻辑
        # ======================
        if i < pause_until:
            continue

        if not position and signal(df, i):

            entry = next_row["open"] * (1 + SLIPPAGE)

            atr = row["ATR"]
            sl = entry - atr * 0.6
            tp = entry + atr * 0.8

            risk_percent = BASE_RISK if consecutive_loss == 0 else REDUCED_RISK

            risk_amount = balance * risk_percent
            stop_distance = abs(entry - sl)

            qty = risk_amount / stop_distance

            max_qty = (balance * MAX_LEVERAGE_CAP) / entry
            qty = min(qty, max_qty)

            position = {
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty
            }

    win = len([t for t in trades if t > 0])
    total = len(trades)
    win_rate = win / total * 100 if total else 0

    return balance, total, win_rate, trades


# =========================
# 主流程
# =========================
df = get_data()
df = add_indicators(df)

fig = go.Figure(data=[go.Candlestick(
    x=df["ts"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])
st.plotly_chart(fig, use_container_width=True)

if st.button("运行升级版回测"):

    bal, trades, wr, trade_list = backtest(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("最终资金", f"{bal:.2f}")
    col2.metric("交易次数", trades)
    col3.metric("胜率", f"{wr:.1f}%")

    st.line_chart(pd.Series(trade_list).cumsum())
