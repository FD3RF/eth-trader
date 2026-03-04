import streamlit as st
import ccxt
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("5分钟以太坊合约K线策略（真实数据）")

# ========== 配置 ==========
symbol = "ETH/USDT"
timeframe = "5m"
limit = 200  # 获取最近200根K线

exchange = ccxt.binance()

def fetch_ohlcv():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def add_indicators(df):
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    return df

def generate_signals(df):
    df["signal"] = 0

    # 策略：突破 + 实体 + ATR过滤
    for i in range(1, len(df)):
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        atr = df["atr"].iloc[i] or 0

        if body > atr * 0.6 and df["close"].iloc[i] > df["high"].iloc[i-1]:
            df.loc[i, "signal"] = 1  # 多
        elif body > atr * 0.6 and df["close"].iloc[i] < df["low"].iloc[i-1]:
            df.loc[i, "signal"] = -1 # 空

    return df

def plot_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="K线"
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["ema20"],
        name="EMA20",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["ema50"],
        name="EMA50",
        mode="lines"
    ))

    # 信号标记
    buy = df[df["signal"] == 1]
    sell = df[df["signal"] == -1]

    fig.add_trace(go.Scatter(
        x=buy["time"],
        y=buy["low"] * 0.999,
        mode="markers",
        marker=dict(symbol="triangle-up", size=12),
        name="多信号"
    ))

    fig.add_trace(go.Scatter(
        x=sell["time"],
        y=sell["high"] * 1.001,
        mode="markers",
        marker=dict(symbol="triangle-down", size=12),
        name="空信号"
    ))

    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

# ========== 主逻辑 ==========
if st.button("刷新行情"):
    df = fetch_ohlcv()
    df = add_indicators(df)
    df = generate_signals(df)

    st.subheader("最新数据")
    st.dataframe(df.tail())

    # 策略统计
    total = len(df[df["signal"] != 0])
    buys = len(df[df["signal"] == 1])
    sells = len(df[df["signal"] == -1])

    st.write(f"信号总数: {total} | 多: {buys} | 空: {sells}")

    plot_chart(df)
else:
    st.info("点击『刷新行情』获取5分钟以太坊真实K线")
