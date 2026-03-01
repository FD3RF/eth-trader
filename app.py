import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import pandas_ta as ta
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# =========================
# 配置
# =========================
SYMBOL = "ETH-USDT"
INTERVAL = "5m"
LIMIT = 300

# OKX API
KLINE_URL = "https://www.okx.com/api/v5/market/candles"


# =========================
# 数据获取
# =========================
def fetch_kline(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT):
    params = {
        "instId": symbol,
        "bar": interval,
        "limit": str(limit)
    }
    try:
        r = requests.get(KLINE_URL, params=params, timeout=5)
        data = r.json()
        if data.get("code") != "0":
            return pd.DataFrame()

        rows = data["data"]
        df = pd.DataFrame(rows, columns=[
            "ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.sort_values("ts")
        df[["o", "h", "l", "c", "vol"]] = df[["o", "h", "l", "c", "vol"]].astype(float)
        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return pd.DataFrame()


# =========================
# 技术指标
# =========================
def add_indicators(df):
    if df.empty:
        return df

    df["ema20"] = ta.ema(df["c"], length=20)
    df["ema50"] = ta.ema(df["c"], length=50)
    df["rsi"] = ta.rsi(df["c"], length=14)
    df["atr"] = ta.atr(df["h"], df["l"], df["c"], length=14)

    return df


# =========================
# 策略信号（基础）
# =========================
def signal(df):
    if df.empty or len(df) < 50:
        return "NO SIGNAL"

    last = df.iloc[-1]

    if last["ema20"] > last["ema50"] and last["rsi"] < 70:
        return "LONG"
    if last["ema20"] < last["ema50"] and last["rsi"] > 30:
        return "SHORT"

    return "NO SIGNAL"


# =========================
# 绘图
# =========================
def plot_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["ts"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"]
    ))

    if "ema20" in df:
        fig.add_trace(go.Scatter(x=df["ts"], y=df["ema20"], name="EMA20"))
    if "ema50" in df:
        fig.add_trace(go.Scatter(x=df["ts"], y=df["ema50"], name="EMA50"))

    st.plotly_chart(fig, use_container_width=True)


# =========================
# 主界面
# =========================
def main():
    st.set_page_config(layout="wide", page_title="ETH Trader")

    st.title("ETH 交易监控")

    # 自动刷新
    st_autorefresh(interval=60 * 1000, limit=None)

    df = fetch_kline()
    if df.empty:
        st.warning("暂无数据")
        return

    df = add_indicators(df)
    sig = signal(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("最新价格", f"{df.iloc[-1]['c']:.2f}")
    with col2:
        st.metric("信号", sig)

    plot_chart(df)


if __name__ == "__main__":
    main()
