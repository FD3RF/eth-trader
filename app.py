# -*- coding: utf-8 -*-
"""
OKX UI 风格复刻看盘终端（工程级）
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np

# ===========================
# 配置
# ===========================
SYMBOL = "ETH-USDT"
LIMIT = 200
INTERVAL = "5m"

# ===========================
# 数据
# ===========================
def fetch_okx(bar=INTERVAL, limit=LIMIT):
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={bar}&limit={limit}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()["data"]
        df = pd.DataFrame(data, columns=[
            "ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"
        ])
        df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
        df[["o","h","l","c"]] = df[["o","h","l","c"]].astype(float)
        df = df.sort_values("ts")
        return df
    except:
        return None

# ===========================
# FVG
# ===========================
def detect_fvg(df):
    fvg = []
    for i in range(2, len(df)):
        prev = df.iloc[i-2]
        cur = df.iloc[i]

        if prev["h"] < cur["l"]:
            fvg.append({"type":"bull","low":prev["h"],"high":cur["l"],"ts":cur["ts"]})
        if prev["l"] > cur["h"]:
            fvg.append({"type":"bear","low":cur["h"],"high":prev["l"],"ts":cur["ts"]})
    return fvg

# ===========================
# 多周期
# ===========================
def higher_tf_score():
    df = fetch_okx(bar="1H", limit=100)
    if df is None or df.empty:
        return 0
    df["ema20"] = df["c"].ewm(span=20).mean()
    df["ema50"] = df["c"].ewm(span=50).mean()
    return 1 if df.iloc[-1]["ema20"] > df.iloc[-1]["ema50"] else -1

# ===========================
# 评分
# ===========================
def swing_score(df):
    df["ema20"] = df["c"].ewm(span=20).mean()
    df["ema50"] = df["c"].ewm(span=50).mean()
    last = df.iloc[-1]
    score = 0
    score += 2 if last["ema20"] > last["ema50"] else -2
    score += 1 if last["c"] > last["ema20"] else -1
    return score

# ===========================
# 自动判断
# ===========================
def trade_condition(df):
    score_5m = swing_score(df)
    score_1h = higher_tf_score()
    high = df["h"].rolling(10).max().dropna().iloc[-1]
    low = df["l"].rolling(10).min().dropna().iloc[-1]
    last = df["c"].iloc[-1]

    long = score_5m > 1 and score_1h > 0 and last <= low * 1.005
    short = score_5m < -1 and score_1h < 0 and last >= high * 0.995

    return "long" if long else "short" if short else "wait"

# ===========================
# 仓位
# ===========================
def calc_position_size(account, entry, stop):
    risk = account * 0.01
    distance = abs(entry - stop)
    return risk / distance if distance else 0

# ===========================
# UI
# ===========================
def main():
    st.set_page_config(
        page_title="OKX 看盘终端",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
    body { background-color: #0b0f19; }
    .stApp { background: #0b0f19; }
    </style>
    """, unsafe_allow_html=True)

    st.title("OKX 风格看盘终端")

    df = fetch_okx()
    if df is None or df.empty:
        st.warning("暂无数据")
        return

    score_5m = swing_score(df)
    score_1h = higher_tf_score()
    signal = trade_condition(df)

    # 信号
    if signal == "long":
        st.success("做多")
    elif signal == "short":
        st.error("做空")
    else:
        st.info("观望")

    # K线
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["ts"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ))

    # FVG
    last_ts = df["ts"].iloc[-1]
    for gap in detect_fvg(df):
        fig.add_shape(
            type="rect",
            x0=gap["ts"],
            x1=last_ts,
            y0=gap["low"],
            y1=gap["high"],
            fillcolor="rgba(0,255,0,0.12)" if gap["type"]=="bull" else "rgba(255,0,0,0.12)",
            line_width=0
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=20, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 信息
    st.markdown("---")
    st.subheader("多周期")
    if score_1h > 0 and score_5m > 1:
        st.success("多周期一致：做多")
    elif score_1h < 0 and score_5m < -1:
        st.error("多周期一致：做空")
    else:
        st.info("周期分歧")

    st.subheader("评分")
    st.write(f"5m: {score_5m}")
    st.write(f"1h: {score_1h}")

    st.subheader("最新价")
    st.write(f"{df['c'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
