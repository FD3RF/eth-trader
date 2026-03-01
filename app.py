# -*- coding: utf-8 -*-
"""
工程级看盘 + 回测 + 胜率（最终版本）
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ===========================
# 配置
# ===========================
SYMBOL = "ETH-USDT"
LIMIT = 300
INTERVAL = "5m"

# ===========================
# 交易日志
# ===========================
trade_log = []

def record_trade(entry, direction, result):
    trade_log.append({"entry": entry, "direction": direction, "result": result})

def win_rate():
    total = len(trade_log)
    return 0 if total == 0 else sum(1 for t in trade_log if t["result"] == "win") / total

# ===========================
# 数据获取
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
# 波段评分
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
# 自动交易条件
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
# 动态止损 / 止盈
# ===========================
def dynamic_levels(high, low):
    return (
        low * 0.997,  # 多止损
        low * 1.015,  # 多止盈
        high * 1.003, # 空止损
        high * 0.985  # 空止盈
    )

# ===========================
# 回测（历史胜率）
# ===========================
def backtest(df):
    wins = 0
    total = 0

    for i in range(20, len(df)):
        window = df.iloc[:i]
        signal = trade_condition(window)

        if signal == "wait":
            continue

        entry = window["c"].iloc[-1]
        high = window["h"].rolling(10).max().dropna().iloc[-1]
        low = window["l"].rolling(10).min().dropna().iloc[-1]
        sl_long, tp_long, sl_short, tp_short = dynamic_levels(high, low)

        future = df.iloc[i:i+50]

        if signal == "long":
            total += 1
            if any(future["c"] >= tp_long):
                wins += 1
            elif any(future["c"] <= sl_long):
                pass

        if signal == "short":
            total += 1
            if any(future["c"] <= tp_short):
                wins += 1
            elif any(future["c"] >= sl_short):
                pass

    return 0 if total == 0 else wins / total

# ===========================
# 主界面
# ===========================
def main():
    st.set_page_config(layout="wide")
    st.title("工程级看盘 + 回测终版")

    df = fetch_okx()
    if df is None or df.empty:
        st.warning("暂无数据")
        return

    # 回测
    bt_rate = backtest(df)
    st.subheader("回测胜率")
    st.write(f"{bt_rate * 100:.2f}%")

    # 实时
    score_5m = swing_score(df)
    score_1h = higher_tf_score()
    signal = trade_condition(df)

    st.subheader("实时判断")
    if signal == "long":
        st.success("做多")
    elif signal == "short":
        st.error("做空")
    else:
        st.info("观望")

    # 胜率统计
    result = None
    if signal != "wait":
        high = df["h"].rolling(10).max().dropna().iloc[-1]
        low = df["l"].rolling(10).min().dropna().iloc[-1]
        sl_long, tp_long, sl_short, tp_short = dynamic_levels(high, low)
        price = df["c"].iloc[-1]

        if signal == "long":
            result = "win" if price >= tp_long else "lose" if price <= sl_long else None
        if signal == "short":
            result = "win" if price <= tp_short else "lose" if price >= sl_short else None

        if result:
            record_trade(price, signal, result)

    st.subheader("胜率统计")
    st.write(f"总交易: {len(trade_log)}")
    st.write(f"胜率: {win_rate() * 100:.2f}%")

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

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 信息
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

if __name__ == "__main__":
    main()
