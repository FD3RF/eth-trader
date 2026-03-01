# -*- coding: utf-8 -*-
"""
交易级分析终端（最终版本）
不自动交易，仅分析与决策
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
LIMIT = 500
INTERVAL = "5m"
RISK = 0.01  # 风险 1%

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
    df = fetch_okx(bar="1H", limit=200)
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
# 自动交易条件（仅分析）
# ===========================
def trade_condition(df):
    score_5m = swing_score(df)
    score_1h = higher_tf_score()
    high = df["h"].rolling(10).max().dropna().iloc[-1]
    low = df["l"].rolling(10).min().dropna().iloc[-1]
    last = df["c"].iloc[-1]

    long = score_5m > 1 and score_1h > 0 and last <= low * 1.005
    short = score_5m < -1 and score_1h < 0 and last >= high * 0.995

    if long:
        return "long"
    if short:
        return "short"
    return "wait"

# ===========================
# 动态止损 / 止盈
# ===========================
def dynamic_levels(high, low):
    sl_long = low * 0.997
    tp_long = low * 1.015
    sl_short = high * 1.003
    tp_short = high * 0.985
    return sl_long, tp_long, sl_short, tp_short

# ===========================
# 仓位建议
# ===========================
def calc_position_size(account, entry, stop):
    risk = account * RISK
    distance = abs(entry - stop)
    return risk / distance if distance else 0

# ===========================
# 回测
# ===========================
def backtest(df):
    wins = 0
    total = 0

    for i in range(30, len(df)):
        window = df.iloc[:i]
        signal = trade_condition(window)
        if signal == "wait":
            continue

        high = window["h"].rolling(10).max().dropna().iloc[-1]
        low = window["l"].rolling(10).min().dropna().iloc[-1]
        sl_long, tp_long, sl_short, tp_short = dynamic_levels(high, low)
        future = df.iloc[i:i+50]

        total += 1
        if signal == "long" and any(future["c"] >= tp_long):
            wins += 1
        if signal == "short" and any(future["c"] <= tp_short):
            wins += 1

    return 0 if total == 0 else wins / total

# ===========================
# 主界面
# ===========================
def main():
    st.set_page_config(layout="wide")
    st.title("交易级分析终端")

    df = fetch_okx()
    if df is None or df.empty:
        st.warning("暂无数据")
        return

    # 回测
    bt_rate = backtest(df)
    st.subheader("回测胜率")
    st.write(f"{bt_rate * 100:.2f}%")

    # 实时判断
    score_5m = swing_score(df)
    score_1h = higher_tf_score()
    signal = trade_condition(df)

    st.subheader("实时判断")
    st.write(signal)

    # 多周期
    st.subheader("多周期")
    if score_1h > 0 and score_5m > 1:
        st.success("做多倾向")
    elif score_1h < 0 and score_5m < -1:
        st.error("做空倾向")
    else:
        st.info("观望")

    # 关键价格
    high = df["h"].rolling(10).max().dropna().iloc[-1]
    low = df["l"].rolling(10).min().dropna().iloc[-1]
    sl_long, tp_long, sl_short, tp_short = dynamic_levels(high, low)
    price = df["c"].iloc[-1]

    st.subheader("关键价格")
    st.write(f"止损多: {sl_long:.2f}")
    st.write(f"止盈多: {tp_long:.2f}")
    st.write(f"止损空: {sl_short:.2f}")
    st.write(f"止盈空: {tp_short:.2f}")
    st.write(f"最新价: {price:.2f}")

    # 胜率统计
    result = None
    if signal != "wait":
        result = (
            "win" if (signal == "long" and price >= tp_long) or
                   (signal == "short" and price <= tp_short)
            else "lose"
        )
        record_trade(price, signal, result)

    st.subheader("胜率")
    st.write(f"{win_rate() * 100:.2f}%")

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

if __name__ == "__main__":
    main()
