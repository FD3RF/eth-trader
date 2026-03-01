# -*- coding: utf-8 -*-
"""
工程级看盘终端 vFinal
功能：
- 实时K线
- FVG 标注
- 多周期方向
- 进场 / 止损 / 止盈
- 自动仓位计算
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
LIMIT = 200
INTERVAL = "5m"

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
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# ===========================
# FVG 检测
# ===========================
def detect_fvg(df):
    fvg = []
    for i in range(2, len(df)):
        prev = df.iloc[i-2]
        mid = df.iloc[i-1]
        cur = df.iloc[i]

        # Bull FVG（上涨缺口）
        if prev["h"] < cur["l"]:
            fvg.append({
                "type":"bull",
                "low": prev["h"],
                "high": cur["l"],
                "ts": cur["ts"]
            })

        # Bear FVG（下跌缺口）
        if prev["l"] > cur["h"]:
            fvg.append({
                "type":"bear",
                "low": cur["h"],
                "high": prev["l"],
                "ts": cur["ts"]
            })
    return fvg

# ===========================
# 扫单极值
# ===========================
def sweep_levels(df):
    highs = df["h"].rolling(10).max()
    lows = df["l"].rolling(10).min()
    return highs.dropna().iloc[-1], lows.dropna().iloc[-1]

# ===========================
# 波段评分
# ===========================
def swing_score(df):
    df["ema20"] = df["c"].ewm(span=20).mean()
    df["ema50"] = df["c"].ewm(span=50).mean()

    last = df.iloc[-1]
    score = 0

    if last["ema20"] > last["ema50"]:
        score += 2
    else:
        score -= 2

    if last["c"] > last["ema20"]:
        score += 1
    else:
        score -= 1

    return score

# ===========================
# 高周期方向（1H）
# ===========================
def higher_tf_score():
    df = fetch_okx(bar="1H", limit=100)
    if df is None or df.empty:
        return 0

    df["ema20"] = df["c"].ewm(span=20).mean()
    df["ema50"] = df["c"].ewm(span=50).mean()

    last = df.iloc[-1]
    return 1 if last["ema20"] > last["ema50"] else -1

# ===========================
# 仓位计算（风险 1%）
# ===========================
def calc_position_size(account, entry, stop):
    risk = account * 0.01
    distance = abs(entry - stop)
    if distance == 0:
        return 0
    return risk / distance

# ===========================
# 动态止损 / 止盈
# ===========================
def dynamic_levels(high, low):
    sl_long = low * 0.997
    sl_short = high * 1.003

    tp_long = low * 1.015
    tp_short = high * 0.985

    return sl_long, tp_long, sl_short, tp_short

# ===========================
# 主界面
# ===========================
def main():
    st.title("工程级看盘终端（最终整合版）")

    df = fetch_okx()
    if df is None or df.empty:
        st.warning("暂无数据")
        return

    # 关键计算
    high, low = sweep_levels(df)
    sl_long, tp_long, sl_short, tp_short = dynamic_levels(high, low)
    score_5m = swing_score(df)
    score_1h = higher_tf_score()

    # 多周期信号
    if score_1h > 0 and score_5m > 1:
        signal = "多周期一致：做多"
        signal_type = "success"
    elif score_1h < 0 and score_5m < -1:
        signal = "多周期一致：做空"
        signal_type = "error"
    else:
        signal = "周期分歧：观望"
        signal_type = "info"

    st.subheader("多周期信号")
    if signal_type == "success":
        st.success(signal)
    elif signal_type == "error":
        st.error(signal)
    else:
        st.info(signal)

    # K线图
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"]
    ))

    last_ts = df["ts"].iloc[-1]

    # FVG 标注
    fvg = detect_fvg(df)
    for gap in fvg:
        fig.add_shape(
            type="rect",
            x0=gap["ts"],
            x1=last_ts,
            y0=gap["low"],
            y1=gap["high"],
            fillcolor="rgba(0,255,0,0.12)" if gap["type"]=="bull" else "rgba(255,0,0,0.12)",
            line_width=0
        )

    # 进场标记
    fig.add_trace(go.Scatter(
        x=[last_ts],
        y=[low],
        mode="markers+text",
        text=["多单进场"],
        textposition="top center"
    ))

    fig.add_trace(go.Scatter(
        x=[last_ts],
        y=[high],
        mode="markers+text",
        text=["空单进场"],
        textposition="bottom center"
    ))

    # 止损 / 止盈
    fig.add_hline(y=sl_long, line_dash="dash", annotation_text="多单止损")
    fig.add_hline(y=tp_long, line_dash="dot", annotation_text="多单止盈")

    fig.add_hline(y=sl_short, line_dash="dash", annotation_text="空单止损")
    fig.add_hline(y=tp_short, line_dash="dot", annotation_text="空单止盈")

    st.plotly_chart(fig, use_container_width=True)

    # 仓位计算
    st.subheader("自动仓位计算（风险1%）")
    account = st.number_input("账户资金", value=10000)
    entry = df["c"].iloc[-1]
    stop = sl_long if score_5m > 1 else sl_short

    size = calc_position_size(account, entry, stop)
    st.write(f"建议仓位（合约价值）：{size:.2f}")

    # 关键价格
    st.subheader("关键价格")
    st.write(f"扫单高点: {high:.2f}")
    st.write(f"扫单低点: {low:.2f}")
    st.write(f"多单止损: {sl_long:.2f}")
    st.write(f"多单止盈: {tp_long:.2f}")
    st.write(f"空单止损: {sl_short:.2f}")
    st.write(f"空单止盈: {tp_short:.2f}")
    st.write(f"最新价: {entry:.2f}")

    st.subheader("波段评分")
    st.write(f"5m评分: {score_5m}")
    st.write(f"1h评分: {score_1h}")

    if score_5m > 1:
        st.success("5m做多倾向")
    elif score_5m < -1:
        st.error("5m做空倾向")
    else:
        st.info("5m震荡")

if __name__ == "__main__":
    main()
