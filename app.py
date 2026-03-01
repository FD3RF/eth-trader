# -*- coding: utf-8 -*-
"""
工程级 ETH/BTC 5分钟波段看盘分析终端
功能：
- 实时OKX数据
- K线图
- FVG
- 扫单
- 波段方向建议
- 支撑阻力
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
# 数据
# ===========================
def fetch_okx():
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
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
# FVG 计算
# ===========================
def detect_fvg(df):
    fvg = []
    for i in range(2, len(df)):
        prev = df.iloc[i-2]
        mid = df.iloc[i-1]
        cur = df.iloc[i]

        # Bull FVG
        if prev["h"] < cur["l"]:
            fvg.append({
                "type":"bull",
                "low": prev["h"],
                "high": cur["l"],
                "ts": cur["ts"]
            })
        # Bear FVG
        if prev["l"] > cur["h"]:
            fvg.append({
                "type":"bear",
                "low": cur["h"],
                "high": prev["l"],
                "ts": cur["ts"]
            })
    return fvg

# ===========================
# 扫单（简单极值）
# ===========================
def sweep_levels(df):
    highs = df["h"].rolling(10).max()
    lows = df["l"].rolling(10).min()
    return highs.dropna().iloc[-1], lows.dropna().iloc[-1]

# ===========================
# 波段方向评分
# ===========================
def swing_score(df):
    # EMA趋势
    df["ema20"] = df["c"].ewm(span=20).mean()
    df["ema50"] = df["c"].ewm(span=50).mean()

    last = df.iloc[-1]
    score = 0

    if last["ema20"] > last["ema50"]:
        score += 2
    else:
        score -= 2

    # 价格位置
    if last["c"] > last["ema20"]:
        score += 1
    else:
        score -= 1

    return score

# ===========================
# UI
# ===========================
def main():
    st.title("工程级 5分钟波段看盘终端")

    df = fetch_okx()
    if df is None or df.empty:
        st.warning("暂无数据")
        return

    # K线
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"]
    ))
    st.plotly_chart(fig, use_container_width=True)

    # FVG
    fvg = detect_fvg(df)
    st.subheader("FVG 数量")
    st.write(len(fvg))

    # 扫单
    high, low = sweep_levels(df)
    st.subheader("扫单极值")
    st.write(f"高点扫单: {high:.2f}")
    st.write(f"低点扫单: {low:.2f}")

    # 波段评分
    score = swing_score(df)
    st.subheader("波段方向评分")
    st.write(score)

    if score > 1:
        st.success("方向：做多倾向（回调可多）")
    elif score < -1:
        st.error("方向：做空倾向（反弹可空）")
    else:
        st.info("方向：震荡观望")

    # 最新价
    last = df.iloc[-1]
    st.write(f"最新价: {last['c']:.2f}")

if __name__ == "__main__":
    main()
