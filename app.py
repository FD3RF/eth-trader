import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ===================== 配置 =====================
SYMBOL = "ETH-USDT"
INTERVAL = "5m"
LIMIT = 200

# ===================== 数据 =====================
def safe_request(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json()
    except:
        return None

def get_candles():
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    data = safe_request(url)
    if not data or data.get("code") != "0":
        return None

    df = pd.DataFrame(data["data"], columns=["ts","o","h","l","c","v","x","y","z","w"])[::-1]
    df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df

# ===================== 指标 =====================
def add_indicators(df):
    # 趋势
    df["ema_fast"] = df["c"].ewm(span=12).mean()
    df["ema_slow"] = df["c"].ewm(span=26).mean()

    # RSI
    delta = df["c"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["c"].ewm(span=12).mean()
    ema26 = df["c"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_hist"] = df["macd"] - df["macd"].ewm(span=9).mean()

    # ATR（风险参考）
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - df["c"].shift()).abs(),
        (df["l"] - df["c"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    return df

# ===================== 分析逻辑（不建议下单）====================
def analyze(df):
    last = df.iloc[-1]
    score = 50
    reasons = []

    # 大趋势
    trend = 1 if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1] else -1
    if trend == 1:
        reasons.append("大趋势：上升")
    else:
        reasons.append("大趋势：下降")

    # 动量
    if last["rsi"] < 25:
        score += 10
        reasons.append("RSI超卖（动量可能回升）")
    elif last["rsi"] > 75:
        score -= 10
        reasons.append("RSI超买（动量可能回落）")

    if last["macd_hist"] > 0:
        score += 8
        reasons.append("MACD动量为正")
    else:
        score -= 8
        reasons.append("MACD动量为负")

    # 风险
    risk = last["atr"]
    reasons.append(f"波动参考(ATR)：{risk:.2f}")

    return {
        "score": max(min(score, 95), 5),
        "trend": trend,
        "reasons": reasons,
        "atr": risk
    }

# ===================== Streamlit =====================
def main():
    st.set_page_config(layout="wide")
    st.title("5分钟以太坊分析系统（仅分析）")

    df = get_candles()
    if df is None or len(df) < 50:
        st.error("数据不足")
        return

    add_indicators(df)
    result = analyze(df)
    last = df.iloc[-1]

    # 分析卡
    st.metric("分析得分", f"{result['score']:.1f}")
    st.write("分析理由:", result["reasons"])

    st.write("大趋势:", "📈上升" if result["trend"] == 1 else "📉下降")
    st.write("最新价格:", last["c"])
    st.write("RSI:", last["rsi"])
    st.write("ATR:", result["atr"])

    # 图表
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"]
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_fast"], name="EMA快"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_slow"], name="EMA慢"))

    st.plotly_chart(fig, use_container_width=True)

    st_autorefresh(interval=15000)

if __name__ == "__main__":
    main()
