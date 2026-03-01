import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

SYMBOL = "ETH-USDT"
INTERVAL = "5m"
LIMIT = 100

def safe_request(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

def get_candles():
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    data = safe_request(url)
    if not data or data.get("code") != "0":
        return None

    df = pd.DataFrame(data["data"], columns=[
        "ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"
    ])[::-1]

    df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col])

    df["returns"] = df["c"].pct_change()

    # 指标
    df["ema_fast"] = df["c"].ewm(span=12).mean()
    df["ema_slow"] = df["c"].ewm(span=26).mean()

    delta = df["c"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["macd"] = df["c"].ewm(12).mean() - df["c"].ewm(26).mean()
    df["macd_hist"] = df["macd"] - df["macd"].ewm(9).mean()

    df["bb_mid"] = df["c"].rolling(20).mean()
    df["bb_std"] = df["c"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]

    df["vol_ma"] = df["v"].rolling(10).mean()
    df["atr"] = (df["h"] - df["l"]).rolling(14).mean()

    return df

def get_ls_ratio():
    url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
    data = safe_request(url)
    if data and data.get("code") == "0":
        return float(data["data"][0][1])
    return 1.0

def generate_signal(df, ls_ratio):
    if df is None or len(df) < 50:
        return 50, 0, "数据不足", None, None

    last = df.iloc[-1]
    score = 50
    reasons = []

    # 4H趋势简化：用EMA长短判断
    trend = 1 if last["ema_fast"] > last["ema_slow"] else -1

    # EMA
    if last["ema_fast"] > last["ema_slow"]:
        score += 20
        reasons.append("EMA多")
    else:
        score -= 18
        reasons.append("EMA空")

    # RSI
    rsi = last["rsi"]
    if 30 < rsi < 70:
        score += 10
    elif rsi > 75:
        score -= 10
    elif rsi < 25:
        score += 5

    # MACD
    if last["macd_hist"] > 0:
        score += 12
    else:
        score -= 12

    # 极点突破
    extreme = False
    if last["c"] > last["bb_upper"] and last["v"] > last["vol_ma"]*1.5:
        extreme = True
        score += 20
        reasons.append("突破上轨放量")
    elif last["c"] < last["bb_lower"] and last["v"] > last["vol_ma"]*1.5:
        extreme = True
        score += 20
        reasons.append("跌破下轨放量")

    # 多空比
    if ls_ratio < 0.95:
        score += 8
        reasons.append("多空极空")
    elif ls_ratio > 1.05:
        score -= 8
        reasons.append("多空极多")

    prob = max(min(score, 95), 5)

    # 方向
    direction = 1 if (trend == 1 and extreme and prob > 60) else \
                -1 if (trend == -1 and extreme and prob < 40) else 0

    atr = last["atr"]
    if direction == 1:
        sl = last["c"] - atr*1.5
        tp = last["c"] + atr*2.5
        entry = f"{last['c']-atr*0.5:.1f}~{last['c']+atr*0.5:.1f}"
    elif direction == -1:
        sl = last["c"] + atr*1.5
        tp = last["c"] - atr*2.5
        entry = f"{last['c']-atr*0.5:.1f}~{last['c']+atr*0.5:.1f}"
    else:
        sl = tp = None
        entry = "观望"

    reason = " | ".join(reasons) if reasons else "无明显信号"
    return prob, direction, entry, sl, tp

def main():
    st.set_page_config(layout="wide")
    st.title("5分钟 ETH 波段信号")

    df = get_candles()
    ls = get_ls_ratio()

    if df is None:
        st.error("无法获取数据")
        return

    prob, direction, entry, sl, tp = generate_signal(df, ls)

    st.metric("胜率", f"{prob:.1f}%")
    st.write("方向", "多" if direction==1 else "空" if direction==-1 else "观望")
    st.write("入场区", entry)
    st.write("止损", sl)
    st.write("止盈", tp)

    # K线
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["o"], high=df["h"], low=df["l"], close=df["c"]
    ))
    st.plotly_chart(fig, use_container_width=True)

    st_autorefresh(interval=15000)

if __name__ == "__main__":
    main()
