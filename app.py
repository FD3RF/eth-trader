import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ==========================
# 配置
# ==========================
SYMBOL = "ETH-USDT"
INTERVAL = "5m"
HTF_INTERVAL = "1H"
LIMIT = 300
RISK_PER_TRADE = 0.008  # 0.8%
ATR_MULT_SL_BUFFER = 0.3

# ==========================
# 安全请求
# ==========================
def safe_request(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# ==========================
# K线获取
# ==========================
def get_candles(interval):
    url = f"https://www.okx.com/api/v5/market/candles?instId={SYMBOL}&bar={interval}&limit={LIMIT}"
    data = safe_request(url)
    if not data or data.get("code") != "0":
        return None

    df = pd.DataFrame(data["data"], columns=[
        "ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"
    ])[::-1]

    df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col])

    return df

# ==========================
# 真 ATR
# ==========================
def compute_atr(df, period=14):
    high = df["h"]
    low = df["l"]
    close = df["c"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(period).mean()
    return df

# ==========================
# 多空比
# ==========================
def get_ls_ratio():
    url = f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId={SYMBOL}&period=5m"
    data = safe_request(url)
    if data and data.get("code") == "0":
        return float(data["data"][0][1])
    return 1.0

# ==========================
# 扫单识别
# ==========================
def detect_sweep(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    vol_ma = df["v"].rolling(20).mean().iloc[-1]

    # 上扫
    sweep_high = (
        last["h"] > df["h"].iloc[-20:-1].max() and
        (last["h"] - max(last["o"], last["c"])) > (last["h"] - last["l"]) * 0.6 and
        last["v"] > vol_ma * 1.5 and
        prev["h"] < last["h"]
    )

    # 下扫
    sweep_low = (
        last["l"] < df["l"].iloc[-20:-1].min() and
        (min(last["o"], last["c"]) - last["l"]) > (last["h"] - last["l"]) * 0.6 and
        last["v"] > vol_ma * 1.5 and
        prev["l"] > last["l"]
    )

    return sweep_high, sweep_low

# ==========================
# 生成信号
# ==========================
def generate_signal(df, htf_df, ls_ratio, equity=10000):

    if df is None or htf_df is None:
        return None

    df = compute_atr(df)
    htf_df["ema20"] = htf_df["c"].ewm(span=20, adjust=False).mean()
    htf_df["ema50"] = htf_df["c"].ewm(span=50, adjust=False).mean()

    trend = 1 if htf_df.iloc[-1]["ema20"] > htf_df.iloc[-1]["ema50"] else -1

    sweep_high, sweep_low = detect_sweep(df)
    last = df.iloc[-1]
    atr = last["atr"]

    if atr is None or np.isnan(atr):
        return None

    direction = 0

    if sweep_high and trend == -1 and ls_ratio > 1.05:
        direction = -1
    elif sweep_low and trend == 1 and ls_ratio < 0.95:
        direction = 1

    if direction == 0:
        return None

    if direction == 1:
        entry = last["c"]
        sl = last["l"] - atr * ATR_MULT_SL_BUFFER
        tp1 = entry + (entry - sl)
        tp2 = entry + 2*(entry - sl)
        tp3 = entry + 3*(entry - sl)
    else:
        entry = last["c"]
        sl = last["h"] + atr * ATR_MULT_SL_BUFFER
        tp1 = entry - (sl - entry)
        tp2 = entry - 2*(sl - entry)
        tp3 = entry - 3*(sl - entry)

    risk_amount = equity * RISK_PER_TRADE
    position_size = risk_amount / abs(entry - sl)

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "size": position_size
    }

# ==========================
# 主程序
# ==========================
def main():

    st.set_page_config(layout="wide")
    st.title("ETH 5分钟 激进扫单高爆发系统")

    df = get_candles(INTERVAL)
    htf_df = get_candles(HTF_INTERVAL)
    ls_ratio = get_ls_ratio()

    if df is None or htf_df is None:
        st.error("数据获取失败")
        return

    signal = generate_signal(df, htf_df, ls_ratio)

    if signal:
        st.success("发现高爆发扫单机会")

        st.metric("方向", "做多" if signal["direction"] == 1 else "做空")
        st.metric("入场", round(signal["entry"],2))
        st.metric("止损", round(signal["sl"],2))
        st.metric("TP1", round(signal["tp1"],2))
        st.metric("TP2", round(signal["tp2"],2))
        st.metric("TP3", round(signal["tp3"],2))
        st.metric("建议仓位", round(signal["size"],4))
    else:
        st.info("无高质量扫单结构")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"]
    ))
    st.plotly_chart(fig, use_container_width=True)

    st_autorefresh(interval=15000)

if __name__ == "__main__":
    main()
