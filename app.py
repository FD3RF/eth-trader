import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="突破统计优化版", page_icon="📈")
st.title("📊 突破 / 趋势 / 期望值 · 工业统计版")


# ========================= 获取K线 =========================
@st.cache_data(ttl=30)
def get_candles(symbol="ETH-USDT", limit=400, bar="15m"):
    try:
        res = requests.get(
            f"{BASE_URL}/api/v5/market/candles",
            params={"instId": symbol, "limit": limit, "bar": bar},
            timeout=6
        ).json()

        if res.get("code") != "0":
            st.warning(res.get("msg", "获取失败"))
            return pd.DataFrame()

        df = pd.DataFrame(
            res["data"],
            columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"]
        )[::-1]

        df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")

        # 数值强转
        for col in ["o","h","l","c","v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception:
        st.error("获取失败，请检查网络")
        return pd.DataFrame()


# ========================= ATR =========================
def atr(df, period=14):
    high_low = df["h"] - df["l"]
    high_close = (df["h"] - df["c"].shift()).abs()
    low_close = (df["l"] - df["c"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ========================= 期望值 =========================
def expectancy(samples, cost=0.0006):
    samples = samples.dropna()
    if len(samples) < 30:
        return 0

    # 扣交易成本
    samples = samples - cost

    win = samples[samples > 0]
    loss = samples[samples <= 0]

    win_rate = len(win) / len(samples)
    avg_win = win.mean() if len(win) else 0
    avg_loss = abs(loss.mean()) if len(loss) else 0

    if avg_loss == 0:
        return 0

    rr = avg_win / avg_loss
    return (win_rate * rr) - (1 - win_rate)


# ========================= 趋势判断（EMA斜率）=========================
def trend(df, fast=20, slow=50):
    df["ema_fast"] = df["c"].ewm(span=fast).mean()
    df["ema_slow"] = df["c"].ewm(span=slow).mean()

    df["trend_up"] = df["ema_fast"] > df["ema_slow"]
    df["trend_down"] = df["ema_fast"] < df["ema_slow"]

    # 斜率（避免横盘假趋势）
    df["slope"] = df["ema_fast"].diff()
    df["trend_up"] &= df["slope"] > 0
    df["trend_down"] &= df["slope"] < 0

    return df


# ========================= 主逻辑 =========================
def main():
    symbol = st.selectbox("交易品种", ["ETH-USDT","BTC-USDT","SOL-USDT"])
    bar = st.selectbox("K线周期", ["5m","15m","1h"], index=1)
    lookback = st.slider("区间周期", 20, 120, 40)
    atr_mult = st.slider("ATR突破倍数", 1.0, 3.0, 1.8, 0.1)

    df = get_candles(symbol=symbol, limit=400, bar=bar)
    if df.empty:
        st.error("暂无数据")
        return

    # ===================== 指标计算 =====================
    df["range_high"] = df["h"].rolling(lookback).max()
    df["range_low"] = df["l"].rolling(lookback).min()

    df = trend(df)

    df["atr"] = atr(df, 14)
    df["body"] = (df["c"] - df["o"]).abs()

    # ===================== 突破定义 =====================
    df["raw_break_up"] = df["c"] > df["range_high"].shift(1)
    df["raw_break_down"] = df["c"] < df["range_low"].shift(1)

    # 放量过滤
    vol_threshold = df["v"].rolling(lookback).mean() * 1.5
    df["vol_break"] = df["v"] > vol_threshold

    # 有效突破
    df["valid_break_up"] = (
        df["raw_break_up"] &
        df["trend_up"] &
        df["vol_break"] &
        (df["body"] > df["atr"] * 0.4)
    )

    df["valid_break_down"] = (
        df["raw_break_down"] &
        df["trend_down"] &
        df["vol_break"] &
        (df["body"] > df["atr"] * 0.4)
    )

    # ===================== 动态出场（ATR止盈）====================
    # 前视修正：从突破信号K线的下一根收盘开始统计
    df["entry_price"] = df["c"].shift(-1)
    df["exit_price"] = df["entry_price"] + df["atr"] * 2  # 动态止盈

    df["return_5"] = (df["exit_price"] / df["entry_price"]) - 1
    df["return_10"] = (df["c"].shift(-10) / df["entry_price"]) - 1
    df["return_20"] = (df["c"].shift(-20) / df["entry_price"]) - 1

    # ===================== 期望值 =====================
    up = df[df["valid_break_up"]]["return_10"]
    down = df[df["valid_break_down"]]["return_10"]

    exp_up = expectancy(up)
    exp_down = expectancy(down)

    st.subheader("📈 多周期期望值（扣成本）")
    st.write({
        "突破上期望(10根%)": round(exp_up * 100, 2),
        "突破下期望(10根%)": round(exp_down * 100, 2),
        "样本数(上)": len(up.dropna()),
        "样本数(下)": len(down.dropna())
    })

    if len(up.dropna()) < 30 or len(down.dropna()) < 30:
        st.warning("样本不足，统计参考有限")

    # ===================== 可视化 =====================
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"],
        name="K线"
    ))

    fig.add_trace(go.Scatter(x=df["time"], y=df["range_high"], name="区间高"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["range_low"], name="区间低"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_fast"], name="EMA快"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema_slow"], name="EMA慢"))

    up_sig = df[df["valid_break_up"]]
    down_sig = df[df["valid_break_down"]]

    if not up_sig.empty:
        fig.add_trace(go.Scatter(
            x=up_sig["time"], y=up_sig["c"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12),
            name="有效突破上"
        ))

    if not down_sig.empty:
        fig.add_trace(go.Scatter(
            x=down_sig["time"], y=down_sig["c"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12),
            name="有效突破下"
        ))

    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ===================== 最近信号 =====================
    st.subheader("🔔 最近有效突破")
    signals = df[(df["valid_break_up"] | df["valid_break_down"])].tail(10)

    if signals.empty:
        st.info("暂无有效突破")
    else:
        st.dataframe(
            signals[["time","c","valid_break_up","valid_break_down","vol_break","body","return_10"]],
            use_container_width=True
        )


if __name__ == "__main__":
    main()
