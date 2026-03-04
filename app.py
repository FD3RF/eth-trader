import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="突破+趋势+期望值终极版", page_icon="📈")
st.title("📊 突破 / 趋势 / 期望值 · 终极版")


# ========================= 获取K线 =========================
@st.cache_data(ttl=30)
def get_candles(limit=300, bar="15m"):
    try:
        res = requests.get(
            f"{BASE_URL}/api/v5/market/candles",
            params={"instId": "ETH-USDT", "limit": limit, "bar": bar},
            timeout=6
        ).json()

        if res.get("code") != "0":
            st.warning(res.get("msg", "获取失败"))
            return pd.DataFrame()

        df = pd.DataFrame(
            res["data"],
            columns=["ts", "o", "h", "l", "c", "v", "volCcy", "volCcyQuote", "confirm"]
        )[::-1]

        df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")

        # 强制数值转换（防异常）
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception:
        st.error("获取失败，请检查网络")
        return pd.DataFrame()


# ========================= 期望值 =========================
def expectancy(samples: pd.Series):
    if samples is None:
        return 0

    samples = samples.dropna()
    if len(samples) < 30:
        return 0

    win = samples[samples > 0]
    loss = samples[samples <= 0]

    if len(samples) == 0:
        return 0

    win_rate = len(win) / len(samples)
    avg_win = win.mean() if len(win) else 0
    avg_loss = abs(loss.mean()) if len(loss) else 0

    if avg_loss == 0:
        return 0

    rr = avg_win / avg_loss
    return (win_rate * rr) - (1 - win_rate)


# ========================= 主逻辑 =========================
def main():
    bar = st.selectbox("K线周期", ["5m", "15m", "1h"], index=1)
    lookback = st.slider("区间周期", 10, 100, 20)
    trend_fast = st.slider("快均线", 10, 50, 20)
    trend_slow = st.slider("慢均线", 30, 200, 50)
    volume_mult = st.slider("量能倍数", 1.2, 3.0, 1.5, 0.1)

    df = get_candles(limit=300, bar=bar)
    if df.empty:
        st.error("暂无数据")
        return

    # ===================== 滚动计算 =====================
    df["range_high"] = df["h"].rolling(lookback, min_periods=lookback).max()
    df["range_low"] = df["l"].rolling(lookback, min_periods=lookback).min()

    df["ma_fast"] = df["c"].rolling(trend_fast, min_periods=trend_fast).mean()
    df["ma_slow"] = df["c"].rolling(trend_slow, min_periods=trend_slow).mean()

    df["trend_up"] = df["ma_fast"] > df["ma_slow"]
    df["trend_down"] = df["ma_fast"] < df["ma_slow"]

    # 量能
    df["vol_ma"] = df["v"].rolling(lookback, min_periods=lookback).mean()
    df["vol_median"] = df["v"].rolling(lookback, min_periods=lookback).median()

    df["vol_break"] = (
        (df["v"] > df["vol_ma"] * volume_mult) &
        (df["v"] > df["vol_median"])
    )

    # 实体（防除零）
    df["body"] = (df["c"] - df["o"]).abs() / df["o"].replace(0, np.nan)

    # 原始突破
    df["raw_break_up"] = df["c"] > df["range_high"].shift(1)
    df["raw_break_down"] = df["c"] < df["range_low"].shift(1)

    # 有效突破（假突破过滤）
    df["valid_break_up"] = (
        df["raw_break_up"] &
        df["trend_up"] &
        df["vol_break"] &
        (df["body"] > 0.002)
    )

    df["valid_break_down"] = (
        df["raw_break_down"] &
        df["trend_down"] &
        df["vol_break"] &
        (df["body"] > 0.002)
    )

    # ===================== 期望值 =====================
    future = 3
    df["future_return"] = df["c"].shift(-future) / df["c"] - 1

    up_samples = df[df["valid_break_up"]]["future_return"]
    down_samples = df[df["valid_break_down"]]["future_return"]

    exp_up = expectancy(up_samples)
    exp_down = expectancy(down_samples)

    st.subheader("📈 期望值统计")
    st.write({
        "突破上期望(%)": round(exp_up * 100, 2),
        "突破下期望(%)": round(exp_down * 100, 2),
        "样本数(上)": len(up_samples.dropna()),
        "样本数(下)": len(down_samples.dropna())
    })

    if len(up_samples.dropna()) < 30 or len(down_samples.dropna()) < 30:
        st.warning("样本不足，期望值参考意义有限")

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
    fig.add_trace(go.Scatter(x=df["time"], y=df["ma_fast"], name="快均线"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ma_slow"], name="慢均线"))

    up = df[df["valid_break_up"]]
    down = df[df["valid_break_down"]]

    if not up.empty:
        fig.add_trace(go.Scatter(
            x=up["time"], y=up["c"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12),
            name="有效突破上"
        ))

    if not down.empty:
        fig.add_trace(go.Scatter(
            x=down["time"], y=down["c"],
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
            signals[["time", "c", "valid_break_up", "valid_break_down", "vol_break", "body"]],
            use_container_width=True
        )


if __name__ == "__main__":
    main()
