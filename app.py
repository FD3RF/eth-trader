import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ==========================
# 配置
# ==========================

REFRESH_MS = 5000
RISK_PER_TRADE = 0.005
MAX_LOSS_COUNT = 3
MAX_DAILY_LOSS = 0.02
IMBALANCE_THRESHOLD = 0.15

exchange = ccxt.okx({"enableRateLimit": True})

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

# ==========================
# Session 状态初始化
# ==========================

if "loss_count" not in st.session_state:
    st.session_state.loss_count = 0

if "daily_loss" not in st.session_state:
    st.session_state.daily_loss = 0

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

# ==========================
# 数据层
# ==========================

def fetch_ohlcv(symbol, tf="1m", limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except:
        return None

def orderbook_imbalance(symbol, depth=10):
    try:
        ob = exchange.fetch_order_book(symbol, limit=depth)
        bids = ob["bids"][:depth]
        asks = ob["asks"][:depth]
        bv = sum(b[1] for b in bids)
        av = sum(a[1] for a in asks)
        return (bv - av) / (bv + av) if (bv + av) > 0 else 0
    except:
        return 0

# ==========================
# 指标
# ==========================

def indicators(df):
    df["ema5"] = df["close"].ewm(5).mean()
    df["ema13"] = df["close"].ewm(13).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["range"] = df["high"] - df["low"]
    df["avg_range20"] = df["range"].rolling(20).mean()
    df["vol_avg10"] = df["volume"].rolling(10).mean()
    df["high_10"] = df["high"].rolling(10).max()

    return df

# ==========================
# 多周期评分
# ==========================

def multi_score(df1, df5):
    s = 0
    l1 = df1.iloc[-1]
    l5 = df5.iloc[-1]

    cond1 = (
        l1["range"] > l1["avg_range20"] * 1.5 and
        l1["ema5"] > l1["ema13"] and
        55 < l1["rsi"] < 80 and
        l1["volume"] > l1["vol_avg10"] * 1.8 and
        l1["close"] > df1["high_10"].iloc[-2]
    )

    if cond1:
        s += 1

    cond2 = (
        l5["ema5"] > l5["ema13"] and
        l5["rsi"] > 50
    )

    if cond2:
        s += 1

    return s

# ==========================
# 订单簿过滤
# ==========================

def pass_imbalance(symbol):
    return orderbook_imbalance(symbol) > IMBALANCE_THRESHOLD

# ==========================
# 熔断
# ==========================

def circuit_break(balance, atr_ratio):
    if st.session_state.loss_count >= MAX_LOSS_COUNT:
        return True, "连续亏损"

    if st.session_state.daily_loss >= balance * MAX_DAILY_LOSS:
        return True, "单日亏损"

    if atr_ratio > 0.02:
        return True, "波动异常"

    return False, ""

# ==========================
# 风控 + 仓位
# ==========================

def risk_model(df, balance):
    l = df.iloc[-1]
    entry = l["close"]

    stop = max(l["low"], entry - l["atr"] * 0.8)
    risk = entry - stop
    target = entry + risk * 1.5
    rr = (target - entry) / risk if risk > 0 else 0

    pos = (balance * RISK_PER_TRADE) / risk if risk > 0 else 0

    atr_ratio = l["atr"] / l["close"]
    if atr_ratio > 0.01:
        pos *= 0.5
    elif atr_ratio > 0.005:
        pos *= 0.7

    return entry, stop, target, rr, pos

# ==========================
# UI
# ==========================

def render(df, score, signal, balance):
    l = df.iloc[-1]

    st.subheader("状态")
    st.write(f"评分: {score}")

    col1, col2 = st.columns(2)
    col1.metric("价格", round(l["close"],2))
    col1.metric("RSI", round(l["rsi"],2))
    col1.metric("ATR", round(l["atr"],4))

    if signal:
        entry, stop, target, rr, pos = risk_model(df, balance)
        col2.metric("入场", round(entry,2))
        col2.metric("止损", round(stop,2))
        col2.metric("止盈", round(target,2))
        col2.metric("仓位", round(pos,4))
        col2.metric("盈亏比", round(rr,2))
    else:
        col2.write("等待信号")

    # K线
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"]
    ))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["ema5"], name="EMA5"))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["ema13"], name="EMA13"))

    if signal:
        fig.add_trace(go.Scatter(
            x=[df["ts"].iloc[-1]],
            y=[df["close"].iloc[-1]],
            mode="markers",
            marker=dict(size=12),
            name="信号"
        ))

    st.plotly_chart(fig, use_container_width=True)

# ==========================
# 主流程（无rerun刷屏）
# ==========================

def main():
    st.set_page_config(layout="wide")
    st.title("工程级监控（多指标 + 熔断）")

    symbol = st.sidebar.selectbox("交易对", SYMBOLS)
    balance = st.sidebar.number_input("余额", value=10000.0)

    df1 = fetch_ohlcv(symbol, "1m")
    df5 = fetch_ohlcv(symbol, "5m")

    if df1 is None or df5 is None:
        st.error("数据获取失败")
        return

    df1 = indicators(df1)
    df5 = indicators(df5)

    score = multi_score(df1, df5)
    imbalance_ok = pass_imbalance(symbol)

    signal = (score >= 2) and imbalance_ok

    atr_ratio = df1.iloc[-1]["atr"] / df1.iloc[-1]["close"]
    broken, reason = circuit_break(balance, atr_ratio)

    if broken:
        signal = False
        st.warning(f"熔断: {reason}")

    render(df1, score, signal, balance)

    # 自动刷新（避免日志刷屏）
    st_autorefresh(interval=REFRESH_MS)

if __name__ == "__main__":
    main()
