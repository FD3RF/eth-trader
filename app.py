import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import joblib

st.set_page_config(layout="wide")

exchange = ccxt.binance()

symbol = "ETH/USDT"

# =========================
# 获取K线
# =========================

def get_ohlcv(tf):

    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=200)

    df = pd.DataFrame(
        ohlcv,
        columns=["time","open","high","low","close","volume"]
    )

    df["time"] = pd.to_datetime(df["time"], unit="ms")

    return df


# =========================
# 指标计算
# =========================

def indicators(df):

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    df["vol_ma"] = df["volume"].rolling(20).mean()

    df["vol_ratio"] = df["volume"] / df["vol_ma"]

    df["price_change"] = df["close"].pct_change()

    return df


# =========================
# 三角背离
# =========================

def bottom_divergence(df):

    if len(df) < 5:
        return False

    p1 = df["low"].iloc[-3]
    p2 = df["low"].iloc[-1]

    v1 = df["volume"].iloc[-3]
    v2 = df["volume"].iloc[-1]

    return p2 < p1 and v2 > v1


def top_divergence(df):

    if len(df) < 5:
        return False

    p1 = df["high"].iloc[-3]
    p2 = df["high"].iloc[-1]

    v1 = df["volume"].iloc[-3]
    v2 = df["volume"].iloc[-1]

    return p2 > p1 and v2 < v1


# =========================
# AI评分
# =========================

def ai_score(df):

    try:

        model = joblib.load("eth_ai_model.pkl")

        last = df.iloc[-1]

        X = np.array([[
            last["price_change"],
            last["vol_ratio"],
            abs(last["close"]-last["ma20"])
        ]])

        pred = model.predict_proba(X)[0]

        bull = round(pred[1]*100,2)
        bear = round(pred[0]*100,2)

        score = int((bull-bear)+50)

        return score,bull,bear

    except:

        return 50,50,50


# =========================
# 主力行为识别
# =========================

def smart_money(df):

    last = df.iloc[-1]

    absorb = last["vol_ratio"] > 1.5 and abs(last["price_change"]) < 0.002

    pump = last["vol_ratio"] > 2 and last["price_change"] > 0.01

    dump = last["vol_ratio"] > 2 and last["price_change"] < -0.01

    fake = last["vol_ratio"] > 1.2 and abs(last["price_change"]) < 0.001

    return absorb,pump,dump,fake


# =========================
# 多周期共振
# =========================

def resonance():

    df1 = indicators(get_ohlcv("1m"))
    df5 = indicators(get_ohlcv("5m"))
    df15 = indicators(get_ohlcv("15m"))

    t1 = df1["ma20"].iloc[-1] > df1["ma50"].iloc[-1]
    t5 = df5["ma20"].iloc[-1] > df5["ma50"].iloc[-1]
    t15 = df15["ma20"].iloc[-1] > df15["ma50"].iloc[-1]

    score = t1 + t5 + t15

    if score == 3:
        return "多"
    if score == 0:
        return "空"
    return "mixed"


# =========================
# 胜率
# =========================

def winrate(df):

    wins = (df["price_change"] > 0).sum()
    total = len(df)

    return round(wins/total*100,2)


# =========================
# 画图
# =========================

def chart(df):

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["ma20"],
        name="MA20"
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["ma50"],
        name="MA50"
    ))

    return fig


# =========================
# 主逻辑
# =========================

def render():

    df = indicators(get_ohlcv("5m"))

    last = df.iloc[-1]

    price = last["close"]

    support = df["low"].tail(50).min()
    resistance = df["high"].tail(50).max()

    bottom = bottom_divergence(df)
    top = top_divergence(df)

    absorb,pump,dump,fake = smart_money(df)

    score,bull,bear = ai_score(df)

    res = resonance()

    trend = "up" if last["ma20"] > last["ma50"] else "down"

    strength = abs(last["ma20"]-last["ma50"])*100

    win = winrate(df)

    st.title("ETH AI 盯盘系统")

    st.metric("ETH价格", price)

    col1,col2,col3 = st.columns(3)

    col1.metric("支撑位", round(support,2))
    col2.metric("压力位", round(resistance,2))
    col3.metric("趋势强度", round(strength,2))

    st.plotly_chart(chart(df),use_container_width=True)

    st.subheader("AI策略播报")

    signals = []

    if bottom:
        signals.append("牛市共振 / 底背离")

    if top:
        signals.append("空头砸盘 / 顶背离")

    if absorb:
        signals.append("主力吸筹")

    if pump:
        signals.append("庄家拉升")

    if dump:
        signals.append("砸盘预警")

    if fake:
        signals.append("假突破")

    if last["vol_ratio"] > 1.5:
        signals.append("放量突破")

    if last["vol_ratio"] < 0.7:
        signals.append("缩量反弹")

    if score > 60:
        signals.append("准备做多")

    if score < 40:
        signals.append("准备做空")

    for s in signals:
        st.write("•",s)

    st.write("")

    st.write("多周期共振:",res)
    st.write("趋势方向:",trend)

    st.write("AI信号评分:",score,"/100")

    st.write("多头概率:",bull,"%")
    st.write("空头概率:",bear,"%")

    st.write("策略历史胜率:",win,"%")

render()
