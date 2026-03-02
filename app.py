import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("ETH 高频监控 + 胜率统计 (轻量顶级)")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "signals_history.csv"

st_autorefresh(interval=10000, key="refresh")

# ========== 数据 ==========
def get_data(interval):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": interval, "limit": 300}
    r = requests.get(url, params=params)
    data = r.json()["data"]
    df = pd.DataFrame(data, columns=[
        "ts","open","high","low","close","volume",
        "volCcy","volCcyQuote","confirm"
    ])
    df = df.astype(float)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.sort_values("ts")

df = get_data("5m")
df_htf = get_data("1H")

# ========== 指标 ==========
df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)
df["VWAP"] = ta.volume.volume_weighted_average_price(
    df["high"], df["low"], df["close"], df["volume"])
df["RSI"] = ta.momentum.rsi(df["close"], 14)
df["ATR"] = ta.volatility.average_true_range(
    df["high"], df["low"], df["close"], 14)

df_htf["EMA60"] = ta.trend.ema_indicator(df_htf["close"], 60)

latest = df.iloc[-1]
htf = df_htf.iloc[-1]

# ========== 趋势 ==========
trend = "无趋势"
if htf["close"] > htf["EMA60"] and latest["EMA20"] > latest["EMA60"]:
    trend = "多头"
elif htf["close"] < htf["EMA60"] and latest["EMA20"] < latest["EMA60"]:
    trend = "空头"

# ========== 信号评分（高频宽松版） ==========
score = 0

if trend in ["多头", "空头"]:
    score += 2

near_vwap = abs(latest["close"] - latest["VWAP"]) < latest["ATR"] * 0.8
if near_vwap:
    score += 1

if not pd.isna(latest["RSI"]):
    if latest["RSI"] > 65 or latest["RSI"] < 35:
        score += 1

atr_mean = df["ATR"].rolling(20).mean().iloc[-1]
if not pd.isna(latest["ATR"]) and latest["ATR"] > atr_mean * 0.5:
    score += 1

entry = latest["close"]
if trend == "多头":
    stop = entry - latest["ATR"] * 1.2
    tp = entry + latest["ATR"] * 1.2 * 1.6
else:
    stop = entry + latest["ATR"] * 1.2
    tp = entry - latest["ATR"] * 1.2 * 1.6

risk = abs(entry - stop)
rr = round(abs((tp - entry) / risk), 2) if risk > 0 else 0

if rr >= 1.2:
    score += 2

signal = trend if score >= 4 else None

# ========== 交易提示（新增） ==========
trade_tip = "暂无建议"

if signal == "多头" and rr >= 1.2:
    trade_tip = "🚀 可考虑做多（高频提示）"
elif signal == "空头" and rr >= 1.2:
    trade_tip = "📉 可考虑做空（高频提示）"
else:
    trade_tip = "⛔ 当前不建议进场"

quality = "低"
if score >= 5 and rr >= 1.5:
    quality = "高"
elif score >= 4:
    quality = "中"

# ========== 胜率与历史 ==========
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["time","direction","score","rr","result"])

def save_history(row):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def calc_winrate():
    df = load_history()
    if len(df)==0:
        return 0,0
    wins = len(df[df["result"]=="WIN"])
    total = len(df[df["result"]!="NONE"])
    return (wins/total*100 if total>0 else 0), total

# ========== 记录信号 ==========
if signal:
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": signal,
        "score": score,
        "rr": round(rr,2),
        "result": "NONE"
    }
    save_history(row)

# ========== 图表 ==========
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], name="EMA20"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], name="EMA60"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["VWAP"], name="VWAP"))

st.plotly_chart(fig, use_container_width=True)

# ========== 面板 ==========
winrate, total = calc_winrate()

st.subheader("状态")
st.write("趋势:", trend)
st.write("评分:", score)
st.write("RSI:", round(latest["RSI"],2))
st.write("RR:", round(rr,2))
st.write("建议止损:", round(stop,4))
st.write("建议止盈:", round(tp,4))

# 交易提示
st.subheader("交易提示")
st.write("提示:", trade_tip)
st.write("质量:", quality)

st.subheader("胜率统计")
st.write("已统计信号:", total)
st.write("胜率:", f"{round(winrate,2)}%")

# 历史
st.subheader("信号历史")
st.dataframe(load_history().tail(20))

# 信号提示
if signal == "多头":
    st.success("📈 高质量做多")
elif signal == "空头":
    st.error("📉 高质量做空")
else:
    st.info("暂无高质量信号")
