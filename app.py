import streamlit as st
import requests
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime

# ==========================
# 配置
# ==========================
st.set_page_config(layout="wide")
st.title("ETH 高频监控 + 胜率统计 (终极稳定版)")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "signals_history.csv"

# 15秒刷新（高频但不过载）
st_autorefresh(interval=15000, key="refresh")

# ==========================
# 数据获取（容错）
# ==========================
def get_data(interval):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": interval, "limit": 300}
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        if "data" not in j or not j["data"]:
            return pd.DataFrame()
        data = j["data"]
        df = pd.DataFrame(data, columns=[
            "ts","open","high","low","close","volume",
            "volCcy","volCcyQuote","confirm"
        ])
        df = df.astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df.sort_values("ts")
    except Exception:
        return pd.DataFrame()

df = get_data("5m")
df_htf = get_data("1H")

if df.empty or df_htf.empty:
    st.error("数据为空：接口/网络/限流")
    st.stop()

# ==========================
# 指标计算（安全）
# ==========================
def safe_indicators(df):
    if len(df) < 50:
        df["EMA20"] = np.nan
        df["EMA60"] = np.nan
        df["VWAP"] = np.nan
        df["RSI"] = np.nan
        df["ATR"] = np.nan
        return df

    df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
    df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)

    try:
        df["VWAP"] = ta.volume.volume_weighted_average_price(
            df["high"], df["low"], df["close"], df["volume"]
        )
    except Exception:
        df["VWAP"] = np.nan

    df["RSI"] = ta.momentum.rsi(df["close"], 14)
    df["ATR"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], 14
    )
    return df

df = safe_indicators(df)

def safe_indicators_htf(df):
    if len(df) < 60:
        df["EMA60"] = np.nan
    else:
        df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)
    return df

df_htf = safe_indicators_htf(df_htf)

latest = df.iloc[-1]
htf = df_htf.iloc[-1]

# ==========================
# 趋势判断
# ==========================
trend = "无趋势"
if latest["EMA20"] > latest["EMA60"] and htf["close"] > htf["EMA60"]:
    trend = "多头"
elif latest["EMA20"] < latest["EMA60"] and htf["close"] < htf["EMA60"]:
    trend = "空头"

# ==========================
# 信号评分
# ==========================
score = 0
if trend in ["多头", "空头"]:
    score += 2

near_vwap = not np.isnan(latest["VWAP"]) and abs(latest["close"] - latest["VWAP"]) < latest["ATR"] * 0.5
if near_vwap:
    score += 2

if not np.isnan(latest["RSI"]):
    if (trend == "多头" and latest["RSI"] > 50) or (trend == "空头" and latest["RSI"] < 50):
        score += 1

atr_mean = df["ATR"].rolling(20).mean().iloc[-1]
if not np.isnan(latest["ATR"]) and latest["ATR"] > atr_mean * 0.6:
    score += 1

entry = latest["close"]
if trend == "多头":
    stop = entry - latest["ATR"] * 1.2
    tp = entry + latest["ATR"] * 1.2 * 1.8
else:
    stop = entry + latest["ATR"] * 1.2
    tp = entry - latest["ATR"] * 1.2 * 1.8

risk = abs(entry - stop)
rr = round(abs((tp - entry) / risk), 2) if risk > 0 else 0

if rr >= 1.5:
    score += 2

signal = trend if score >= 6 else None

# ==========================
# 历史与胜率
# ==========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["time","direction","score","rr","result"])

def save_history(row):
    dfh = load_history()
    # 防止concat空警告（先过滤空行）
    dfh = dfh.dropna(how="all")
    dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
    dfh.to_csv(HISTORY_FILE, index=False)

def calc_winrate():
    dfh = load_history()
    if len(dfh) == 0:
        return 0, 0
    total = len(dfh[dfh["result"] != "NONE"])
    wins = len(dfh[dfh["result"] == "WIN"])
    return (wins / total * 100 if total > 0 else 0), total

dfh = load_history()
last_dir = dfh.iloc[-1]["direction"] if len(dfh) > 0 else None

# 只记录新方向
if signal and signal != last_dir:
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": signal,
        "score": score,
        "rr": rr,
        "result": "NONE"
    }
    save_history(row)

# ==========================
# 图表（新API）
# ==========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], name="EMA20"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], name="EMA60"))
if not df["VWAP"].isna().all():
    fig.add_trace(go.Scatter(x=df["ts"], y=df["VWAP"], name="VWAP"))

st.plotly_chart(fig, width="stretch")

# ==========================
# 状态面板
# ==========================
winrate, total = calc_winrate()

st.subheader("状态")
st.write("趋势:", trend)
st.write("评分:", score)
st.write("RSI:", round(latest["RSI"], 2) if not np.isnan(latest["RSI"]) else "N/A")
st.write("RR:", rr)
st.write("建议止损:", round(stop, 4))
st.write("建议止盈:", round(tp, 4))

st.subheader("胜率统计")
st.write("已统计信号:", total)
st.write("胜率:", f"{round(winrate,2)}%")

# 手动标记
st.subheader("标记最近信号结果")
col1, col2 = st.columns(2)
if col1.button("最近信号：赢"):
    dfh = load_history()
    if len(dfh) > 0:
        dfh.loc[dfh.index[-1], "result"] = "WIN"
        dfh.to_csv(HISTORY_FILE, index=False)
        st.success("已标记 WIN")

if col2.button("最近信号：亏"):
    dfh = load_history()
    if len(dfh) > 0:
        dfh.loc[dfh.index[-1], "result"] = "LOSS"
        dfh.to_csv(HISTORY_FILE, index=False)
        st.success("已标记 LOSS")

# 历史
st.subheader("信号历史")
st.dataframe(load_history().tail(20))

# 提示
if signal == "多头":
    st.success("📈 高质量做多")
elif signal == "空头":
    st.error("📉 高质量做空")
else:
    st.info("暂无高质量信号")
