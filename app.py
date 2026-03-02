import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("ETH 高频监控 + 胜率统计")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "signals_history.csv"
LAST_SIGNAL_FILE = "last_signal.txt"

st_autorefresh(interval=10000, key="refresh")


def get_data(interval):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": interval, "limit": 300}

    try:
        resp = requests.get(url, params=params, timeout=5)
        j = resp.json()
    except Exception:
        return pd.DataFrame()

    if "data" not in j or not j["data"]:
        return pd.DataFrame()

    df = pd.DataFrame(j["data"], columns=[
        "ts", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ])

    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df.sort_values("ts")


df = get_data("5m")
df_htf = get_data("1H")

if df.empty or df_htf.empty:
    st.error("数据获取失败")
    st.stop()


# 指标
df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)
df["VWAP"] = ta.volume.volume_weighted_average_price(
    df["high"], df["low"], df["close"], df["volume"])
df["RSI"] = ta.momentum.rsi(df["close"], 14)
df["ATR"] = ta.volatility.average_true_range(
    df["high"], df["low"], df["close"], 14)

df_htf["EMA60"] = ta.trend.ema_indicator(df_htf["close"], 60)

df_valid = df.dropna()
df_htf_valid = df_htf.dropna()

if df_valid.empty or df_htf_valid.empty:
    st.error("指标数据不足")
    st.stop()

latest = df_valid.iloc[-1]
htf = df_htf_valid.iloc[-1]


# 趋势
trend = "无趋势"
if (df["EMA20"].iloc[-3:] > df["EMA60"].iloc[-3:]).all() and htf["close"] > htf["EMA60"]:
    trend = "多头"
elif (df["EMA20"].iloc[-3:] < df["EMA60"].iloc[-3:]).all() and htf["close"] < htf["EMA60"]:
    trend = "空头"


# 评分
score = 0
if trend in ["多头", "空头"]:
    score += 2

near_vwap = abs(latest["close"] - latest["VWAP"]) < latest["ATR"] * 0.5
if near_vwap:
    score += 1

if not pd.isna(latest["RSI"]):
    if latest["RSI"] > 70 or latest["RSI"] < 30:
        score += 1

atr_mean = df["ATR"].rolling(20).mean().iloc[-1]
if not pd.isna(latest["ATR"]) and latest["ATR"] > atr_mean * 0.6:
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

signal = trend if score >= 5 and rr >= 1.5 else None


# 防重复
def already_recorded(signal, price):
    if not os.path.exists(LAST_SIGNAL_FILE):
        return False
    try:
        with open(LAST_SIGNAL_FILE) as f:
            last = f.read().split(",")
            return last[0] == signal and abs(float(last[1]) - price) < 0.001
    except:
        return False


def mark_record(signal, price):
    with open(LAST_SIGNAL_FILE, "w") as f:
        f.write(f"{signal},{price}")


# 胜率
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["time", "direction", "entry", "stop", "tp", "result"])


def save_history(row):
    dfh = load_history()
    dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
    dfh = dfh.tail(5000)
    dfh.to_csv(HISTORY_FILE, index=False)


def update_results():
    dfh = load_history()
    if dfh.empty:
        return dfh

    price = latest["close"]

    for idx, row in dfh.iterrows():
        if pd.isna(row.get("result")) or row["result"] == "":
            if row["direction"] == "多头":
                if price >= row["tp"]:
                    dfh.at[idx, "result"] = "win"
                elif price <= row["stop"]:
                    dfh.at[idx, "result"] = "lose"
            else:
                if price <= row["tp"]:
                    dfh.at[idx, "result"] = "win"
                elif price >= row["stop"]:
                    dfh.at[idx, "result"] = "lose"

    dfh.to_csv(HISTORY_FILE, index=False)
    return dfh


def calc_winrate():
    dfh = load_history()
    if dfh.empty or "result" not in dfh:
        return 0, 0

    total = dfh["result"].notna().sum()
    wins = (dfh["result"] == "win").sum()
    return (wins / total * 100) if total > 0 else 0, total


history = update_results()


# 记录信号
if signal and not already_recorded(signal, entry):
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": signal,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp": round(tp, 4),
        "result": ""
    }
    save_history(row)
    mark_record(signal, entry)


# 图表
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], name="EMA20"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], name="EMA60"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["VWAP"], name="VWAP"))

st.plotly_chart(fig, use_container_width=True)


# 面板
winrate, total = calc_winrate()

st.subheader("状态")
st.write("趋势:", trend)
st.write("评分:", score)
st.write("RSI:", round(latest["RSI"], 2))
st.write("RR:", round(rr, 2))
st.write("建议止损:", round(stop, 4))
st.write("建议止盈:", round(tp, 4))

quality = "高" if score >= 6 and rr >= 1.8 else "中" if score >= 5 else "低"

st.subheader("交易提示")
st.write("信号质量:", quality)

if signal == "多头":
    st.success(f"""
📈 做多信号
入场: {round(entry,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {round(rr,2)}
""")
    st.info("建议：等待回调靠近入场再执行；轻仓优先；确认成交量支持上涨")
elif signal == "空头":
    st.error(f"""
📉 做空信号
入场: {round(entry,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {round(rr,2)}
""")
    st.info("建议：顺势执行；避免追空；关注流动性与支撑位")
else:
    st.warning("暂无高质量信号")

st.subheader("胜率统计")
st.write("已统计信号:", total)
st.write("胜率:", f"{round(winrate,2)}%")

st.subheader("信号历史")
st.dataframe(history.tail(20))
