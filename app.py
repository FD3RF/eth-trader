import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime

# ==========================
# 页面
# ==========================
st.set_page_config(layout="wide")
st.title("ETH 高频盯盘 + 胜率统计（工业级）")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "signals_history.csv"
LAST_SIGNAL_FILE = "last_signal.txt"

st_autorefresh(interval=10000, key="refresh")


# ==========================
# 安全数据获取
# ==========================
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}

    try:
        resp = requests.get(url, params=params, timeout=5)
        j = resp.json()

        if not isinstance(j, dict) or "data" not in j or not j["data"]:
            return pd.DataFrame()

        df = pd.DataFrame(j["data"], columns=[
            "ts", "open", "high", "low", "close", "volume",
            "volCcy", "volCcyQuote", "confirm"
        ])

        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_values("ts")

    except Exception:
        return pd.DataFrame()


df = get_data()
if df.empty:
    st.error("数据获取失败")
    st.stop()


# ==========================
# 指标
# ==========================
df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)
df["RSI"] = ta.momentum.rsi(df["close"], 14)
df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)

df_valid = df.dropna()
if df_valid.empty:
    st.error("指标数据不足")
    st.stop()

latest = df_valid.iloc[-1]


# ==========================
# 信号逻辑（低买高卖）
# ==========================
price = latest["close"]

mean = df["close"].rolling(20).mean().iloc[-1]
std = df["close"].rolling(20).std().iloc[-1]
z = (price - mean) / std if std > 0 else 0

trend_up = price > df["close"].rolling(10).mean().iloc[-1]
trend_down = price < df["close"].rolling(10).mean().iloc[-1]

is_oversold = z < -1.2
is_overbought = z > 1.2

vol_ok = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 0.7

atr = latest["ATR"] if not pd.isna(latest["ATR"]) else 0.01
stop_distance = max(atr * 1.2, price * 0.005)

if trend_up:
    stop = price - stop_distance
    tp = price + stop_distance * 1.6
else:
    stop = price + stop_distance
    tp = price - stop_distance * 1.6

risk = abs(price - stop)
rr = round(abs((tp - price) / risk), 2) if risk > 0 else 0

score = 0
if trend_up or trend_down:
    score += 2
if abs(z) > 1.2:
    score += 1
if vol_ok:
    score += 1
if rr >= 1.3:
    score += 2

quality = "高" if score >= 6 and rr >= 1.6 else "中" if score >= 4 else "低"

signal = None
if trend_up and is_oversold and vol_ok and rr >= 1.3:
    signal = "多单"
elif trend_down and is_overbought and vol_ok and rr >= 1.3:
    signal = "空单"


# ==========================
# 胜率历史
# ==========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=[
        "time", "direction", "entry", "stop", "tp",
        "result", "tp_hit", "sl_hit", "score", "rr", "quality"
    ])


def save_history(row):
    dfh = load_history()
    dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
    dfh = dfh.tail(5000)
    dfh.to_csv(HISTORY_FILE, index=False)


def update_results():
    dfh = load_history()
    if dfh.empty:
        return dfh

    price = df["close"].iloc[-1]

    for idx, row in dfh.iterrows():
        if row["result"] in [None, ""]:
            if row["direction"] == "多单":
                if price >= row["tp"]:
                    dfh.at[idx, "tp_hit"] = True
                if price <= row["stop"]:
                    dfh.at[idx, "sl_hit"] = True
            else:
                if price <= row["tp"]:
                    dfh.at[idx, "tp_hit"] = True
                if price >= row["stop"]:
                    dfh.at[idx, "sl_hit"] = True

            if row.get("tp_hit"):
                dfh.at[idx, "result"] = "win"
            elif row.get("sl_hit"):
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


# ==========================
# 防重复记录
# ==========================
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


if signal and not already_recorded(signal, price):
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": signal,
        "entry": round(price, 4),
        "stop": round(stop, 4),
        "tp": round(tp, 4),
        "result": "",
        "tp_hit": False,
        "sl_hit": False,
        "score": score,
        "rr": rr,
        "quality": quality
    }
    save_history(row)
    mark_record(signal, price)


# ==========================
# 图表
# ==========================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"]
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA20"], name="EMA20"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], name="EMA60"))

st.plotly_chart(fig, use_container_width=True)


# ==========================
# 面板
# ==========================
winrate, total = calc_winrate()

st.subheader("状态")
st.write("趋势:", "多头" if trend_up else "空头" if trend_down else "无趋势")
st.write("Z-score:", round(z, 2))
st.write("RSI:", round(latest["RSI"], 2))
st.write("RR:", round(rr, 2))
st.write("建议止损:", round(stop, 4))
st.write("建议止盈:", round(tp, 4))
st.write("信号质量:", quality)

st.subheader("交易提示")
if signal == "多单":
    st.success(f"""
📈 多单信号
入场: {round(price,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {round(rr,2)}
""")
elif signal == "空单":
    st.error(f"""
📉 空单信号
入场: {round(price,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {round(rr,2)}
""")
else:
    st.warning("暂无高质量信号")

# ==========================
# 胜率与日志
# ==========================
st.subheader("胜率统计")
st.write("已统计信号:", total)
st.write("胜率:", f"{round(winrate,2)}%")

st.subheader("信号历史")
st.dataframe(history.tail(20))

# ==========================
# 日志可视化
# ==========================
st.subheader("日志可视化")

if not history.empty:
    history["time"] = pd.to_datetime(history["time"])

    history["cum_win"] = (history["result"] == "win").cumsum()
    history["cum_total"] = range(1, len(history) + 1)
    history["win_rate_curve"] = history["cum_win"] / history["cum_total"] * 100

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=history["time"],
        y=history["win_rate_curve"],
        mode="lines",
        name="胜率曲线"
    ))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("分级统计")
    grade = history.groupby("quality").agg({
        "result": lambda x: (x == "win").sum(),
        "direction": "count"
    }).rename(columns={"direction": "total"})

    grade["win_rate"] = (grade["result"] / grade["total"]) * 100
    st.dataframe(grade)

    st.subheader("每日统计（10–20单目标）")
    history["day"] = history["time"].str[:10]
    stats = history.groupby("day").agg({
        "direction": "count",
        "result": lambda x: (x == "win").sum()
    }).rename(columns={"direction": "signals", "result": "wins"})

    stats["win_rate"] = stats["wins"] / stats["signals"] * 100
    st.dataframe(stats.tail(10))
