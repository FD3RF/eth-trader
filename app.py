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
st.title("ETH 高频盯盘 + 回测 + 胜率（终极完美版）")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "signals_history.csv"
LAST_SIGNAL_FILE = "last_signal.txt"

st_autorefresh(interval=10000, key="refresh")


# ==========================
# 数据获取（安全）
# ==========================
def get_data(interval="5m"):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": interval, "limit": 300}

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


df = get_data("5m")
df_15m = get_data("15m")

if df.empty:
    st.error("数据获取失败")
    st.stop()


# ==========================
# 指标计算
# ==========================
def add_indicators(df):
    df = df.copy()
    df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
    df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)
    df["RSI"] = ta.momentum.rsi(df["close"], 14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
    return df


df = add_indicators(df)
df_15m = add_indicators(df_15m)

df_valid = df.dropna()
if df_valid.empty:
    st.error("指标数据不足")
    st.stop()

latest = df_valid.iloc[-1]


# ==========================
# 多周期趋势（更稳）
# ==========================
def trend(df):
    if df.empty or df["EMA60"].isna().all():
        return 0
    return 1 if df["close"].iloc[-1] > df["EMA60"].iloc[-1] else -1


trend_5m = trend(df)
trend_15m = trend(df_15m)
trend_multi = trend_5m + trend_15m  # >0 多，<0 空


# ==========================
# 信号逻辑（低买高卖）
# ==========================
price = latest["close"]

mean = df["close"].rolling(20).mean().iloc[-1]
std = df["close"].rolling(20).std().iloc[-1]
z = (price - mean) / std if std > 0 else 0

is_oversold = z < -1.2
is_overbought = z > 1.2

vol_ok = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 0.7

atr = latest["ATR"] if not pd.isna(latest["ATR"]) else 0.01
stop_distance = max(atr * 1.2, price * 0.005)

if trend_multi > 0:
    stop = price - stop_distance
    tp = price + stop_distance * 1.6
else:
    stop = price + stop_distance
    tp = price - stop_distance * 1.6

risk = abs(price - stop)
rr = round(abs((tp - price) / risk), 2) if risk > 0 else 0

score = 0
if trend_multi > 0:
    score += 2
if abs(z) > 1.2:
    score += 1
if vol_ok:
    score += 1
if rr >= 1.3:
    score += 2

quality = "高" if score >= 7 and rr >= 1.6 else "中" if score >= 5 else "低"

signal = None
if trend_multi > 0 and is_oversold and vol_ok and rr >= 1.3:
    signal = "多单"
elif trend_multi < 0 and is_overbought and vol_ok and rr >= 1.3:
    signal = "空单"


# ==========================
# 交易成本模型（净期望）
# ==========================
FEE = 0.0004
SLIPPAGE = 0.0003

def net_profit(entry, exit):
    if entry <= 0:
        return 0
    gross = (exit - entry) / entry
    return gross - (FEE + SLIPPAGE)


# ==========================
# 回测与历史
# ==========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=[
        "time", "direction", "entry", "stop", "tp",
        "result", "tp_hit", "sl_hit",
        "score", "rr", "quality",
        "net", "high", "low"
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

    for idx, row in dfh.iterrows():
        if row["result"] in [None, ""]:
            if row["direction"] == "多单":
                if row["high"] >= row["tp"]:
                    dfh.at[idx, "tp_hit"] = True
                if row["low"] <= row["stop"]:
                    dfh.at[idx, "sl_hit"] = True
            else:
                if row["low"] <= row["tp"]:
                    dfh.at[idx, "tp_hit"] = True
                if row["high"] >= row["stop"]:
                    dfh.at[idx, "sl_hit"] = True

            if row.get("tp_hit"):
                dfh.at[idx, "result"] = "win"
            elif row.get("sl_hit"):
                dfh.at[idx, "result"] = "lose"

    dfh.to_csv(HISTORY_FILE, index=False)
    return dfh


def calc_metrics(dfh):
    if dfh.empty or "result" not in dfh:
        return 0, 0, 0

    completed = dfh[dfh["result"].notna()]
    total = len(completed)
    wins = (completed["result"] == "win").sum()

    win_rate = (wins / total * 100) if total > 0 else 0

    net = dfh["net"].dropna().sum() if "net" in dfh else 0

    cum = dfh["net"].cumsum() if "net" in dfh else pd.Series([])
    dd = (cum - cum.cummax()).min() if not cum.empty else 0

    return win_rate, net, dd


history = update_results()
winrate, net, dd = calc_metrics(history)


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


# ==========================
# 记录信号
# ==========================
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
        "quality": quality,
        "net": net_profit(price, tp),
        "high": df["high"].iloc[-1],
        "low": df["low"].iloc[-1]
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
st.subheader("状态")
st.write("多周期趋势:", "多头" if trend_multi > 0 else "空头" if trend_multi < 0 else "中性")
st.write("Z-score:", round(z, 2))
st.write("RSI:", round(latest["RSI"], 2))
st.write("RR:", round(rr, 2))
st.write("信号质量:", quality)

st.subheader("交易提示")
if signal == "多单":
    st.success(f"""
📈 多单
入场: {round(price,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {round(rr,2)}
""")
elif signal == "空单":
    st.error(f"""
📉 空单
入场: {round(price,4)}
止损: {round(stop,4)}
止盈: {round(tp,4)}
RR: {round(rr,2)}
""")
else:
    st.warning("暂无高质量信号")


# ==========================
# 回测统计
# ==========================
st.subheader("回测统计")
st.write("胜率:", f"{round(winrate,2)}%")
st.write("净收益:", round(net, 4))
st.write("最大回撤:", round(dd, 4))

st.subheader("信号历史")
st.dataframe(history.tail(20))


# ==========================
# 可视化
# ==========================
st.subheader("胜率曲线")
if not history.empty:
    h = history.copy()
    h["time"] = pd.to_datetime(h["time"])
    h["cum_win"] = (h["result"] == "win").cumsum()
    h["cum_total"] = range(1, len(h) + 1)
    h["win_rate_curve"] = h["cum_win"] / h["cum_total"] * 100

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=h["time"],
        y=h["win_rate_curve"],
        mode="lines",
        name="胜率曲线"
    ))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("分级统计")
    grade = h.groupby("quality").agg({
        "result": lambda x: (x == "win").sum(),
        "direction": "count"
    }).rename(columns={"direction": "total"})

    grade["win_rate"] = (grade["result"] / grade["total"]) * 100
    st.dataframe(grade)

    st.subheader("每日统计")
    h["day"] = h["time"].str[:10]
    stats = h.groupby("day").agg({
        "direction": "count",
        "result": lambda x: (x == "win").sum()
    })

    stats["win_rate"] = stats["result"] / stats["direction"] * 100
    st.dataframe(stats.tail(10))
