import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime

# ==========================
# 基础配置
# ==========================
st.set_page_config(layout="wide")
st.title("顶级模型 | 高质量单（胜率优先）")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "history.csv"
LAST_SIGNAL_FILE = "last_signal.txt"

st_autorefresh(interval=8000, key="refresh")


# ==========================
# 数据获取（可靠降级）
# ==========================
def get_data():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}
    try:
        r = requests.get(url, timeout=5, params=params)
        j = r.json()
    except Exception:
        return pd.DataFrame()

    if "data" not in j:
        return pd.DataFrame()

    df = pd.DataFrame(j["data"], columns=[
        "ts","open","high","low","close","volume",
        "volCcy","volCcyQuote","confirm"
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("ts")


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
df = df.dropna()

latest = df.iloc[-1]

# ==========================
# 趋势
# ==========================
trend = "无"
if (df["EMA20"].iloc[-3:] > df["EMA60"].iloc[-3:]).all():
    trend = "多"
elif (df["EMA20"].iloc[-3:] < df["EMA60"].iloc[-3:]).all():
    trend = "空"

# ==========================
# 结构与回调（高胜率核心）
# ==========================
# 关键结构
res = df["high"].iloc[-20:].max()
sup = df["low"].iloc[-20:].min()

# 回调定义
def is_retest(price, level):
    return price <= level * 1.002 and price >= level * 0.998

# 假突破过滤
def big_shadow(row):
    body = abs(row["close"] - row["open"])
    upper = row["high"] - max(row["close"], row["open"])
    return upper > body * 1.2

if big_shadow(latest):
    structure_ok = False
else:
    structure_ok = True

# 动能
rsi = latest["RSI"]
rsi_ok = (45 <= rsi <= 65) if trend == "多" else (35 <= rsi <= 55)

# 盈亏比
entry = latest["close"]
atr = latest["ATR"]
stop = entry - atr if trend == "多" else entry + atr
tp = entry + atr * 2 if trend == "多" else entry - atr * 2
rr = abs((tp - entry) / (entry - stop)) if (entry - stop) != 0 else 0

# 成交量过滤
volume_avg = df["volume"].rolling(20).mean().iloc[-1]
volume_ok = latest["volume"] > volume_avg * 1.3

# 最终信号（极苛刻）
signal = None
if structure_ok and volume_ok and rsi_ok and rr >= 2 and is_retest(entry, res if trend=="多" else sup):
    signal = "多" if trend == "多" else "空"

quality = "高" if signal else "无"

# ==========================
# 防重复
# ==========================
def already(sig, price):
    if not os.path.exists(LAST_SIGNAL_FILE):
        return False
    try:
        with open(LAST_SIGNAL_FILE) as f:
            s = f.read().split(",")
            return s[0]==sig and abs(float(s[1])-price)<0.001
    except:
        return False

def mark(sig, price):
    with open(LAST_SIGNAL_FILE,"w") as f:
        f.write(f"{sig},{price}")

# ==========================
# 历史
# ==========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["time","direction","entry","stop","tp","result"])

def save(row):
    dfh = load_history()
    dfh = pd.concat([dfh,pd.DataFrame([row])],ignore_index=True)
    dfh = dfh.tail(5000)
    dfh.to_csv(HISTORY_FILE,index=False)

def update_results():
    dfh = load_history()
    if dfh.empty:
        return dfh
    price = latest["close"]
    for i,r in dfh.iterrows():
        if pd.isna(r.get("result")) or r["result"]=="":
            if r["direction"]=="多":
                if price>=r["tp"]: dfh.at[i,"result"]="win"
                elif price<=r["stop"]: dfh.at[i,"result"]="lose"
            else:
                if price<=r["tp"]: dfh.at[i,"result"]="win"
                elif price>=r["stop"]: dfh.at[i,"result"]="lose"
    dfh.to_csv(HISTORY_FILE,index=False)
    return dfh

history = update_results()

# ==========================
# 生成信号
# ==========================
if signal and not already(signal, entry):
    save({
        "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction":signal,
        "entry":round(entry,4),
        "stop":round(stop,4),
        "tp":round(tp,4),
        "result":""
    })
    mark(signal, entry)

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
st.write("趋势:", trend)
st.write("RSI:", round(rsi,2))
st.write("RR:", round(rr,2))
st.write("质量:", quality)

st.subheader("建议")
st.write("入场:", round(entry,4))
st.write("止损:", round(stop,4))
st.write("止盈:", round(tp,4))

if signal=="多":
    st.success("📈 高质量做多")
elif signal=="空":
    st.error("📉 高质量做空")
else:
    st.warning("暂无高质量机会")

st.subheader("历史")
st.dataframe(history.tail(20))
