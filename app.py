import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import os
import json
import threading
import time
from datetime import datetime
import websocket

st.set_page_config(layout="wide", page_title="ETH WS高频完美版")
st.title("🚀 ETH-USDT-SWAP 5分钟高频监控（WebSocket毫秒级·云端终极完美版）")

SYMBOL = "ETH-USDT-SWAP"
HISTORY_FILE = "signals_history.csv"

# ========================== Session State（云端稳定） ==========================
for key in ["df", "history", "ws_thread", "last_signal_time"]:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame() if key in ["df", "history"] else None
if "lock" not in st.session_state:
    st.session_state.lock = threading.Lock()

st_autorefresh(interval=800, key="cloud_refresh")

# ========================== 100元本金 + 开关 ==========================
st.sidebar.header("💰 100元本金模拟")
capital = st.sidebar.number_input("初始资金 (RMB)", value=100.0, min_value=50.0)
leverage = st.sidebar.slider("杠杆倍数", 10, 50, 20)
risk_percent = st.sidebar.slider("单笔风险%", 0.5, 5.0, 2.0) / 100
long_only = st.sidebar.checkbox("🔒 只做多单（100元强烈推荐）", value=True)

# ========================== WebSocket线程（云端完美） ==========================
def ws_thread_func():
    ws_url = "wss://ws.okx.com:8443/ws/v5/public"
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if "data" in data and data.get("arg", {}).get("channel") == "candle5m":
                candle = data["data"][0]
                ts = pd.to_datetime(int(candle[0]), unit="ms")
                row = {
                    "ts": ts, "open": float(candle[1]), "high": float(candle[2]),
                    "low": float(candle[3]), "close": float(candle[4]), "volume": float(candle[5])
                }
                with st.session_state.lock:
                    df = st.session_state.df
                    if not df.empty and df.iloc[-1]["ts"] == ts:
                        df.iloc[-1] = row
                    else:
                        st.session_state.df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).tail(300)
        except:
            pass
    def on_open(ws):
        ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "candle5m", "instId": SYMBOL}]}))
    while True:
        try:
            ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except:
            time.sleep(5)

if not st.session_state.ws_thread or not st.session_state.ws_thread.is_alive():
    st.session_state.ws_thread = threading.Thread(target=ws_thread_func, daemon=True)
    st.session_state.ws_thread.start()

if st.session_state.df.empty:
    st.warning("⏳ 连接OKX WebSocket中...（Cloud首次加载需3-5秒）")
    time.sleep(3)
    st.rerun()

df = st.session_state.df.copy()

# ========================== 指标计算 ==========================
def add_indicators(df):
    df = df.copy()
    df["EMA60"] = ta.trend.ema_indicator(df["close"], 60)
    df["RSI"] = ta.momentum.rsi(df["close"], 14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
    df["MACD"] = ta.trend.macd_diff(df["close"])
    df["MACD_signal"] = ta.trend.macd_signal(df["close"])
    df["BB_upper"] = ta.volatility.bollinger_hband(df["close"], window=20)
    df["BB_lower"] = ta.volatility.bollinger_lband(df["close"], window=20)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["close"]
    return df.dropna()

df = add_indicators(df)
if df.empty:
    st.error("数据不足，稍等几秒")
    st.stop()

latest = df.iloc[-1]
price = latest["close"]

# ========================== 信号逻辑（多空对称） ==========================
trend = 1 if price > latest["EMA60"] else -1
mean = df["close"].rolling(20).mean().iloc[-1]
std = df["close"].rolling(20).std().iloc[-1]
z = (price - mean) / std if std > 0 else 0
bb_squeeze = df["BB_width"].iloc[-1] < df["BB_width"].rolling(20).mean().iloc[-1] * 0.75
vol_ok = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.3
atr = latest["ATR"] if not pd.isna(latest["ATR"]) else price * 0.005
stop_distance = max(atr * 1.3, price * 0.006)

signal = None
direction = None
stop = tp = rr = 0.0
score = 0

if trend > 0:  # 多单
    macd_cross = (df["MACD"].iloc[-1] > df["MACD_signal"].iloc[-1]) and (df["MACD"].iloc[-2] <= df["MACD_signal"].iloc[-2])
    if z < -1.3 and bb_squeeze and macd_cross and vol_ok and latest["RSI"] < 38:
        stop = price - stop_distance
        tp = price + stop_distance * 1.8
        rr = round((tp - price) / stop_distance, 2)
        score = 10 if rr >= 1.8 else 8
        signal = "多单"
        direction = "多单"
elif not long_only and trend < 0:  # 空单
    macd_cross = (df["MACD"].iloc[-1] < df["MACD_signal"].iloc[-1]) and (df["MACD"].iloc[-2] >= df["MACD_signal"].iloc[-2])
    if z > 1.3 and bb_squeeze and macd_cross and vol_ok and latest["RSI"] > 62:
        stop = price + stop_distance
        tp = price - stop_distance * 1.8
        rr = round((price - tp) / stop_distance, 2)
        score = 10 if rr >= 1.8 else 8
        signal = "空单"
        direction = "空单"

quality = "⭐⭐⭐ 高" if score >= 9 else "⭐⭐ 中" if score >= 6 else "低"

# ========================== 仓位计算 ==========================
risk_amount = capital * risk_percent
contracts = int((risk_amount / stop_distance) * leverage * 0.01) if stop_distance > 0 else 0
margin_used = (price * contracts * 0.01) / leverage
liq_factor = 1.05 if direction == "多单" else 0.95
liquidation_price = round(price * (1 - (1 / leverage) * liq_factor), 2) if contracts else 0

# ========================== 历史记录 ==========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["time", "direction", "entry", "stop", "tp", "result", "score", "rr", "quality", "net", "signal_ts"])

history = load_history()

# ========================== TP/SL真实结算（胜率真实） ==========================
def update_results(history, df):
    if history.empty:
        return history
    for idx, row in history.iterrows():
        if pd.isna(row["result"]) or row["result"] == "":
            signal_ts = pd.to_datetime(row["signal_ts"])
            after = df[df["ts"] >= signal_ts]
            if after.empty:
                continue
            high = after["high"].max()
            low = after["low"].min()
            if row["direction"] == "多单":
                if high >= row["tp"]:
                    history.at[idx, "result"] = "win"
                elif low <= row["stop"]:
                    history.at[idx, "result"] = "lose"
            else:
                if low <= row["tp"]:
                    history.at[idx, "result"] = "win"
                elif high >= row["stop"]:
                    history.at[idx, "result"] = "lose"
    history.to_csv(HISTORY_FILE, index=False)
    return history

history = update_results(history, df)

# ========================== 记录信号（防重复） ==========================
if signal:
    current_time = datetime.now()
    if (st.session_state.last_signal_time is None or 
        (current_time - st.session_state.last_signal_time).total_seconds() > 60 or
        history.empty or history.iloc[-1]["entry"] != round(price, 4)):
        
        row = {
            "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction,
            "entry": round(price, 4),
            "stop": round(stop, 4),
            "tp": round(tp, 4),
            "result": "",
            "score": score,
            "rr": rr,
            "quality": quality,
            "net": round(abs((tp - price) / price - 0.0007), 4),
            "signal_ts": current_time
        }
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True).tail(5000)
        history.to_csv(HISTORY_FILE, index=False)
        st.session_state.last_signal_time = current_time
        st.session_state.history = history

# ========================== 统计 ==========================
completed = history[history["result"].notna()]
win_rate = round((completed["result"] == "win").mean() * 100, 2) if not completed.empty else 0.0
net_profit = round(history["net"].sum(), 4) if "net" in history else 0.0

# ========================== 图表 ==========================
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA60"], name="EMA60", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_upper"], name="BB上轨", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=df["ts"], y=df["BB_lower"], name="BB下轨", line=dict(dash="dash")))
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1: st.metric("当前价格", f"{price:.2f}", "🟢 WebSocket毫秒级")
with col2: st.metric("信号质量", quality)
with col3: st.metric("真实胜率", f"{win_rate}%")

st.subheader("📢 当前信号")
if signal:
    emoji = "🚀" if direction == "多单" else "📉"
    st.success(f"""
    {emoji} **{direction}信号**（质量 {quality}）
    入场: **{round(price,4)}**
    止损: {round(stop,4)}  
    止盈: {round(tp,4)}  
    RR: **{rr}**  
    建议仓位: **{contracts} 张** | 保证金 ≈{margin_used:.2f} USDT
    爆仓参考: **{liquidation_price}**
    """)
else:
    st.warning("⏳ 等待高质量信号...（当前模式：" + ("只多单" if long_only else "多空双向") + "）")

st.subheader("📊 统计")
st.write(f"**胜率**: {win_rate}%　|　**累计净收益**: {net_profit}　|　**总信号**: {len(history)}")

st.subheader("📜 最近信号")
st.dataframe(history.tail(15)[["time", "direction", "entry", "stop", "tp", "result", "quality", "rr"]], use_container_width=True)

if not history.empty:
    h = history.copy()
    h["time"] = pd.to_datetime(h["time"])
    completed_h = h[h["result"].notna()]
    if not completed_h.empty:
        completed_h["cum_win"] = (completed_h["result"] == "win").cumsum()
        completed_h["win_rate_curve"] = completed_h["cum_win"] / range(1, len(completed_h)+1) * 100
        fig2 = go.Figure(go.Scatter(x=completed_h["time"], y=completed_h["win_rate_curve"], mode="lines"))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("✅ **云端终极完美版** 已运行 | WebSocket实时 | 纯模拟监控 | 100元高频极易爆仓，实盘前必须OKX模拟盘测试30天！")
