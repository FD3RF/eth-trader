"""
OKX 24h 监控系统（单文件）
- WebSocket实时K线
- 多周期：5m/15m/1h/4h
- VWAP突破 + EMA9/21金叉 + 量能确认
- 收盘K线信号
- 信号冷却
- 断线重连
- Streamlit仪表盘
- 只监控不交易
"""

import streamlit as st
import pandas as pd
import pandas_ta as ta
import websocket
import json
import threading
import queue
import time
import random
from datetime import datetime

# ==========================
# 配置
# ==========================
SYMBOL = "BTC-USDT"
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
LIMIT = 200
REFRESH_INTERVAL = 10
SIGNAL_COOLDOWN = 60

WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# 数据与同步
data_store = {}
data_lock = threading.RLock()
kline_queue = queue.Queue(maxsize=2000)
ws_instance = None
ws_running = True
reconnect_attempt = 0
reconnect_log = []
reconnect_lock = threading.Lock()


# ==========================
# WebSocket
# ==========================
def start_ws():
    def on_message(ws, message):
        try:
            msg = json.loads(message)
            if "data" in msg and "arg" in msg:
                channel = msg["arg"].get("channel", "")
                if channel.startswith("candle"):
                    tf = channel.replace("candle", "")
                    for item in msg["data"]:
                        if len(item) < 6:
                            continue
                        ts = int(item[0])
                        o, h, l, c, v = map(float, item[1:6])
                        try:
                            kline_queue.put_nowait((tf, ts, o, h, l, c, v))
                        except queue.Full:
                            pass
        except Exception as e:
            print("[WS] parse error:", e)

    def on_open(ws):
        args = [{"channel": f"candle{tf}", "instId": SYMBOL} for tf in TIMEFRAMES]
        ws.send(json.dumps({"op": "subscribe", "args": args}))
        print("[WS] subscribed")

    def on_close(ws, code, msg):
        global reconnect_attempt
        with reconnect_lock:
            if ws_running:
                delay = min(30, 2 ** reconnect_attempt + random.random())
                reconnect_attempt += 1
                time.sleep(delay)
                print("[WS] reconnect...")
                start_ws()

    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
    )
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    return ws


# ==========================
# 数据处理
# ==========================
def processor():
    with data_lock:
        for tf in TIMEFRAMES:
            data_store[tf] = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    while ws_running:
        try:
            tf, ts, o, h, l, c, v = kline_queue.get(timeout=1)
        except queue.Empty:
            continue

        dt = pd.to_datetime(ts, unit="ms")
        with data_lock:
            df = data_store[tf]
            if not df.empty and df["timestamp"].iloc[-1] == dt:
                df.iloc[-1] = [dt, o, h, l, c, v]
            else:
                row = pd.DataFrame([[dt,o,h,l,c,v]], columns=df.columns)
                df = pd.concat([df,row], ignore_index=True)
                if len(df) > LIMIT:
                    df = df.iloc[-LIMIT:]
                data_store[tf] = df


# ==========================
# 指标与信号
# ==========================
def calc_indicators(df):
    if df.empty or len(df) < 30:
        return df.copy()
    df = df.copy()
    df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["EMA9"] = ta.ema(df["close"], length=9)
    df["EMA21"] = ta.ema(df["close"], length=21)
    df["VOL_MA5"] = df["volume"].rolling(5).mean().shift(1)
    return df


def check_signal(df, last_time):
    if df.empty or len(df) < 3:
        return None

    closed = df.iloc[-2]
    prev = df.iloc[-3]
    ktime = closed.name

    if last_time and (ktime - last_time).total_seconds() < SIGNAL_COOLDOWN:
        return None

    if pd.isna(closed["VWAP"]):
        return None

    cond1 = closed["close"] > closed["VWAP"]
    cond2 = closed["EMA9"] > closed["EMA21"] and prev["EMA9"] <= prev["EMA21"]
    cond3 = closed["volume"] > closed["VOL_MA5"]

    if cond1 and cond2 and cond3:
        entry = closed["close"]
        stop = closed["EMA21"]
        if stop >= entry:
            return None
        risk = entry - stop
        target = entry + 2 * risk
        return {
            "time": ktime,
            "entry": entry,
            "stop": stop,
            "target": target,
            "rr": round((target-entry)/(entry-stop), 2)
        }
    return None


# ==========================
# Streamlit
# ==========================
def main():
    st.set_page_config(layout="wide", page_title="OKX 监控")
    st.title("OKX 24h 监控")
    st.caption("只监控，不交易")

    placeholder = st.empty()
    last_signal = {tf: None for tf in TIMEFRAMES}
    signal_history = []

    while True:
        with placeholder.container():
            cols = st.columns(len(TIMEFRAMES))

            with data_lock:
                snapshot = {tf: df.copy() for tf, df in data_store.items()}

            for i, tf in enumerate(TIMEFRAMES):
                df = snapshot.get(tf)
                with cols[i]:
                    st.subheader(tf)
                    if df is None or len(df) < 30:
                        st.warning("数据收集中")
                        continue

                    df = calc_indicators(df)
                    last = df.iloc[-2]
                    price = df.iloc[-1]["close"]

                    st.metric("价格", f"{price:.2f}")

                    trend = "多头" if last["close"] > last["EMA21"] else "空头"
                    st.write("趋势:", trend)

                    cond1 = last["close"] > last["VWAP"]
                    cond2 = last["EMA9"] > last["EMA21"]
                    cond3 = last["volume"] > last["VOL_MA5"]

                    st.write(f"VWAP突破: {'🟢' if cond1 else '🔴'}")
                    st.write(f"EMA金叉: {'🟢' if cond2 else '🔴'}")
                    st.write(f"量能: {'🟢' if cond3 else '🔴'}")

                    sig = check_signal(df, last_signal.get(tf))
                    if sig:
                        last_signal[tf] = sig["time"]
                        signal_history.append({
                            "周期": tf,
                            "时间": sig["time"],
                            "入场": sig["entry"],
                            "止损": sig["stop"],
                            "目标": sig["target"],
                            "RR": sig["rr"]
                        })
                        if len(signal_history) > 20:
                            signal_history = signal_history[-20:]

                        st.success("买入信号")
                        st.write(f"入场: {sig['entry']:.2f}")
                        st.write(f"止损: {sig['stop']:.2f}")
                        st.write(f"目标: {sig['target']:.2f}")
                        st.progress(min(sig["rr"] / 5, 1.0))

            st.subheader("信号记录")
            if signal_history:
                st.dataframe(pd.DataFrame(signal_history))
            else:
                st.info("暂无信号")

            st.caption(f"刷新: {datetime.now()}")

        time.sleep(REFRESH_INTERVAL)


# ==========================
# 启动
# ==========================
if __name__ == "__main__":
    processor_thread = threading.Thread(target=processor, daemon=True)
    processor_thread.start()

    ws_instance = start_ws()

    try:
        main()
    except KeyboardInterrupt:
        ws_running = False
        if ws_instance:
            ws_instance.close()
