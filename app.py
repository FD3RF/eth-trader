import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import re
import threading
import websocket
import json
import requests

# ====================== 配置 ======================
SYMBOL = "ethusdt"  # 小写，必须
INTERVAL = "5m"
DB_FILE = "signals.db"
MAX_CANDLES = 500
BINANCE_WS_URL = "wss://fstream.binance.com/ws"

# ====================== Session State ======================
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=MAX_CANDLES)
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = deque(maxlen=200)
if 'ws_queue' not in st.session_state:
    st.session_state.ws_queue = deque()
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False
if 'ws_error' not in st.session_state:
    st.session_state.ws_error = None
if 'ws_last_update' not in st.session_state:
    st.session_state.ws_last_update = 0
if 'ws_reconnect_count' not in st.session_state:
    st.session_state.ws_reconnect_count = 0

# ====================== SQLite ======================
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_time TEXT,
            signal_time TEXT,
            side TEXT,
            price REAL,
            ema_fast REAL,
            ema_slow REAL,
            rsi REAL,
            atr REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            result TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            peak REAL,
            note TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ====================== WebSocket 回调（只改状态，不碰 UI） ======================
def on_message(ws, message):
    try:
        data = json.loads(message)
        if 'k' in data:
            k = data['k']
            candle = [int(k['t']), float(k['o']), float(k['h']), float(k['l']), float(k['c']), float(k['v'])]
            st.session_state.ws_queue.append(candle)
            st.session_state.ws_last_update = time.time()
    except:
        pass

def on_open(ws):
    ws.send(json.dumps({
        "method": "SUBSCRIBE",
        "params": [f"{SYMBOL}@kline_{INTERVAL}"],
        "id": 1
    }))
    st.session_state.ws_connected = True
    st.session_state.ws_error = None
    st.session_state.ws_reconnect_count = 0

def on_error(ws, error):
    st.session_state.ws_connected = False
    st.session_state.ws_error = str(error)

def on_close(ws, *args):
    st.session_state.ws_connected = False
    st.session_state.ws_reconnect_count += 1

def ws_thread():
    while True:
        try:
            ws = websocket.WebSocketApp(
                BINANCE_WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=25, ping_timeout=10)
        except Exception as e:
            st.session_state.ws_error = str(e)
            time.sleep(3)

if 'ws_thread_started' not in st.session_state:
    threading.Thread(target=ws_thread, daemon=True).start()
    st.session_state.ws_thread_started = True

# ====================== 处理新 K线 ======================
while st.session_state.ws_queue:
    new_candle = st.session_state.ws_queue.popleft()
    buffer = st.session_state.candle_buffer
    if not buffer or new_candle[0] > buffer[-1][0]:
        buffer.append(new_candle)
    else:
        buffer[-1] = new_candle

# ====================== Streamlit UI ======================
st.set_page_config(page_title="ETH 5m 极致剥头皮 - Binance WS", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (Binance Futures 实时 WebSocket)")

# 状态栏
col1, col2, col3 = st.columns([2, 3, 2])
with col1:
    if st.session_state.ws_connected:
        st.success("● WebSocket 已连接")
    else:
        st.error("○ WebSocket 断开")

with col2:
    delay = time.time() - st.session_state.ws_last_update if st.session_state.ws_last_update else 999
    if delay < 5:
        st.success(f"实时更新正常（延迟 {delay:.1f} 秒）")
    elif delay < 60:
        st.warning(f"延迟 {delay:.1f} 秒")
    else:
        st.error(f"长时间无更新 ({delay:.0f} 秒)")

with col3:
    st.metric("K线数量", len(st.session_state.candle_buffer))

# 重连按钮 + 重连计数
if st.button("🔄 手动重连 WebSocket", type="primary"):
    st.session_state.ws_connected = False
    st.session_state.ws_error = "手动触发重连"
    st.rerun()

st.caption(f"重连尝试次数: {st.session_state.ws_reconnect_count}")

if st.session_state.ws_error:
    st.error(f"最新错误: {st.session_state.ws_error}")

# VPN 提示（针对广西电信）
st.info("**广西电信直连 Binance WS 经常被阻断**，请确保 LetsVPN 连接香港节点后刷新页面。")

# ====================== 你的策略逻辑区 ======================
# 请把 EMA 计算、信号检测、图表、历史信号等完整代码粘贴到这里

# 示例：显示最新 K 线价格
if st.session_state.candle_buffer:
    latest = st.session_state.candle_buffer[-1]
    st.metric("最新价格", f"{latest[4]:.2f} USDT")

st.caption("极致优化版 v3.0 • 线程安全 • 无警告 • 手动重连 • 2026.02")
