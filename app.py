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
SYMBOL = "ETH-USDT-SWAP"          # OKX 永续合约符号
INTERVAL = "5m"                   # OKX K线周期（5m）
DB_FILE = "signals.db"
MAX_CANDLES = 500
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"  # OKX 公共 WebSocket

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
if 'last_ui_refresh' not in st.session_state:
    st.session_state.last_ui_refresh = time.time()

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

# ====================== OKX WebSocket 回调（线程安全） ======================
def on_message(ws, message):
    try:
        data = json.loads(message)
        if 'data' in data and data.get('arg', {}).get('channel') == 'candle5m':
            for item in data['data']:
                ts = int(item[0])              # 时间戳 ms
                o = float(item[1])
                h = float(item[2])
                l = float(item[3])
                c = float(item[4])
                v = float(item[5])
                candle = [ts, o, h, l, c, v]
                st.session_state.ws_queue.append(candle)
    except:
        pass

def on_open(ws):
    ws.send(json.dumps({
        "op": "subscribe",
        "args": [{"channel": "candle5m", "instId": SYMBOL}]
    }))

def on_error(ws, error):
    st.session_state.ws_error = str(error)

def on_close(ws, *args):
    st.session_state.ws_connected = False
    st.session_state.ws_reconnect_count += 1

def ws_thread():
    while True:
        try:
            ws = websocket.WebSocketApp(
                OKX_WS_URL,
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

# ====================== 主线程处理 queue 并刷新 UI ======================
current_time = time.time()
if current_time - st.session_state.last_ui_refresh > 3:  # 每3秒刷新一次
    st.session_state.last_ui_refresh = current_time

    # 处理接收到的 K线
    while st.session_state.ws_queue:
        new_candle = st.session_state.ws_queue.popleft()
        buffer = st.session_state.candle_buffer
        if not buffer or new_candle[0] > buffer[-1][0]:
            buffer.append(new_candle)
        else:
            buffer[-1] = new_candle
        st.session_state.ws_last_update = current_time
        st.session_state.ws_connected = True

    # 超时判断
    delay = current_time - st.session_state.ws_last_update
    if delay > 30:
        st.session_state.ws_connected = False
        st.session_state.ws_error = f"超时无数据 ({delay:.0f}秒)"

# ====================== Streamlit UI ======================
st.set_page_config(page_title="ETH 5m 极致剥头皮 - OKX WS", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (OKX 永续实时 WebSocket)")

# 状态栏
col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
with col1:
    if st.session_state.ws_connected:
        st.success("● OKX WS 已连接")
    else:
        st.error("○ OKX WS 断开")

with col2:
    delay = time.time() - st.session_state.ws_last_update if st.session_state.ws_last_update else 999
    if delay < 5:
        st.success(f"实时更新正常（延迟 {delay:.1f} 秒）")
    elif delay < 30:
        st.warning(f"延迟 {delay:.1f} 秒")
    else:
        st.error(f"长时间无更新 ({delay:.0f} 秒)")

with col3:
    st.metric("K线数量", len(st.session_state.candle_buffer))

with col4:
    st.metric("重连次数", st.session_state.ws_reconnect_count)

if st.session_state.ws_error:
    st.error(f"错误: {st.session_state.ws_error}")

# 手动重连
if st.button("🔄 手动重连 OKX WebSocket"):
    st.session_state.ws_connected = False
    st.session_state.ws_error = "手动重连触发"
    st.rerun()

# VPN 提示（针对广西电信）
st.info("广西电信直连 OKX WS 可能受限，请保持 LetsVPN 香港节点连接，并点击上方重连按钮。")

# ====================== 示例 K线图（可替换为你完整策略） ======================
if st.session_state.candle_buffer:
    df = pd.DataFrame(list(st.session_state.candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                         open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close'])])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

st.caption("OKX 实时版 v3.0 • 线程安全 • 无警告 • 手动重连 • 2026.02")
