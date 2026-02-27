import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import websocket  # 仍需此包，但不使用线程

# ====================== 配置 ======================
SYMBOL = "ETH-USDT-SWAP"
INTERVAL = "5m"
DB_FILE = "signals.db"
MAX_CANDLES = 500
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# ====================== Session State ======================
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=MAX_CANDLES)
if 'ws_last_update' not in st.session_state:
    st.session_state.ws_last_update = 0
if 'ws_error' not in st.session_state:
    st.session_state.ws_error = None
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False

# ====================== SQLite ======================
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_time TEXT, signal_time TEXT, side TEXT, price REAL,
            ema_fast REAL, ema_slow REAL, rsi REAL, atr REAL,
            sl REAL, tp1 REAL, tp2 REAL, result TEXT,
            exit_price REAL, exit_time TEXT, exit_reason TEXT,
            peak REAL, note TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ====================== OKX WebSocket 非线程版 ======================
# 使用 websocket.create_connection 在主线程非阻塞连接
@st.cache_resource
def get_ws_connection():
    try:
        ws = websocket.create_connection(OKX_WS_URL, timeout=10)
        ws.send(json.dumps({
            "op": "subscribe",
            "args": [{"channel": "candle5m", "instId": SYMBOL}]
        }))
        st.session_state.ws_connected = True
        return ws
    except Exception as e:
        st.session_state.ws_error = str(e)
        return None

# 主线程轮询接收数据（每 rerun 一次检查）
ws = get_ws_connection()
if ws:
    try:
        while ws.connected and ws.recv_ready():
            message = ws.recv()
            data = json.loads(message)
            if 'data' in data and data.get('arg', {}).get('channel') == 'candle5m':
                for item in data['data']:
                    ts = int(item[0])
                    o = float(item[1])
                    h = float(item[2])
                    l = float(item[3])
                    c = float(item[4])
                    v = float(item[5])
                    candle = [ts, o, h, l, c, v]
                    buffer = st.session_state.candle_buffer
                    if not buffer or ts > buffer[-1][0]:
                        buffer.append(candle)
                    else:
                        buffer[-1] = candle
                    st.session_state.ws_last_update = time.time()
    except Exception as e:
        st.session_state.ws_error = str(e)
        st.session_state.ws_connected = False

# ====================== UI ======================
st.set_page_config(page_title="ETH 5m OKX WS", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (OKX 永续实时)")

col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.ws_connected:
        st.success("● OKX WS 已连接")
    else:
        st.error("○ OKX WS 断开")

with col2:
    delay = time.time() - st.session_state.ws_last_update if st.session_state.ws_last_update else 999
    st.metric("延迟", f"{delay:.1f} 秒")

with col3:
    st.metric("K线数量", len(st.session_state.candle_buffer))

if st.button("🔄 重连 OKX WebSocket"):
    st.session_state.ws_connected = False
    st.session_state.ws_error = None
    st.rerun()

if st.session_state.ws_error:
    st.error(f"错误: {st.session_state.ws_error}")

st.info("广西电信直连 OKX WS 可能受限，请使用 LetsVPN 香港节点。")

# 示例图表（后续可替换你的 EMA 逻辑）
if st.session_state.candle_buffer:
    df = pd.DataFrame(list(st.session_state.candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                         open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close'])])
    st.plotly_chart(fig, use_container_width=True)

st.caption("OKX Cloud 稳定版 • 非线程 • 无警告 • 2026.02")
