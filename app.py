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
import hmac
import hashlib
import base64

# ────────────────────────────────────────────────
#                  配置区（可调参数）
# ────────────────────────────────────────────────
SYMBOL = "ethusdt"                  # Binance 永续合约符号（小写）
INTERVAL = "5m"                     # K线周期：1m,3m,5m,15m...
DB_FILE = "signals.db"
INTERVAL_MS = 5 * 60 * 1000         # 5分钟 = 300000 ms
MAX_CANDLES = 500                   # 内存中最多保留多少根K线

# WebSocket 配置
BINANCE_WS = "wss://fstream.binance.com/ws"
STREAM_NAME = f"{SYMBOL}@kline_{INTERVAL}"

# ────────────────────────────────────────────────
#                  SQLite 工具函数
# ────────────────────────────────────────────────
def get_db_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def init_db():
    with get_db_conn() as conn:
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

def save_signal(record):
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT INTO signals (record_time, signal_time, side, price, ema_fast, ema_slow, rsi, atr, sl, tp1, tp2, result, peak, note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record['record_time'], record['signal_time'], record['side'], record['price'],
            record['ema_fast'], record['ema_slow'], record['rsi'], record['atr'],
            record['sl'], record['tp1'], record['tp2'], record['result'], record['peak'], record.get('note', '')
        ))
        return c.lastrowid

def update_signal(sid, **kwargs):
    if not kwargs: return
    with get_db_conn() as conn:
        c = conn.cursor()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        values = list(kwargs.values()) + [sid]
        c.execute(f"UPDATE signals SET {sets} WHERE id=?", values)
        conn.commit()

def load_history(limit=200):
    with get_db_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT ?", conn, params=(limit,))
        return deque(df.to_dict('records'), maxlen=limit)

# 初始化数据库 & 加载历史
init_db()
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = load_history(200)
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=MAX_CANDLES)

# ────────────────────────────────────────────────
#                  WebSocket 实时数据
# ────────────────────────────────────────────────
def on_message(ws, message):
    try:
        data = json.loads(message)
        if 'k' in data:
            k = data['k']
            candle = [
                int(k['t']),           # open time
                float(k['o']),
                float(k['h']),
                float(k['l']),
                float(k['c']),
                float(k['v'])
            ]
            q = st.session_state.get('ws_queue', deque())
            q.append(candle)
            st.session_state.ws_queue = q
            st.session_state.ws_last_update = time.time()
    except Exception as e:
        st.session_state.ws_error = str(e)

def on_open(ws):
    ws.send(json.dumps({
        "method": "SUBSCRIBE",
        "params": [STREAM_NAME],
        "id": 1
    }))
    st.session_state.ws_connected = True
    st.session_state.ws_error = None

def on_error(ws, error):
    st.session_state.ws_connected = False
    st.session_state.ws_error = str(error)

def on_close(ws, code, msg):
    st.session_state.ws_connected = False

def ws_thread():
    while True:
        try:
            ws = websocket.WebSocketApp(
                BINANCE_WS,
                on_message=on_message,
                on_open=on_open,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=25, ping_timeout=10)
        except Exception as e:
            st.session_state.ws_error = str(e)
            time.sleep(3)

# 启动 WebSocket（只启动一次）
if 'ws_thread_started' not in st.session_state:
    threading.Thread(target=ws_thread, daemon=True).start()
    st.session_state.ws_thread_started = True
    st.session_state.ws_queue = deque()
    st.session_state.ws_connected = False
    st.session_state.ws_last_update = 0

# ────────────────────────────────────────────────
#                  实时数据处理
# ────────────────────────────────────────────────
buffer = st.session_state.candle_buffer
queue = st.session_state.ws_queue

while queue:
    new = queue.popleft()
    ts, o, h, l, c, v = new

    if not buffer or ts > buffer[-1][0]:
        # 新 K线到来，先补缺（理论上 WS 不太会缺，但以防万一）
        if buffer:
            last_ts = buffer[-1][0]
            while last_ts + INTERVAL_MS < ts:
                last_ts += INTERVAL_MS
                buffer.append([last_ts, buffer[-1][4]] * 4 + [0])
        buffer.append(new)
    else:
        # 更新当前正在形成的 K线
        buffer[-1] = new

# ────────────────────────────────────────────────
#                  Streamlit 界面
# ────────────────────────────────────────────────
st.set_page_config(page_title="ETH 5m 极致剥头皮 - Binance WS 实时", layout="wide")

st.title("📈 ETH 5分钟 EMA 剥头皮策略（Binance Futures 实时 WebSocket）")

# WebSocket 状态栏
col1, col2, col3 = st.columns([1,2,1])
with col1:
    if st.session_state.ws_connected:
        st.success("● WebSocket 已连接")
    else:
        st.error("○ WebSocket 断开")

with col2:
    last = st.session_state.get('ws_last_update', 0)
    delay = time.time() - last if last else 0
    if delay < 10:
        st.caption(f"实时更新正常（延迟 ≈ {delay:.1f} 秒）")
    elif delay < 60:
        st.warning(f"延迟 {delay:.1f} 秒 - 正在重连...")
    else:
        st.error(f"长时间无更新 ({delay:.0f} 秒)")

with col3:
    st.metric("K线数量", len(buffer))

# 其余参数、指标计算、信号检测、UI 卡片、图表、历史记录等代码
# 请将你之前最满意的版本直接粘贴到这里（从侧边栏到页面底部）
# 例如：

# with st.sidebar:
#     st.header("策略参数")
#     fast_ema = st.number_input("快线 EMA", 1, 50, 8)
#     ... 其他参数 ...

# 指标计算、信号逻辑、图表绘制、信号卡片等保持你原来的实现

st.markdown("---")
st.caption("极致优化版 • Binance Futures WebSocket 实时 • 自动重连 • 2026")
