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
import requests

# ====================== 配置 ======================
SYMBOL = "ETHUSDT"                  # 永续合约符号（大写）
INTERVAL = "5m"
DB_FILE = "signals.db"
MAX_CANDLES = 500
BINANCE_WS_URL = "wss://fstream.binance.com/ws"
BINANCE_REST_URL = "https://fapi.binance.com"

# 从 Streamlit Secrets 读取密钥（最安全方式）
API_KEY = st.secrets.get("BINANCE_API_KEY", "")
API_SECRET = st.secrets.get("BINANCE_API_SECRET", "")

if not API_KEY or not API_SECRET:
    st.error("请在 Streamlit Secrets 中配置 BINANCE_API_KEY 和 BINANCE_API_SECRET")

# ====================== Session State ======================
for key, default in [
    ('candle_buffer', deque(maxlen=MAX_CANDLES)),
    ('signal_history', deque(maxlen=200)),
    ('ws_queue', deque()),
    ('ws_connected', False),
    ('ws_error', None),
    ('ws_last_update', 0),
    ('account_balance', {}),
    ('positions', [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

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

# ====================== Binance REST API 签名请求 ======================
def binance_signed_request(endpoint, method="GET", params=None):
    if not API_KEY or not API_SECRET:
        return {"error": "API Key/Secret 未配置"}

    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    if params:
        query_string += '&' + '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    query_string += f"&signature={signature}"
    url = f"{BINANCE_REST_URL}{endpoint}?{query_string}"
    
    headers = {"X-MBX-APIKEY": API_KEY}
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            resp = requests.post(url, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# ====================== 查询账户信息 ======================
def get_account_balance():
    data = binance_signed_request("/fapi/v2/balance")
    if "error" in data:
        return data
    return {item['asset']: float(item['balance']) for item in data if float(item['balance']) > 0}

def get_positions():
    data = binance_signed_request("/fapi/v2/positionRisk")
    if "error" in data:
        return data
    return [p for p in data if float(p['positionAmt']) != 0]

# ====================== WebSocket 实时 K线 ======================
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
        "params": [f"{SYMBOL.lower()}@kline_{INTERVAL}"],
        "id": 1
    }))
    st.session_state.ws_connected = True
    st.session_state.ws_error = None

def on_error(ws, error):
    st.session_state.ws_connected = False
    st.session_state.ws_error = str(error)

def on_close(ws, *args):
    st.session_state.ws_connected = False

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
        except:
            time.sleep(3)

if 'ws_thread_started' not in st.session_state:
    threading.Thread(target=ws_thread, daemon=True).start()
    st.session_state.ws_thread_started = True

# 处理新 K线
while st.session_state.ws_queue:
    new_candle = st.session_state.ws_queue.popleft()
    buffer = st.session_state.candle_buffer
    if not buffer or new_candle[0] > buffer[-1][0]:
        buffer.append(new_candle)
    else:
        buffer[-1] = new_candle

# ====================== Streamlit UI ======================
st.set_page_config(page_title="ETH 5m 极致剥头皮 - Binance", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (Binance Futures 实时)")

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
    else:
        st.error(f"长时间无更新 ({delay:.0f} 秒)")

with col3:
    st.metric("K线数量", len(st.session_state.candle_buffer))

# API 状态
st.subheader("账户状态")
if st.button("刷新账户余额 & 持仓"):
    st.session_state.account_balance = get_account_balance()
    st.session_state.positions = get_positions()

if st.session_state.account_balance:
    st.json(st.session_state.account_balance)
else:
    st.info("点击上方按钮刷新余额")

if st.session_state.positions:
    st.dataframe(pd.DataFrame(st.session_state.positions))
else:
    st.info("无持仓或点击刷新")

# ====================== 你的策略逻辑区 ======================
# 请把 EMA 计算、信号检测、图表、历史信号等代码粘贴到这里

# 示例：简单 K线图
if st.session_state.candle_buffer:
    df = pd.DataFrame(list(st.session_state.candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                         open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close'])])
    st.plotly_chart(fig, use_container_width=True)

st.caption("安全版 v3.0 • Secrets 密钥 • 实时 WebSocket • 账户查询 • 2026.02")
