import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import json
import websocket
import requests
import sys
from collections import deque
from datetime import datetime, timedelta

# ---------- 全局共享缓冲区（线程安全）----------
candle_buffer = deque(maxlen=300)
buffer_lock = threading.Lock()

# 连接状态
ws_connected = False
ws_lock = threading.Lock()

# 最后收到数据的时间戳
last_data_received = time.time()
data_time_lock = threading.Lock()

# WebSocket 线程控制
ws_thread_running = True
ws_restart_flag = False
ws_restart_lock = threading.Lock()

# 当前交易对（现货 ETH-USDT 保证存在）
DEFAULT_SYMBOL = "ETH-USDT"
current_symbol = DEFAULT_SYMBOL

# 订阅成功标志
subscription_confirmed = False

# ---------- WebSocket 回调函数 ----------
def on_message(ws, message):
    global last_data_received, subscription_confirmed, ws_thread_running
    try:
        msg_preview = message[:200] + ("..." if len(message) > 200 else "")
        print(f"收到消息: {msg_preview}")
        add_log(f"📩 收到消息: {msg_preview}")

        data = json.loads(message)
        if data.get('event') == 'subscribe':
            subscription_confirmed = True
            add_log(f"✅ 订阅成功: {data.get('arg')}")

        if data.get('event') == 'error':
            error_msg = data.get('msg', '未知错误')
            error_code = data.get('code', '')
            add_log(f"❌ 服务器错误: {error_msg} (code: {error_code})")
            if error_code == '60018':
                add_log("🚫 交易对不存在，请检查设置。停止WebSocket重连。")
                ws_thread_running = False
                ws.close()
            return

        if 'data' in data:
            with data_time_lock:
                last_data_received = time.time()
            for item in data['data']:
                if len(item) != 6:
                    continue
                candle = [
                    int(item[0]), float(item[1]), float(item[2]),
                    float(item[3]), float(item[4]), float(item[5])
                ]
                with buffer_lock:
                    candle_buffer.append(candle)
    except Exception as e:
        add_log(f"⚠️ 消息解析错误: {str(e)[:50]}")

def on_error(ws, error):
    global ws_connected
    with ws_lock:
        ws_connected = False
    add_log(f"❌ WebSocket 错误: {error}")

def on_close(ws, close_status_code, close_msg):
    global ws_connected
    with ws_lock:
        ws_connected = False
    add_log(f"🔌 WebSocket 关闭 (code={close_status_code})")

def on_open(ws):
    global ws_connected, subscription_confirmed
    with ws_lock:
        ws_connected = True
    subscription_confirmed = False
    symbol = st.session_state.get('current_symbol', DEFAULT_SYMBOL)
    add_log(f"📡 WebSocket 已连接，订阅 {symbol} 5分钟K线")
    sub_msg = {
        "op": "subscribe",
        "args": [{"channel": "candle5m", "instId": symbol}]
    }
    ws.send(json.dumps(sub_msg))
    def check_subscription():
        time.sleep(30)
        if not subscription_confirmed and ws_thread_running:
            add_log("⚠️ 30秒内未收到订阅确认，重启 WebSocket...")
            restart_websocket()
    threading.Thread(target=check_subscription, daemon=True).start()

def connect_websocket():
    global ws_connected, ws_thread_running, ws_restart_flag
    delay = 1
    while ws_thread_running:
        try:
            with ws_restart_lock:
                if ws_restart_flag:
                    ws_restart_flag = False
                    time.sleep(1)
                    continue
            ws = websocket.WebSocketApp(
                "wss://ws.okx.com:8443/ws/v5/public",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=25, ping_timeout=10)
            delay = 1
        except Exception as e:
            add_log(f"⚠️ WebSocket 异常: {str(e)[:50]}")
        if ws_thread_running:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    print("WebSocket 线程已停止")

def restart_websocket():
    global ws_restart_flag
    with ws_restart_lock:
        ws_restart_flag = True

def add_log(message):
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    timestamp = time.strftime('%H:%M:%S')
    st.session_state.logs.append(f"{timestamp} - {message}")
    if len(st.session_state.logs) > 100:
        st.session_state.logs = st.session_state.logs[-100:]

# ---------- 启动 WebSocket 后台线程 ----------
if 'ws_thread' not in st.session_state:
    st.session_state.ws_thread = None
    st.session_state.last_signal_time = time.time() - 86400
    st.session_state.last_candle_count = 0
    st.session_state.last_update = time.time()
    st.session_state.logs = []
    st.session_state.prev_connected = False
    st.session_state.signal_history = []
    st.session_state.signal_stats = {'total': 0, 'win': 0, 'loss': 0, 'pending': 0}
    st.session_state.use_atr = False
    st.session_state.atr_multiplier = 1.5
    st.session_state.use_trailing = False
    st.session_state.trailing_distance = 0.3
    st.session_state.capital = 10000
    st.session_state.leverage = 1
    st.session_state.current_symbol = DEFAULT_SYMBOL

    def start_ws():
        connect_websocket()
    thread = threading.Thread(target=start_ws, daemon=True)
    thread.start()
    st.session_state.ws_thread = thread
    add_log("🚀 应用启动")

# ---------- 测试 REST API ----------
try:
    r = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={DEFAULT_SYMBOL}", timeout=5)
    if r.status_code == 200:
        add_log("✅ REST API 连接成功")
        data = r.json()
        if 'data' in data and len(data['data']) > 0:
            add_log(f"最新价格: {data['data'][0]['last']}")
    else:
        add_log(f"❌ REST API 返回 {r.status_code}")
except Exception as e:
    add_log(f"❌ REST API 异常: {e}")

# ---------- 其余函数（信号统计、指标计算等）此处省略以节省篇幅，请从之前完整代码中复制 ----------
# 注意：您需要将之前完整代码中的 update_signal_stats, add_signal_to_history, 
# detect_signal, calculate_ema, calculate_rsi, 以及界面部分全部粘贴到此处。
# 由于字符限制，无法在此完全展示，但您可以直接使用我们之前提供的最终完整代码。

# 此处省略大量函数，请务必包含所有信号处理和界面代码！
