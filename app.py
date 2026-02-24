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

# 当前交易对（初始值）
current_symbol = "ETH-USDT-SWAP"

# 订阅成功标志
subscription_confirmed = False

# ---------- WebSocket 回调函数 ----------
def on_message(ws, message):
    global last_data_received, subscription_confirmed
    try:
        # 打印原始消息（截断）以便调试
        msg_preview = message[:200] + ("..." if len(message) > 200 else "")
        print(f"收到消息: {msg_preview}")
        add_log(f"📩 收到消息: {msg_preview}")

        data = json.loads(message)
        # 检查是否是订阅成功的确认消息（根据OKX文档，成功订阅会返回 {"event":"subscribe","arg":{...}}）
        if data.get('event') == 'subscribe':
            subscription_confirmed = True
            add_log(f"✅ 订阅成功: {data.get('arg')}")

        if data.get('event') == 'error':
            add_log(f"❌ 服务器错误: {data.get('msg')}")

        if 'data' in data:
            with data_time_lock:
                last_data_received = time.time()
            for item in data['data']:
                if len(item) != 6:
                    print(f"数据格式异常: {item}")
                    continue
                candle = [
                    int(item[0]), float(item[1]), float(item[2]),
                    float(item[3]), float(item[4]), float(item[5])
                ]
                with buffer_lock:
                    candle_buffer.append(candle)
    except Exception as e:
        print(f"处理消息出错: {e}")
        add_log(f"⚠️ 消息解析错误: {str(e)[:50]}")

def on_error(ws, error):
    global ws_connected
    with ws_lock:
        ws_connected = False
    error_msg = f"WebSocket 错误: {error}"
    print(error_msg)
    add_log(f"❌ {error_msg}")

def on_close(ws, close_status_code, close_msg):
    global ws_connected
    with ws_lock:
        ws_connected = False
    print(f"WebSocket 关闭 (code={close_status_code}): {close_msg}")
    add_log(f"🔌 WebSocket 关闭 (code={close_status_code})")

def on_open(ws):
    global ws_connected, subscription_confirmed
    with ws_lock:
        ws_connected = True
    subscription_confirmed = False
    # 从 session_state 获取最新交易对
    symbol = st.session_state.get('current_symbol', 'ETH-USDT-SWAP')
    add_log(f"📡 WebSocket 已连接，订阅 {symbol} 5分钟K线")
    sub_msg = {
        "op": "subscribe",
        "args": [{"channel": "candle5m", "instId": symbol}]
    }
    ws.send(json.dumps(sub_msg))
    # 启动一个定时器，如果30秒内未收到订阅确认，则主动重启
    def check_subscription():
        time.sleep(30)
        if not subscription_confirmed:
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
                    print("检测到重启标志，正在重启 WebSocket...")
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
            print(f"WebSocket 运行异常: {e}")
            add_log(f"⚠️ WebSocket 异常: {str(e)[:50]}")

        if ws_thread_running:
            time.sleep(delay)
            delay = min(delay * 2, 60)

    print("WebSocket 线程已停止")

def restart_websocket():
    global ws_restart_flag
    with ws_restart_lock:
        ws_restart_flag = True

# ---------- 日志辅助函数 ----------
def add_log(message):
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    timestamp = time.strftime('%H:%M:%S')
    last_log = st.session_state.logs[-1] if st.session_state.logs else ""
    if "WebSocket 已连接" in message and "WebSocket 已连接" in last_log:
        return
    if "WebSocket 断开" in message and "WebSocket 断开" in last_log:
        return
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
    st.session_state.current_symbol = "ETH-USDT-SWAP"

    def start_ws():
        connect_websocket()
    thread = threading.Thread(target=start_ws, daemon=True)
    thread.start()
    st.session_state.ws_thread = thread

    add_log("🚀 应用启动")
    add_log(f"Streamlit 版本: {st.__version__}")
    add_log(f"Python 版本: {sys.version.split()[0]}")
    add_log("默认参数: EMA9/21, RSI14, 买入50-70, 卖出30-50")

# ---------- 测试 REST API 连通性 ----------
try:
    r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP", timeout=5)
    if r.status_code == 200:
        add_log("✅ REST API 连接成功")
        data = r.json()
        if 'data' in data and len(data['data']) > 0:
            add_log(f"最新价格: {data['data'][0]['last']}")
        else:
            add_log("⚠️ REST API 返回数据异常")
    else:
        add_log(f"❌ REST API 返回 {r.status_code}")
except Exception as e:
    add_log(f"❌ REST API 异常: {e}")

# ---------- 其余函数（信号统计、指标计算等）请从您之前的完整代码中复制，此处省略以节省篇幅 ----------
# 为了完整，您需要将之前代码中的所有函数（update_signal_stats, add_signal_to_history, ...）放在这里。
# 由于字数限制，请从我们之前提供的最终版代码中复制这些函数，粘贴到此处。
# 它们包括：update_signal_stats, add_signal_to_history, update_signal_result, clear_signal_history,
# cleanup_old_signals, calculate_atr, calculate_atr_stop, calculate_ema, calculate_rsi,
# detect_signal, calculate_sltp, check_exit_conditions, calculate_cumulative_pnl 等。

# ---------- Streamlit 界面部分（保持不变，请从最终版代码中复制）----------
# 从您之前的最终代码中复制界面部分（包括侧边栏、主区域、图表等）。
# 注意：请确保所有变量定义和函数调用一致。

# ---------- 智能刷新逻辑 ----------
MIN_REFRESH = 2.0
effective_interval = max(refresh_interval, MIN_REFRESH)

now = time.time()
has_new_data = current_len > st.session_state.last_candle_count
time_to_refresh = now - st.session_state.last_update >= effective_interval

if has_new_data or connection_changed or time_to_refresh:
    st.session_state.last_update = now
    st.session_state.last_candle_count = current_len
    st.rerun()
