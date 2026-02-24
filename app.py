import streamlit as st
import json
import threading
import time
import pandas as pd
import plotly.graph_objects as go
from websocket import WebSocketApp, WebSocketConnectionClosedException
from datetime import datetime

# ---------- 配置参数 ----------
# 正确的交易对 ID（现货），如需永续合约可改为 "ETH-USD-SWAP"
SYMBOL = "ETH-USDT"
CHANNEL = "candle5m"
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# ---------- 全局变量 ----------
ws = None
ws_thread = None
should_reconnect = True
data_buffer = []          # 存储最新 K 线数据
MAX_BUFFER = 200          # 保留最近 200 根 K 线

# ---------- WebSocket 回调函数 ----------
def on_message(ws, message):
    """处理接收到的消息"""
    global data_buffer
    try:
        data = json.loads(message)
        
        # 处理错误消息
        if data.get("event") == "error":
            error_msg = data.get("msg", "")
            code = data.get("code", "")
            st.error(f"WebSocket 错误 [{code}]: {error_msg}")
            # 如果是交易对不存在错误，停止重连
            if code == "60018":
                st.stop()
            return
        
        # 处理 K 线数据
        if "data" in data and data.get("arg", {}).get("channel") == CHANNEL:
            candle = data["data"][0]
            # 解析 K 线： [ts, o, h, l, c, vol, ...]
            ts = int(candle[0])
            o = float(candle[1])
            h = float(candle[2])
            l = float(candle[3])
            c = float(candle[4])
            vol = float(candle[5])
            dt = datetime.fromtimestamp(ts / 1000)
            
            # 存入缓冲区
            new_row = {
                "time": dt,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": vol
            }
            data_buffer.append(new_row)
            if len(data_buffer) > MAX_BUFFER:
                data_buffer = data_buffer[-MAX_BUFFER:]
                
    except Exception as e:
        st.error(f"解析消息异常: {e}")

def on_error(ws, error):
    st.error(f"WebSocket 错误: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket 连接已关闭")
    if should_reconnect:
        time.sleep(3)
        start_ws()

def on_open(ws):
    """连接成功后订阅频道"""
    subscribe_msg = {
        "op": "subscribe",
        "args": [{
            "channel": CHANNEL,
            "instId": SYMBOL
        }]
    }
    ws.send(json.dumps(subscribe_msg))
    st.success(f"已订阅 {SYMBOL} {CHANNEL}")

# ---------- WebSocket 启动函数 ----------
def start_ws():
    global ws, ws_thread
    # 创建 WebSocketApp
    ws = WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    # 在后台线程中运行
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

# ---------- Streamlit 界面 ----------
st.set_page_config(page_title="ETH 实时K线", layout="wide")
st.title("📈 ETH 实时行情 (5分钟K线)")

# 启动 WebSocket（只启动一次）
if 'ws_started' not in st.session_state:
    start_ws()
    st.session_state.ws_started = True

# 显示数据
chart_placeholder = st.empty()
table_placeholder = st.empty()

while True:
    if data_buffer:
        df = pd.DataFrame(data_buffer)
        
        # 绘制 K 线图
        fig = go.Figure(data=[go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线'
        )])
        fig.update_layout(
            title=f"{SYMBOL} 5分钟K线",
            xaxis_title="时间",
            yaxis_title="价格 (USDT)",
            template="plotly_dark",
            height=600
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # 显示最新数据表格
        table_placeholder.dataframe(df.tail(10).style.format({
            "open": "{:.2f}",
            "high": "{:.2f}",
            "low": "{:.2f}",
            "close": "{:.2f}",
            "volume": "{:.2f}"
        }))
    
    time.sleep(2)  # 更新间隔
