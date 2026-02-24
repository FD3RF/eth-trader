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

# 当前交易对（默认使用正确的ID，根据OKX文档，币本位永续合约为 ETH-USD-SWAP）
DEFAULT_SYMBOL = "ETH-USD-SWAP"
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
        # 检查是否是订阅成功的确认消息（OKX格式）
        if data.get('event') == 'subscribe':
            subscription_confirmed = True
            add_log(f"✅ 订阅成功: {data.get('arg')}")

        # 处理服务器返回的错误
        if data.get('event') == 'error':
            error_msg = data.get('msg', '未知错误')
            error_code = data.get('code', '')
            add_log(f"❌ 服务器错误: {error_msg} (code: {error_code})")
            # 如果是交易对不存在，停止重连
            if error_code == '60018':  # instrument ID does not exist
                add_log("🚫 交易对不存在，请检查设置。停止WebSocket重连。")
                ws_thread_running = False
                ws.close()
            return

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
    symbol = st.session_state.get('current_symbol', DEFAULT_SYMBOL)
    add_log(f"📡 WebSocket 已连接，订阅 {symbol} 5分钟K线")
    sub_msg = {
        "op": "subscribe",
        "args": [{"channel": "candle5m", "instId": symbol}]
    }
    ws.send(json.dumps(sub_msg))
    # 启动一个定时器，如果30秒内未收到订阅确认，则主动重启
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
    st.session_state.current_symbol = DEFAULT_SYMBOL

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
    r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USD-SWAP", timeout=5)
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

# ---------- 信号统计函数 ----------
def update_signal_stats():
    total = len(st.session_state.signal_history)
    if total == 0:
        st.session_state.signal_stats = {'total': 0, 'win': 0, 'loss': 0, 'pending': 0, 'win_rate': 0}
        return
    
    win = sum(1 for s in st.session_state.signal_history if s.get('result') == 'win')
    loss = sum(1 for s in st.session_state.signal_history if s.get('result') == 'loss')
    pending = sum(1 for s in st.session_state.signal_history if s.get('result') == 'pending')
    
    st.session_state.signal_stats = {
        'total': total,
        'win': win,
        'loss': loss,
        'pending': pending,
        'win_rate': round(win / (win + loss) * 100, 1) if (win + loss) > 0 else 0
    }

def add_signal_to_history(signal):
    signal_record = {
        'time': pd.to_datetime(signal['timestamp'], unit='s').strftime('%Y-%m-%d %H:%M'),
        'side': signal['side'],
        'price': signal['price'],
        'ema_fast': round(signal['ema_fast'], 2),
        'ema_slow': round(signal['ema_slow'], 2),
        'rsi': round(signal['rsi'], 1),
        'sl': round(signal['sl'], 2),
        'tp1': round(signal['tp1'], 2),
        'tp2': round(signal['tp2'], 2),
        'atr_sl': round(signal.get('atr_sl', 0), 2) if signal.get('atr_sl') else None,
        'atr_tp1': round(signal.get('atr_tp1', 0), 2) if signal.get('atr_tp1') else None,
        'atr_tp2': round(signal.get('atr_tp2', 0), 2) if signal.get('atr_tp2') else None,
        'result': 'pending',
        'exit_price': None,
        'exit_time': None,
        'exit_reason': None,
        'tp1_hit': False,
        'highest_price': signal['price'] if signal['side'] == 'BUY' else None,
        'lowest_price': signal['price'] if signal['side'] == 'SELL' else None,
        'note': ''
    }
    st.session_state.signal_history.insert(0, signal_record)
    if len(st.session_state.signal_history) > 200:
        st.session_state.signal_history = st.session_state.signal_history[:200]
    update_signal_stats()

def update_signal_result(index, result, exit_price=None, exit_reason=None, note=''):
    if 0 <= index < len(st.session_state.signal_history):
        st.session_state.signal_history[index]['result'] = result
        if exit_price:
            st.session_state.signal_history[index]['exit_price'] = round(exit_price, 2)
            st.session_state.signal_history[index]['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        if exit_reason:
            st.session_state.signal_history[index]['exit_reason'] = exit_reason
        if note:
            st.session_state.signal_history[index]['note'] = note
        update_signal_stats()

def clear_signal_history():
    st.session_state.signal_history = []
    update_signal_stats()
    add_log("🧹 历史信号已清空")

def cleanup_old_signals():
    if len(st.session_state.signal_history) == 0:
        return
    
    cutoff_time = datetime.now() - timedelta(hours=24)
    cleaned = []
    for s in st.session_state.signal_history:
        try:
            signal_time = datetime.strptime(s['time'], '%Y-%m-%d %H:%M')
            if signal_time > cutoff_time or s['result'] != 'pending':
                cleaned.append(s)
            else:
                add_log(f"🧹 清理超时待定信号: {s['time']} {s['side']}")
        except:
            cleaned.append(s)
    
    st.session_state.signal_history = cleaned
    update_signal_stats()

# ---------- ATR计算函数 ----------
def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_atr_stop(entry_price, side, atr_value, multiplier=1.5):
    if side == 'BUY':
        sl = entry_price - (atr_value * multiplier)
        tp1 = entry_price + (atr_value * multiplier * 0.6)
        tp2 = entry_price + (atr_value * multiplier * 1.2)
    else:
        sl = entry_price + (atr_value * multiplier)
        tp1 = entry_price - (atr_value * multiplier * 0.6)
        tp2 = entry_price - (atr_value * multiplier * 1.2)
    return sl, tp1, tp2

# ---------- 指标计算函数 ----------
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_signal(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max):
    if len(df) < max(slow, rsi_period) + 5:
        return None
    last = df.iloc[-1]
    ema_fast_now = df['ema_fast'].iloc[-1]
    ema_slow_now = df['ema_slow'].iloc[-1]
    ema_fast_prev = df['ema_fast'].iloc[-2]
    ema_slow_prev = df['ema_slow'].iloc[-2]
    rsi_now = df['rsi'].iloc[-1]

    golden_cross = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
    death_cross = ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now

    if golden_cross and last['close'] > ema_fast_now and buy_min < rsi_now < buy_max:
        return {
            'side': 'BUY',
            'price': last['close'],
            'ema_fast': ema_fast_now,
            'ema_slow': ema_slow_now,
            'rsi': rsi_now,
            'timestamp': last.name.timestamp()
        }
    if death_cross and last['close'] < ema_fast_now and sell_min < rsi_now < sell_max:
        return {
            'side': 'SELL',
            'price': last['close'],
            'ema_fast': ema_fast_now,
            'ema_slow': ema_slow_now,
            'rsi': rsi_now,
            'timestamp': last.name.timestamp()
        }
    return None

def calculate_sltp(entry_price, side):
    if side == 'BUY':
        sl = entry_price * 0.994
        tp1 = entry_price * 1.006
        tp2 = entry_price * 1.012
    else:
        sl = entry_price * 1.006
        tp1 = entry_price * 0.994
        tp2 = entry_price * 0.988
    return sl, tp1, tp2

# ---------- 自动出场检查函数（整合功能1、2 + 移动止损）----------
def check_exit_conditions(df, current_signal, current_price, new_signal=None, atr_value=None):
    if current_signal['result'] != 'pending':
        return False, None, None, None
    
    side = current_signal['side']
    entry = current_signal['price']
    
    if side == 'BUY':
        if current_signal.get('highest_price', entry) < current_price:
            current_signal['highest_price'] = current_price
    else:
        if current_signal.get('lowest_price', entry) > current_price:
            current_signal['lowest_price'] = current_price
    
    if st.session_state.use_atr and atr_value and atr_value > 0:
        sl = current_signal.get('atr_sl', entry - atr_value * st.session_state.atr_multiplier if side == 'BUY' else entry + atr_value * st.session_state.atr_multiplier)
        tp1 = current_signal.get('atr_tp1', entry + atr_value * st.session_state.atr_multiplier * 0.6 if side == 'BUY' else entry - atr_value * st.session_state.atr_multiplier * 0.6)
        tp2 = current_signal.get('atr_tp2', entry + atr_value * st.session_state.atr_multiplier * 1.2 if side == 'BUY' else entry - atr_value * st.session_state.atr_multiplier * 1.2)
    else:
        sl = current_signal['sl']
        tp1 = current_signal['tp1']
        tp2 = current_signal['tp2']
    
    tp1_hit = current_signal.get('tp1_hit', False)
    
    trailing_sl = None
    if st.session_state.use_trailing and not tp1_hit:
        trail_dist = st.session_state.trailing_distance / 100
        if side == 'BUY' and current_signal.get('highest_price'):
            trailing_sl = current_signal['highest_price'] * (1 - trail_dist)
        elif side == 'SELL' and current_signal.get('lowest_price'):
            trailing_sl = current_signal['lowest_price'] * (1 + trail_dist)
    
    if trailing_sl is not None:
        current_sl = max(sl, trailing_sl) if side == 'BUY' else min(sl, trailing_sl)
    else:
        current_sl = sl
    
    if side == 'BUY':
        if not tp1_hit and current_price >= tp1:
            current_signal['tp1_hit'] = True
            current_signal['sl'] = entry
            add_log(f"💡 TP1触发：{current_signal['time']} {side}信号保本价已移动至 {entry:.2f}")
            return False, None, None, None
        
        if current_signal.get('tp1_hit'):
            current_sl = entry
        
        if current_price <= current_sl:
            return True, 'loss', current_price, '止损触发'
        elif current_price >= tp2:
            return True, 'win', current_price, 'TP2触发'
    else:
        if not tp1_hit and current_price <= tp1:
            current_signal['tp1_hit'] = True
            current_signal['sl'] = entry
            add_log(f"💡 TP1触发：{current_signal['time']} {side}信号保本价已移动至 {entry:.2f}")
            return False, None, None, None
        
        if current_signal.get('tp1_hit'):
            current_sl = entry
        
        if current_price >= current_sl:
            return True, 'loss', current_price, '止损触发'
        elif current_price <= tp2:
            return True, 'win', current_price, 'TP2触发'
    
    if new_signal and new_signal['side'] != side:
        if side == 'BUY':
            pnl_pct = (current_price - entry) / entry * 100
        else:
            pnl_pct = (entry - current_price) / entry * 100
        result = 'win' if pnl_pct > 0 else 'loss'
        return True, result, current_price, f'反向信号出场 ({pnl_pct:.2f}%)'
    
    if len(df) >= 2:
        ema_fast = df['ema_fast'].iloc[-1]
        if side == 'BUY' and current_price < ema_fast:
            pnl_pct = (current_price - entry) / entry * 100
            result = 'win' if pnl_pct > 0 else 'loss'
            return True, result, current_price, f'跌破EMA快线 ({pnl_pct:.2f}%)'
        elif side == 'SELL' and current_price > ema_fast:
            pnl_pct = (entry - current_price) / entry * 100
            result = 'win' if pnl_pct > 0 else 'loss'
            return True, result, current_price, f'突破EMA快线 ({pnl_pct:.2f}%)'
    
    signal_time = datetime.strptime(current_signal['time'], '%Y-%m-%d %H:%M')
    if datetime.now() - signal_time > timedelta(minutes=60):
        if side == 'BUY':
            pnl_pct = (current_price - entry) / entry * 100
        else:
            pnl_pct = (entry - current_price) / entry * 100
        result = 'win' if pnl_pct > 0 else 'loss'
        return True, result, current_price, f'超时出场 ({pnl_pct:.2f}%)'
    
    return False, None, None, None

# ---------- 计算累计PNL ----------
def calculate_cumulative_pnl():
    closed_signals = [s for s in st.session_state.signal_history 
                     if s['result'] in ['win', 'loss']]
    if not closed_signals:
        return pd.DataFrame()
    pnl_list = []
    cumulative = 0
    for s in reversed(closed_signals):
        if s['side'] == 'BUY':
            pnl_pct = (s['exit_price'] - s['price']) / s['price']
        else:
            pnl_pct = (s['price'] - s['exit_price']) / s['price']
        trade_pnl = pnl_pct * 0.01
        cumulative += trade_pnl
        pnl_list.append({
            'time': s['exit_time'] or s['time'],
            'pnl': cumulative * 100
        })
    return pd.DataFrame(pnl_list)

# ---------- Streamlit 页面配置 ----------
st.set_page_config(page_title="加密货币 5分钟剥头皮监控", layout="wide")
st.title("📈 加密货币 5分钟 EMA 剥头皮监控 (增强版)")

# ---------- 侧边栏参数 ----------
st.sidebar.header("策略参数")

# 交易对选择（手动输入或选择，这里提供常用选项）
symbol_options = ["ETH-USD-SWAP", "BTC-USD-SWAP", "SOL-USD-SWAP"]
selected_symbol = st.sidebar.selectbox("交易对", symbol_options, index=0)
if selected_symbol != st.session_state.get('current_symbol', DEFAULT_SYMBOL):
    st.session_state.current_symbol = selected_symbol
    with buffer_lock:
        candle_buffer.clear()
    add_log(f"🔄 切换交易对至 {selected_symbol}")
    restart_websocket()

fast_ema = st.sidebar.number_input("快线 EMA", 1, 50, 9, 1)
slow_ema = st.sidebar.number_input("慢线 EMA", 2, 100, 21, 1)
rsi_period = st.sidebar.number_input("RSI 周期", 2, 50, 14, 1)
buy_min = st.sidebar.number_input("多头 RSI 下限", 0, 100, 50, 1)
buy_max = st.sidebar.number_input("多头 RSI 上限", 0, 100, 70, 1)
sell_min = st.sidebar.number_input("空头 RSI 下限", 0, 100, 30, 1)
sell_max = st.sidebar.number_input("空头 RSI 上限", 0, 100, 50, 1)
refresh_interval = st.sidebar.number_input("刷新间隔(秒)", 1, 30, 4, 1)

st.sidebar.markdown("---")

# ---------- ATR设置 ----------
st.sidebar.subheader("⚙️ ATR动态止损")
st.session_state.use_atr = st.sidebar.checkbox("启用ATR止损", value=False)
if st.session_state.use_atr:
    st.session_state.atr_multiplier = st.sidebar.slider("ATR倍数", 0.5, 3.0, 1.5, 0.1)
    st.sidebar.caption("ATR止损会更适应市场波动")

st.sidebar.markdown("---")

# ---------- 移动止损设置 ----------
st.sidebar.subheader("⚙️ 移动止损")
st.session_state.use_trailing = st.sidebar.checkbox("启用移动止损", value=False)
if st.session_state.use_trailing:
    st.session_state.trailing_distance = st.sidebar.slider("移动止损距离 (%)", 0.1, 2.0, 0.3, 0.1)
    st.sidebar.caption("价格朝有利方向移动时动态上移止损")

st.sidebar.markdown("---")

# ---------- 资金设置 ----------
st.sidebar.subheader("💰 资金管理")
st.session_state.capital = st.sidebar.number_input("本金 (USDT)", 100, 1000000, 10000, 100)
st.session_state.leverage = st.sidebar.number_input("杠杆", 1, 100, 1, 1)

st.sidebar.markdown("---")

# ---------- 实时状态显示 ----------
with ws_lock:
    is_connected = ws_connected
conn_status = st.sidebar.empty()
if is_connected:
    conn_status.success("✅ WebSocket 已连接")
else:
    conn_status.error("❌ WebSocket 未连接")

with data_time_lock:
    last_recv = last_data_received
last_recv_str = time.strftime('%H:%M:%S', time.localtime(last_recv))
st.sidebar.caption(f"📡 最后收到数据：{last_recv_str}")

connection_changed = False
if st.session_state.prev_connected != is_connected:
    connection_changed = True
    st.session_state.prev_connected = is_connected
    if is_connected:
        add_log("✅ WebSocket 已连接")
    else:
        add_log("❌ WebSocket 断开")

# ---------- 信号统计卡片 ----------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 信号统计")

update_signal_stats()
stats = st.session_state.signal_stats

col1, col2 = st.sidebar.columns(2)
col1.metric("总信号", stats['total'])
col2.metric("胜率", f"{stats['win_rate']}%" if stats['total'] > 0 else "0%")

col1, col2 = st.sidebar.columns(2)
col1.metric("✅ 盈利", stats['win'])
col2.metric("❌ 亏损", stats['loss'])

if stats['pending'] > 0:
    st.sidebar.info(f"⏳ 待定信号: {stats['pending']}")

if st.sidebar.button("🧹 清空历史信号"):
    clear_signal_history()
    st.rerun()

# ---------- 运行日志 ----------
with st.sidebar.expander("📋 运行日志（最近20条）", expanded=False):
    for log in st.session_state.logs[-20:]:
        st.text(log)

if st.sidebar.button("🔄 强制重启 WebSocket"):
    add_log("🔄 用户触发强制重启 WebSocket")
    restart_websocket()

if st.sidebar.button("🔄 重置所有状态（保留信号）"):
    keep_keys = ['ws_thread', 'logs', 'signal_history', 'signal_stats']
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]
    st.session_state.last_signal_time = time.time() - 86400
    st.session_state.last_candle_count = 0
    st.session_state.last_update = time.time()
    st.session_state.prev_connected = False
    update_signal_stats()
    add_log("🔄 所有状态已重置")
    st.rerun()

# ---------- 读取最新数据 ----------
with buffer_lock:
    candles = list(candle_buffer)
    current_len = len(candles)

cleanup_old_signals()

if current_len < 30:
    st.warning(f"正在等待数据积累... 当前 {current_len}/30 根")
else:
    try:
        df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates(subset='ts', keep='last').sort_values('ts')
        df['time'] = pd.to_datetime(df['ts'], unit='ms', errors='coerce')
        df = df.dropna(subset=['time'])
        df.set_index('time', inplace=True)

        df['ema_fast'] = calculate_ema(df['close'], fast_ema)
        df['ema_slow'] = calculate_ema(df['close'], slow_ema)
        df['rsi'] = calculate_rsi(df['close'], rsi_period)
        
        if st.session_state.use_atr:
            df['atr'] = calculate_atr(df, 14)
            current_atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else None
        else:
            current_atr = None

        last_kline_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
        st.caption(f"📊 最后K线时间：{last_kline_time}   |   当前 K 线数：{current_len}   |   交易对：{st.session_state.get('current_symbol', DEFAULT_SYMBOL)}")

        new_signal = detect_signal(df, fast_ema, slow_ema, rsi_period,
                                   buy_min, buy_max, sell_min, sell_max)

        if new_signal and new_signal['timestamp'] > st.session_state.last_signal_time:
            st.session_state.last_signal_time = new_signal['timestamp']
            
            if st.session_state.use_atr and current_atr and current_atr > 0:
                sl, tp1, tp2 = calculate_atr_stop(
                    new_signal['price'], new_signal['side'], 
                    current_atr, st.session_state.atr_multiplier
                )
                new_signal['sl'] = sl
                new_signal['tp1'] = tp1
                new_signal['tp2'] = tp2
                new_signal['atr_sl'] = sl
                new_signal['atr_tp1'] = tp1
                new_signal['atr_tp2'] = tp2
                add_log(f"📊 当前ATR: {current_atr:.2f}")
            else:
                sl, tp1, tp2 = calculate_sltp(new_signal['price'], new_signal['side'])
                new_signal['sl'] = sl
                new_signal['tp1'] = tp1
                new_signal['tp2'] = tp2
            
            add_signal_to_history(new_signal)

            signal_time_str = pd.to_datetime(new_signal['timestamp'], unit='s').strftime('%H:%M')
            add_log(f"🔔 {new_signal['side']} 信号 @ {signal_time_str} 价格 {new_signal['price']:.2f}")

            if new_signal['side'] == 'BUY':
                st.success(f"### 🟢 多头信号 @ {pd.to_datetime(new_signal['timestamp'], unit='s').strftime('%Y-%m-%d %H:%M')}")
            else:
                st.error(f"### 🔴 空头信号 @ {pd.to_datetime(new_signal['timestamp'], unit='s').strftime('%Y-%m-%d %H:%M')}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("价格", f"{new_signal['price']:.2f}")
            col2.metric(f"EMA{fast_ema}", f"{new_signal['ema_fast']:.2f}")
            col3.metric(f"EMA{slow_ema}", f"{new_signal['ema_slow']:.2f}")
            col4.metric("RSI", f"{new_signal['rsi']:.1f}")

            if st.session_state.use_atr and current_atr:
                st.info(f"""
                **📉 ATR止损**：{sl:.2f} (ATR×{st.session_state.atr_multiplier})  
                **📈 ATR-TP1**：{tp1:.2f}  
                **🚀 ATR-TP2**：{tp2:.2f}
                """)
            else:
                st.info(f"""
                **📉 止损**：{sl:.2f}  
                **📈 TP1 (0.6%)**：{tp1:.2f}  
                **🚀 TP2 (1.2%)**：{tp2:.2f}
                """)

        if len(st.session_state.signal_history) > 0:
            current_price = df['close'].iloc[-1]
            
            for idx, signal_record in enumerate(st.session_state.signal_history):
                if signal_record['result'] == 'pending':
                    should_exit, result, exit_price, reason = check_exit_conditions(
                        df, signal_record, current_price, new_signal, current_atr
                    )
                    if should_exit:
                        update_signal_result(idx, result, exit_price, reason)
                        add_log(f"🤖 自动出场：{signal_record['side']}信号 @ {signal_record['time']} - {reason} @ {exit_price:.2f}")

        pending_signals = [s for s in st.session_state.signal_history if s['result'] == 'pending']
        if pending_signals:
            st.markdown("---")
            st.subheader("📊 当前持仓盈亏")
            cols = st.columns(min(len(pending_signals), 4))
            for i, s in enumerate(pending_signals[:4]):
                current_price = df['close'].iloc[-1]
                if s['side'] == 'BUY':
                    pnl_pct = (current_price - s['price']) / s['price'] * 100
                    pnl_amount = (current_price - s['price']) * (st.session_state.capital * st.session_state.leverage / s['price'])
                else:
                    pnl_pct = (s['price'] - current_price) / s['price'] * 100
                    pnl_amount = (s['price'] - current_price) * (st.session_state.capital * st.session_state.leverage / s['price'])
                
                with cols[i % 4]:
                    st.metric(
                        f"{s['side']} @ {s['time'][5:16]}",
                        f"{pnl_pct:+.2f}%",
                        delta=f"${pnl_amount:+,.0f}  入场: {s['price']:.2f}"
                    )

        chart_df = df[['close', 'ema_fast', 'ema_slow']].tail(60).dropna(how='all')
        st.subheader("价格走势与EMA")
        st.line_chart(chart_df)

        st.subheader("最近 10 根K线数据")
        display_cols = ['open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 'rsi']
        if st.session_state.use_atr and 'atr' in df.columns:
            display_cols.append('atr')
        
        display_df = df[display_cols].tail(10).round(2)
        display_df = display_df.fillna('-')
        if 'volume' in display_df.columns:
            display_df['volume'] = display_df['volume'].apply(
                lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x
            )
        st.dataframe(display_df)

    except Exception as e:
        add_log(f"❌ 数据转换失败: {str(e)}")
        st.error("数据格式异常，请检查 WebSocket 推送")
        st.stop()

# ---------- 累计PNL曲线 ----------
st.markdown("---")
st.subheader("📈 模拟累计收益 (1%仓位)")

pnl_df = calculate_cumulative_pnl()
if not pnl_df.empty:
    st.line_chart(pnl_df.set_index('time')['pnl'])
    final_pnl = pnl_df['pnl'].iloc[-1]
    st.metric("累计收益率", f"{final_pnl:.2f}%", 
              delta=f"交易次数: {len(pnl_df)}")
else:
    st.info("暂无已完结信号，等待更多数据...")

# ---------- 历史信号表格 ----------
st.markdown("---")
st.subheader("📜 历史信号记录")

if len(st.session_state.signal_history) > 0:
    history_df = pd.DataFrame(st.session_state.signal_history)
    
    base_cols = ['time', 'side', 'price', 'result', 'exit_price', 'exit_reason', 'exit_time']
    atr_cols = ['atr_sl', 'atr_tp1', 'atr_tp2'] if st.session_state.use_atr else []
    display_cols = base_cols + atr_cols
    display_cols = [col for col in display_cols if col in history_df.columns]
    
    display_df = history_df[display_cols].copy()
    
    def color_result(val):
        if val == 'win':
            return 'background-color: #90ee90'
        elif val == 'loss':
            return 'background-color: #ffcccb'
        elif val == 'pending':
            return 'background-color: #fffacd'
        return ''
    
    styled_df = display_df.style.applymap(color_result, subset=['result'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    with st.expander("✏️ 手动标记信号结果", expanded=False):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        
        with col1:
            signal_options = [f"{i+1}. {s['time']} {s['side']} @{s['price']}" 
                             for i, s in enumerate(st.session_state.signal_history[:20])]
            selected_idx = st.selectbox("选择信号", range(len(signal_options)), 
                                       format_func=lambda x: signal_options[x])
        
        with col2:
            result = st.selectbox("结果", ["pending", "win", "loss"])
        
        with col3:
            exit_price = st.number_input("出场价", value=0.0, step=0.1)
        
        with col4:
            if st.button("更新结果"):
                update_signal_result(selected_idx, result, exit_price if exit_price > 0 else None)
                st.rerun()
    
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 导出历史信号 (CSV)",
        data=csv,
        file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("暂无历史信号，等待新信号出现...")

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
