import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import websocket  # 需要 pip install websocket-client
import json
import threading

# ---------- 配置 ----------
SYMBOL = "ETH-USDT"
INTERVAL = "5m"
LIMIT = 200

# ---------- 数据获取（初始历史用REST，实时用WebSocket） ----------
@st.cache_data(ttl=15)
def fetch_klines():
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={SYMBOL}&bar={INTERVAL}&limit={LIMIT}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            candles = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
            candles.sort(key=lambda x: x[0])
            return candles
        return None
    except:
        return None

# ---------- WebSocket实时更新 ----------
def on_message(ws, message):
    data = json.loads(message)
    if 'data' in data and data['arg']['channel'] == f'candle{INTERVAL}':
        candle_data = data['data'][0]  # [ts, open, high, low, close, vol]
        latest_candle = [int(candle_data[0]), float(candle_data[1]), float(candle_data[2]), float(candle_data[3]), float(candle_data[4]), float(candle_data[5])]
        
        if len(st.session_state.candle_buffer) == 0 or latest_candle[0] > st.session_state.candle_buffer[-1][0]:
            st.session_state.candle_buffer.append(latest_candle)
        elif latest_candle[0] == st.session_state.candle_buffer[-1][0]:
            st.session_state.candle_buffer[-1] = latest_candle  # 更新正在形成的K线
        st.rerun()  # 刷新页面

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")

def run_ws():
    ws_url = "wss://ws.okx.com:8443/ws/v5/public"
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = lambda ws: ws.send(json.dumps({
        "op": "subscribe",
        "args": [{"channel": f"candle{INTERVAL}", "instId": SYMBOL}]
    }))
    ws.run_forever()

# ---------- 指标 ----------
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def volume_surge(df, vol_period=20, surge_mult=1.5):
    if len(df) < vol_period + 1: return True
    avg_vol = df['volume'].rolling(window=vol_period).mean().iloc[-1]
    return df['volume'].iloc[-1] > avg_vol * surge_mult

# ========== 核心优化1：持续多头/空头信号检测 ==========
def detect_signal_pro(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max,
                      higher_trend, use_volume_filter, use_slope_filter, use_atr_filter):
    if len(df) < slow + rsi_period + 10: return None
    last = df.iloc[-1]
    ema_f_now = df['ema_fast'].iloc[-1]
    ema_s_now = df['ema_slow'].iloc[-1]
    rsi_now = df['rsi'].iloc[-1]
    atr_now = df.get('atr', pd.Series([0])).iloc[-1]

    is_bullish = (ema_f_now > ema_s_now) and (last['close'] > ema_f_now * 0.999)
    is_bearish = (ema_f_now < ema_s_now) and (last['close'] < ema_f_now * 1.001)

    slope_fast = (ema_f_now - df['ema_fast'].iloc[-4]) / 3 if len(df) >= 4 else 0
    atr_ok = atr_now > last['close'] * 0.002 if use_atr_filter else True
    vol_ok = volume_surge(df) if use_volume_filter else True

    if is_bullish and buy_min < rsi_now < buy_max:
        if use_slope_filter and slope_fast <= 0: return None
        if higher_trend == 'down': return None
        if not atr_ok or not vol_ok: return None
        return ('BUY', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)

    if is_bearish and sell_min < rsi_now < sell_max:
        if use_slope_filter and slope_fast >= 0: return None
        if higher_trend == 'up': return None
        if not atr_ok or not vol_ok: return None
        return ('SELL', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)
    return None

# ========== 核心优化2：ATR动态止损/止盈 ==========
def calculate_sltp(entry_price, side, atr=None, use_atr=False, atr_mult_sl=2.55, atr_mult_tp1=1.0, atr_mult_tp2=2.0):
    if use_atr and atr and atr > 0:
        risk = atr * atr_mult_sl
        if side == 'BUY':
            return entry_price - risk, entry_price + risk * atr_mult_tp1, entry_price + risk * atr_mult_tp2
        else:
            return entry_price + risk, entry_price - risk * atr_mult_tp1, entry_price - risk * atr_mult_tp2
    
    if side == 'BUY':
        return entry_price * 0.994, entry_price * 1.006, entry_price * 1.012
    return entry_price * 1.006, entry_price * 0.994, entry_price * 0.988

# ---------- 信号历史管理 ----------
def update_signal_stats():
    total = len(st.session_state.signal_history)
    win = sum(1 for s in st.session_state.signal_history if s.get('result') == 'win')
    loss = sum(1 for s in st.session_state.signal_history if s.get('result') == 'loss')
    pending = sum(1 for s in st.session_state.signal_history if s.get('result') == 'pending')
    win_rate = round(win / (win + loss) * 100, 1) if (win + loss) > 0 else 0
    st.session_state.signal_stats = {'total': total, 'win': win, 'loss': loss, 'pending': pending, 'win_rate': win_rate}

def add_signal_to_history(signal, sl, tp1, tp2, signal_time_str):
    record = {
        'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'signal_time': signal_time_str,
        'side': signal[0],
        'price': signal[1],
        'ema_fast': signal[2],
        'ema_slow': signal[3],
        'rsi': signal[4],
        'atr': signal[5] if len(signal) > 5 else None,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'result': 'pending',
        'exit_price': None,
        'exit_time': None,
        'exit_reason': '',
        'peak': signal[1],
        'note': ''
    }
    st.session_state.signal_history.insert(0, record)
    if len(st.session_state.signal_history) > 200:
        st.session_state.signal_history = st.session_state.signal_history[:200]
    update_signal_stats()

def update_signal_result(index, result, exit_price=None, exit_reason='', note=''):
    if 0 <= index < len(st.session_state.signal_history):
        st.session_state.signal_history[index]['result'] = result
        if exit_price is not None and exit_price > 0:
            st.session_state.signal_history[index]['exit_price'] = round(exit_price, 2)
            st.session_state.signal_history[index]['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if exit_reason: st.session_state.signal_history[index]['exit_reason'] = exit_reason
        if note: st.session_state.signal_history[index]['note'] = note
        update_signal_stats()

def clear_signal_history():
    st.session_state.signal_history = []
    update_signal_stats()

def check_trailing_stop(record, current_price, trailing_dist):
    side = record['side']
    peak = record['peak']
    if side == 'BUY':
        trailing_sl = peak * (1 - trailing_dist / 100)
        if current_price <= trailing_sl:
            return True, 'loss', trailing_sl, '移动止损触发'
    else:
        trailing_sl = peak * (1 + trailing_dist / 100)
        if current_price >= trailing_sl:
            return True, 'loss', trailing_sl, '移动止损触发'
    return False, None, None, None

# ---------- Streamlit ----------
st.set_page_config(page_title="ETH 5分钟策略 (职业终极版)", layout="wide")
st.title("📈 ETH 5分钟 EMA 剥头皮策略 (职业整合版)")

with st.expander("📘 新手快速上手指南（点击展开）", expanded=True):
    st.markdown("""
    ### 如何根据信号下单？
    当顶部出现 🟢 多头信号 或 🔴 空头信号 时，请按以下步骤操作：
    1. **进场价格**=信号卡片中的“进场价格”数值（例如 1864.56）
    2. **止损价格**=信号卡片中的“止损价格”数值（例如 1853.37）
    3. 在交易所（如OKX）以市价单或限价单买入/卖出，价格尽量接近进场价。
    4. 同时设置止损单和止盈单（TP1平半仓，TP2全平）。
    """)

# ---------- 侧边栏 ----------
st.sidebar.header("策略参数")
fast_ema = st.sidebar.number_input("快线 EMA", 1, 50, 9, 1)
slow_ema = st.sidebar.number_input("慢线 EMA", 2, 100, 21, 1)
rsi_period = st.sidebar.number_input("RSI 周期", 2, 50, 14, 1)
buy_min = st.sidebar.number_input("多头 RSI 下限", 0, 100, 50, 1)
buy_max = st.sidebar.number_input("多头 RSI 上限", 0, 100, 70, 1)
sell_min = st.sidebar.number_input("空头 RSI 下限", 0, 100, 30, 1)
sell_max = st.sidebar.number_input("空头 RSI 上限", 0, 100, 50, 1)
refresh_interval = st.sidebar.number_input("刷新间隔(秒)", 5, 300, 60, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("✨ 高级过滤")
use_slope_filter = st.sidebar.checkbox("启用斜率过滤", value=True)
use_volume_filter = st.sidebar.checkbox("启用成交量爆发过滤", value=True)
use_atr_filter = st.sidebar.checkbox("启用波动率过滤", value=True)
use_higher_tf_filter = st.sidebar.checkbox("启用高周期趋势过滤", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 动态止损")
use_atr_sl = st.sidebar.checkbox("启用ATR动态止损", value=True)
if use_atr_sl:
    atr_mult_sl = st.sidebar.slider("ATR止损倍数", 0.5, 4.0, 2.55, 0.05)
    atr_mult_tp1 = st.sidebar.slider("ATR TP1倍数 (RR 1:1)", 0.5, 3.0, 1.0, 0.05)
    atr_mult_tp2 = st.sidebar.slider("ATR TP2倍数 (RR 2:1)", 1.0, 5.0, 2.0, 0.05)
else:
    atr_mult_sl = atr_mult_tp1 = atr_mult_tp2 = 1.2

st.sidebar.markdown("---")
st.sidebar.subheader("✨ 移动止损")
use_trailing = st.sidebar.checkbox("启用移动止损", value=False)
trailing_distance = st.sidebar.slider("回调距离 (%)", 0.1, 2.0, 0.3, 0.1) if use_trailing else 0.3

sound_enabled = st.sidebar.checkbox("🔊 启用信号声音提醒", value=True)

if st.sidebar.button("立即刷新数据"):
    st.cache_data.clear()
    st.rerun()
if st.sidebar.button("🔄 重置所有状态"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 初始化
if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=500)
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'signal_stats' not in st.session_state:
    st.session_state.signal_stats = {'total': 0, 'win': 0, 'loss': 0, 'pending': 0, 'win_rate': 0}
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = None

candle_buffer = st.session_state.candle_buffer
signal_history = st.session_state.signal_history

# 启动WebSocket（实时更新）
threading.Thread(target=run_ws, daemon=True).start()

# 获取初始历史K线
candles = fetch_klines()
if candles:
    for c in candles:
        if len(candle_buffer) == 0 or c[0] > candle_buffer[-1][0]:
            candle_buffer.append(c)

# 自动刷新（辅助WebSocket）
st_autorefresh(interval=refresh_interval * 1000, key="final")

higher_trend = get_higher_trend() if use_higher_tf_filter else 'neutral'
trend_icon = "🟢" if higher_trend == "up" else "🔴" if higher_trend == "down" else "⚪"
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线数量: {len(candle_buffer)} | 高周期趋势: {trend_icon} {higher_trend.upper()}")

if len(candle_buffer) < 30:
    st.warning(f"⏳ 数据积累中... 当前 {len(candle_buffer)}/30 根")
else:
    df = pd.DataFrame(list(candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)
    df['ema_fast'] = calculate_ema(df['close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['close'], slow_ema)
    df['rsi'] = calculate_rsi_wilder(df['close'], rsi_period)
    df['atr'] = calculate_atr(df, 14)

    signal = detect_signal_pro(df, fast_ema, slow_ema, rsi_period, buy_min, buy_max, sell_min, sell_max,
                               higher_trend, use_volume_filter, use_slope_filter, use_atr_filter)

    show_signal = signal
    if not show_signal and signal_history and len(signal_history) > 0 and signal_history[0]['result'] == 'pending':
        rec = signal_history[0]
        show_signal = (rec['side'], rec['price'], rec['ema_fast'], rec['ema_slow'], rec['rsi'], rec['atr'])

    if signal:
        side, price, ema_f, ema_s, rsi, atr_val = signal
        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        current_kline_time = df.index[-1]
        signal_time_str = current_kline_time.strftime('%Y-%m-%d %H:%M')
        if st.session_state.last_signal_time != current_kline_time:
            st.session_state.last_signal_time = current_kline_time
            add_signal_to_history(signal, sl, tp1, tp2, signal_time_str)
            if sound_enabled:
                st.markdown("""
                <script>
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = 880;
                var g = ctx.createGain(); g.gain.value = 0.4;
                o.connect(g); g.connect(ctx.destination);
                o.start(); setTimeout(() => o.stop(), 180);
                </script>
                """, unsafe_allow_html=True)

    if signal_history and signal_history[0]['result'] == 'pending':
        rec = signal_history[0]
        cp = df['close'].iloc[-1]
        if rec['side'] == 'BUY' and cp > rec['peak']: rec['peak'] = cp
        elif rec['side'] == 'SELL' and cp < rec['peak']: rec['peak'] = cp

    cp = df['close'].iloc[-1]
    for i, r in enumerate(signal_history):
        if r['result'] != 'pending': continue
        s = r['side']; sl = r['sl']; tp2 = r['tp2']
        if s == 'BUY':
            if cp <= sl: update_signal_result(i, 'loss', cp, '止损触发')
            elif cp >= tp2: update_signal_result(i, 'win', cp, 'TP2触发')
        else:
            if cp >= sl: update_signal_result(i, 'loss', cp, '止损触发')
            elif cp <= tp2: update_signal_result(i, 'win', cp, 'TP2触发')
        if use_trailing:
            exit_flag, res, ep, reason = check_trailing_stop(r, cp, trailing_distance)
            if exit_flag:
                update_signal_result(i, res, ep, reason)

    # 豪华美化信号卡片（保持不变）
    # ... (此处省略信号卡片代码，以保持完整性，你可以从之前复制）

    # 美化图表（保持不变）
    # ... (此处省略图表代码）

# ---------- 侧边栏统计和历史记录（保持不变） ----------
# ...

st.markdown("---")
st.caption("🔥 终极职业版 • 实时WebSocket同步 + 持续信号 + ATR动态止损 • 祝交易大赚！💰")
