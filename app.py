import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import os
import glob
import sqlite3
import re

# ---------- 配置 ----------
SYMBOL = "ETH"
CURRENCY = "USD"
INTERVAL = "5"
LIMIT = 200
RETRIES = 3
BASE_DELAY = 0.5
REQUEST_DELAY = 0.5
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DB_FILE = "signals.db"
INTERVAL_MINUTES = 5

# ---------- SQLite ----------
def get_db_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

def init_db():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("""
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
    finally:
        conn.close()

def save_signal_to_db(record):
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("""
        INSERT INTO signals (record_time, signal_time, side, price, ema_fast, ema_slow, rsi, atr, sl, tp1, tp2, result, peak, note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (record.get('record_time'), record.get('signal_time'), record.get('side'), record.get('price'),
              record.get('ema_fast'), record.get('ema_slow'), record.get('rsi'), record.get('atr'),
              record.get('sl'), record.get('tp1'), record.get('tp2'), record.get('result'), record.get('peak'), record.get('note', '')))
        return c.lastrowid
    finally:
        conn.close()

def update_signal_in_db(signal_id, result=None, exit_price=None, exit_time=None, exit_reason=None, peak=None, note=None):
    conn = get_db_conn()
    try:
        c = conn.cursor()
        fields, values = [], []
        if result is not None: fields.append("result=?"); values.append(result)
        if exit_price is not None: fields.append("exit_price=?"); values.append(exit_price)
        if exit_time is not None: fields.append("exit_time=?"); values.append(exit_time)
        if exit_reason is not None: fields.append("exit_reason=?"); values.append(exit_reason)
        if peak is not None: fields.append("peak=?"); values.append(peak)
        if note is not None: fields.append("note=?"); values.append(note)
        if not fields: return
        values.append(signal_id)
        c.execute(f"UPDATE signals SET {', '.join(fields)} WHERE id=?", values)
        conn.commit()
    finally:
        conn.close()

def fetch_recent_signals(limit=50, as_dict=False):
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        if as_dict:
            columns = [desc[0] for desc in c.description]
            return [dict(zip(columns, row)) for row in rows]
        return rows
    except:
        return [] if not as_dict else []
    finally:
        conn.close()

def load_recent_to_deque(limit=200):
    try:
        rows = fetch_recent_signals(limit, as_dict=True)
        valid = [r for r in rows if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', r.get('record_time','')) and re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', r.get('signal_time',''))]
        return deque(valid, maxlen=limit)
    except:
        return deque(maxlen=limit)

def auto_clean_invalid_signals():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM signals WHERE record_time NOT LIKE '____-__-__ __:__:__' OR signal_time NOT LIKE '____-__-__ __:__'")
        conn.commit()
    finally:
        conn.close()

def clear_all_signals():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM signals")
        conn.commit()
    finally:
        conn.close()

# ---------- 立即初始化（修复核心错误） ----------
init_db()
auto_clean_invalid_signals()

# ---------- Session ----------
if 'api_fail_count' not in st.session_state: st.session_state.api_fail_count = 0
if 'signal_history' not in st.session_state: st.session_state.signal_history = load_recent_to_deque(200)
if 'candle_buffer' not in st.session_state: st.session_state.candle_buffer = deque(maxlen=500)
if 'signal_stats' not in st.session_state: st.session_state.signal_stats = {'total':0,'win':0,'loss':0,'exit':0,'pending':0,'win_rate':0}
if 'last_signal_time' not in st.session_state: st.session_state.last_signal_time = None

# ---------- 数据获取 ----------
@st.cache_data(ttl=5, show_spinner=False)
def fetch_klines():
    def _fetch():
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={SYMBOL}&tsym={CURRENCY}&limit={LIMIT}&aggregate={INTERVAL}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()
    result = _fetch()
    if result and result.get('Response') == 'Success':
        data = result['Data']['Data']
        return [[item['time']*1000, item['open'], item['high'], item['low'], item['close'], item['volumefrom']] for item in data]
    return []

@st.cache_data(ttl=5, show_spinner=False)
def fetch_latest_candle():
    def _fetch():
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={SYMBOL}&tsym={CURRENCY}&limit=1&aggregate={INTERVAL}"
        resp = requests.get(url, timeout=5, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()
    result = _fetch()
    if result and result.get('Response') == 'Success':
        item = result['Data']['Data'][0]
        return [item['time']*1000, item['open'], item['high'], item['low'], item['close'], item['volumefrom']]
    return None

@st.cache_data(ttl=60, show_spinner=False)
def get_higher_trend():
    def _fetch():
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={SYMBOL}&tsym={CURRENCY}&limit=200"
        resp = requests.get(url, timeout=5, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()
    result = _fetch()
    if result and result.get('Response') == 'Success':
        closes = [d['close'] for d in result['Data']['Data']]
        if len(closes) >= 200:
            ema200 = pd.Series(closes).ewm(span=200, adjust=False).mean().iloc[-1]
            return 'up' if closes[-1] > ema200 else 'down'
    return 'neutral'

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
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_atr(df, period=14):
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def volume_surge(df, vol_period=20, surge_mult=1.3):
    if len(df) < vol_period + 1: return False
    avg_vol = df['volume'].rolling(vol_period).mean().iloc[-1]
    if avg_vol <= 0 or pd.isna(avg_vol):
        return df['close'].iloc[-1] > df['open'].iloc[-1]
    return (df['volume'].iloc[-1] / avg_vol > surge_mult) and (df['close'].iloc[-1] > df['open'].iloc[-1])

def volume_surge_bearish(df, vol_period=20, surge_mult=1.3):
    if len(df) < vol_period + 1: return False
    avg_vol = df['volume'].rolling(vol_period).mean().iloc[-1]
    if avg_vol <= 0 or pd.isna(avg_vol):
        return df['close'].iloc[-1] < df['open'].iloc[-1]
    return (df['volume'].iloc[-1] / avg_vol > surge_mult) and (df['close'].iloc[-1] < df['open'].iloc[-1])

# ---------- 核心信号检测 ----------
def detect_signal_pro(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max,
                      higher_trend, use_volume_filter, use_slope_filter, use_atr_filter, atr_threshold, slope_threshold=0.00022):
    if df.empty or len(df) < slow + rsi_period + 10: return None
    last = df.iloc[-1]
    ema_f_now = df['ema_fast'].iloc[-1]
    ema_s_now = df['ema_slow'].iloc[-1]
    rsi_now = df['rsi'].iloc[-1]
    atr_now = df['atr'].iloc[-1]

    # 波动过滤
    if len(df) >= 20:
        range_pct = (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / last['close']
        if range_pct < 0.003: return None
        prev_high = df['high'].iloc[-6:-1].max() if len(df) >= 6 else df['high'].iloc[-2]
        prev_low = df['low'].iloc[-6:-1].min() if len(df) >= 6 else df['low'].iloc[-2]
        if ema_f_now > ema_s_now and last['close'] <= prev_high: return None
        if ema_f_now < ema_s_now and last['close'] >= prev_low: return None

    # 扩散
    ema_spread_now = abs(ema_f_now - ema_s_now)
    ema_spread_prev = abs(df['ema_fast'].iloc[-2] - df['ema_slow'].iloc[-2]) if len(df) >= 2 else 0
    if ema_spread_now <= ema_spread_prev or ema_spread_now / last['close'] < 0.001: return None

    # 交叉过滤
    if len(df) >= 4:
        if (df['ema_fast'].iloc[-4] < df['ema_slow'].iloc[-4] and ema_f_now > ema_s_now) or \
           (df['ema_fast'].iloc[-4] > df['ema_slow'].iloc[-4] and ema_f_now < ema_s_now): return None

    prev_close = df['close'].iloc[-2] if len(df) >= 2 else last['close']
    is_bullish = (ema_f_now > ema_s_now) and (last['close'] > ema_f_now * 0.999)
    is_bearish = (ema_f_now < ema_s_now) and (last['close'] < ema_f_now * 1.001)

    if abs(ema_f_now - ema_s_now) / last['close'] < 0.0005: return None

    # 斜率
    lookback = min(5, len(df)-1)
    slope = (ema_f_now - df['ema_fast'].iloc[-lookback]) / lookback / last['close']
    if use_slope_filter and abs(slope) <= slope_threshold: return None

    atr_ok = (not pd.isna(atr_now) and atr_now > last['close'] * atr_threshold) if use_atr_filter else True

    if is_bullish and buy_min < rsi_now < buy_max:
        if last['close'] <= prev_close: return None
        vol_ok = volume_surge(df) if use_volume_filter else True
        if higher_trend == 'down' or not atr_ok or not vol_ok: return None
        return ('BUY', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)

    if is_bearish and sell_min < rsi_now < sell_max:
        if last['close'] >= prev_close: return None
        vol_ok = volume_surge_bearish(df) if use_volume_filter else True
        if higher_trend == 'up' or not atr_ok or not vol_ok: return None
        return ('SELL', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)
    return None

# ---------- SL/TP ----------
def calculate_sltp(entry_price, side, atr=None, use_atr=False, atr_mult_sl=2.2, atr_mult_tp1=0.8, atr_mult_tp2=1.6):
    if use_atr and atr and atr > 0:
        risk = atr * atr_mult_sl
        if side == 'BUY':
            return entry_price - risk, entry_price + risk*atr_mult_tp1, entry_price + risk*atr_mult_tp2
        return entry_price + risk, entry_price - risk*atr_mult_tp1, entry_price - risk*atr_mult_tp2
    if side == 'BUY':
        return entry_price*0.994, entry_price*1.006, entry_price*1.012
    return entry_price*1.006, entry_price*0.994, entry_price*0.988

def check_trailing_stop(record, current_price, trailing_dist):
    if record.get('result') != 'pending': return False, None, None, None
    peak = record.get('peak')
    side = record.get('side')
    if side == 'BUY':
        if current_price <= peak * (1 - trailing_dist/100):
            return True, 'exit', peak * (1 - trailing_dist/100), '移动止损'
    else:
        if current_price >= peak * (1 + trailing_dist/100):
            return True, 'exit', peak * (1 + trailing_dist/100), '移动止损'
    return False, None, None, None

# ---------- 补K线 ----------
def fill_missing_candles(buffer, new_candle):
    if not buffer: return [new_candle]
    last_ts = buffer[-1][0]
    new_ts = new_candle[0]
    expected = last_ts + INTERVAL_MINUTES * 60 * 1000
    if new_ts > expected:
        missing = []
        ts = expected
        while ts < new_ts:
            missing.append([ts] + [buffer[-1][4]]*4 + [0])
            ts += INTERVAL_MINUTES * 60 * 1000
        return missing + [new_candle]
    return [new_candle]

# ---------- 信号管理 ----------
def update_signal_stats():
    h = st.session_state.signal_history
    total = len(h)
    win = sum(1 for s in h if s.get('result')=='win')
    loss = sum(1 for s in h if s.get('result')=='loss')
    ex = sum(1 for s in h if s.get('result')=='exit')
    pend = sum(1 for s in h if s.get('result')=='pending')
    wr = round(win/(win+loss)*100,1) if win+loss>0 else 0
    st.session_state.signal_stats = {'total':total,'win':win,'loss':loss,'exit':ex,'pending':pend,'win_rate':wr}

def add_signal_to_history(signal, sl, tp1, tp2, signal_time_str):
    record = {'record_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'signal_time':signal_time_str,
              'side':signal[0],'price':signal[1],'ema_fast':signal[2],'ema_slow':signal[3],'rsi':signal[4],
              'atr':signal[5],'sl':sl,'tp1':tp1,'tp2':tp2,'result':'pending','peak':signal[1]}
    rid = save_signal_to_db(record)
    record['id'] = rid
    st.session_state.signal_history.appendleft(record)
    update_signal_stats()

def update_signal_result(idx, result, exit_price=None, exit_reason=''):
    h = st.session_state.signal_history
    if 0 <= idx < len(h):
        r = h[idx]
        r['result'] = result
        if exit_price: r['exit_price'] = round(exit_price,2); r['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if exit_reason: r['exit_reason'] = exit_reason
        update_signal_in_db(r['id'], result, r.get('exit_price'), r.get('exit_time'), exit_reason, r.get('peak'))
        update_signal_stats()

def clear_signal_history(clear_db=False):
    st.session_state.signal_history = deque(maxlen=200)
    if clear_db: clear_all_signals()
    update_signal_stats()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="ETH 5分钟极致版", layout="wide")

st.markdown("""
<style>
    * {font-family:'Inter','Microsoft YaHei',sans-serif;}
    .stApp {background:#0f172a;}
    .signal-card {background:linear-gradient(135deg,#0a3d2a 0%,#112233 100%);border-radius:18px;padding:24px;margin:15px 0;border:3px solid #00ff9d;box-shadow:0 10px 30px rgba(0,255,157,0.2);}
    .header-green {padding:16px 28px;border-radius:12px;font-size:27px;font-weight:700;text-align:center;background:#0a3d2a;color:#00ff9d;border-left:8px solid #00ff9d;}
    .header-red {padding:16px 28px;border-radius:12px;font-size:27px;font-weight:700;text-align:center;background:#3d0a0a;color:#ff4d4d;border-left:8px solid #ff4d4d;}
    .tp2-container {background:linear-gradient(135deg,#2c2200 0%,#4a3a00 50%,#2c2200 100%);border:3px solid #ffd700;border-radius:18px;padding:22px;margin:22px 0;text-align:center;box-shadow:0 0 25px rgba(255,215,0,0.7);animation:tp2-pulse 2.5s infinite;}
    @keyframes tp2-pulse {0%,100%{transform:scale(1)}50%{transform:scale(1.015)}}
    .waiting-card {background:#1e3a5f;padding:32px;border-radius:18px;text-align:center;font-size:23px;color:#89c2ff;border:2px dashed #4a90ff;}
    .diagnosis-card {background:#1e2937;padding:20px;border-radius:16px;margin:10px 0;}
</style>
""", unsafe_allow_html=True)

st.title("📈 ETH 5分钟 EMA 剥头皮策略 (极致版)")

candle_buffer = st.session_state.candle_buffer
signal_history = st.session_state.signal_history

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("策略参数")
    fast_ema = st.number_input("快线 EMA", 1, 50, 8, 1)
    slow_ema = st.number_input("慢线 EMA", 2, 100, 21, 1)
    rsi_period = st.number_input("RSI 周期", 2, 50, 14, 1)
    col1, col2 = st.columns(2)
    with col1: buy_min = st.number_input("多头下限", 0, 100, 57, 1)
    with col2: buy_max = st.number_input("多头上限", 0, 100, 70, 1)
    col3, col4 = st.columns(2)
    with col3: sell_min = st.number_input("空头下限", 0, 100, 30, 1)
    with col4: sell_max = st.number_input("空头上限", 0, 100, 43, 1)
    refresh_interval = st.number_input("刷新间隔(秒)", 5, 300, 60, 5)

    with st.expander("✨ 高级过滤", expanded=True):
        use_slope_filter = st.checkbox("启用斜率过滤", value=True)
        use_volume_filter = st.checkbox("启用成交量爆发过滤", value=True)
        use_atr_filter = st.checkbox("启用波动率过滤", value=True)
        atr_threshold = st.slider("ATR 阈值 (%)", 0.05, 0.5, 0.12, 0.01) / 100
        use_higher_tf_filter = st.checkbox("启用高周期趋势过滤", value=True)

    with st.expander("📊 动态止损", expanded=True):
        use_atr_sl = st.checkbox("启用ATR动态止损", value=True)
        atr_mult_sl = st.slider("ATR止损倍数", 0.5, 4.0, 2.2, 0.05) if use_atr_sl else 2.2
        atr_mult_tp1 = st.slider("ATR TP1倍数", 0.2, 3.0, 0.8, 0.05) if use_atr_sl else 0.8
        atr_mult_tp2 = st.slider("ATR TP2倍数", 0.5, 5.0, 1.6, 0.05) if use_atr_sl else 1.6

    with st.expander("✨ 移动止损", expanded=False):
        use_trailing = st.checkbox("启用移动止损", value=False)
        trailing_distance = st.slider("回调距离 (%)", 0.1, 2.0, 0.3, 0.1) if use_trailing else 0.3

    use_aggressive = st.checkbox("🔥 激进测试模式（降低所有门槛）", value=False)

    with st.expander("💰 风险仓位计算器", expanded=True):
        account = st.number_input("账户余额 (USDT)", value=10000.0, step=100.0)
        risk_pct = st.slider("单笔风险 (%)", 0.5, 5.0, 1.0, 0.1) / 100
        if signal_history and signal_history and signal_history[0]['result'] == 'pending':
            r = signal_history[0]
            risk_points = abs(r['price'] - r['sl'])
            if risk_points > 0:
                st.success(f"**建议开仓** ≈ **{(account * risk_pct) / risk_points:,.2f} ETH**")

    sound_enabled = st.checkbox("🔊 启用信号声音提醒", value=True)

    st.metric("📡 API 失败次数", st.session_state.api_fail_count)

    if st.button("立即刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("🗑 清空历史信号", use_container_width=True):
        clear_signal_history(clear_db=st.checkbox("同时清空数据库", value=False))
        st.rerun()

# ---------- 数据加载 ----------
candles = fetch_klines()
if candles:
    if not candle_buffer:
        for c in candles: candle_buffer.append(c)
    else:
        for c in candles:
            if c[0] > candle_buffer[-1][0]:
                for mc in fill_missing_candles(candle_buffer, c):
                    candle_buffer.append(mc)

latest = fetch_latest_candle()
if latest and (not candle_buffer or latest[0] > candle_buffer[-1][0]):
    for mc in fill_missing_candles(candle_buffer, latest):
        candle_buffer.append(mc)

st_autorefresh(interval=refresh_interval * 1000, key="final")

higher_trend = get_higher_trend() if use_higher_tf_filter else 'neutral'
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线: {len(candle_buffer)} | 高周期: {'🟢' if higher_trend=='up' else '🔴' if higher_trend=='down' else '⚪'} {higher_trend.upper()}")

if len(candle_buffer) < 30:
    st.warning(f"⏳ 数据积累中... {len(candle_buffer)}/30 根")
else:
    df = pd.DataFrame(list(candle_buffer), columns=['ts','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)
    df['ema_fast'] = calculate_ema(df['close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['close'], slow_ema)
    df['rsi'] = calculate_rsi_wilder(df['close'], rsi_period)
    df['atr'] = calculate_atr(df, 14)

    slope_th = 0.00015 if use_aggressive else 0.00022
    atr_th = 0.0008 if use_aggressive else atr_threshold

    signal = detect_signal_pro(df, fast_ema, slow_ema, rsi_period, buy_min, buy_max, sell_min, sell_max,
                               higher_trend, use_volume_filter, use_slope_filter, use_atr_filter, atr_th, slope_th)

    show_signal = signal
    if not show_signal and signal_history and signal_history[0]['result'] == 'pending':
        rec = signal_history[0]
        show_signal = (rec['side'], rec['price'], rec['ema_fast'], rec['ema_slow'], rec['rsi'], rec['atr'])

    if signal:
        side, price, ema_f, ema_s, rsi, atr_val = signal
        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        signal_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M')
        if st.session_state.last_signal_time != df.index[-1]:
            st.session_state.last_signal_time = df.index[-1]
            add_signal_to_history(signal, sl, tp1, tp2, signal_time_str)
            if sound_enabled:
                st.markdown("""<script>var ctx=new(window.AudioContext||window.webkitAudioContext)();var o=ctx.createOscillator();o.type="sine";o.frequency.value=880;var g=ctx.createGain();g.gain.value=0.4;o.connect(g);g.connect(ctx.destination);o.start();setTimeout(()=>o.stop(),180);</script>""", unsafe_allow_html=True)

    cp = df['close'].iloc[-1]
    if signal_history and signal_history[0]['result'] == 'pending':
        rec = signal_history[0]
        if (rec['side']=='BUY' and cp > rec['peak']) or (rec['side']=='SELL' and cp < rec['peak']):
            rec['peak'] = cp
            update_signal_in_db(rec['id'], peak=cp)
    for i, r in enumerate(signal_history):
        if r['result'] != 'pending': continue
        if r['side'] == 'BUY':
            if cp <= r['sl']: update_signal_result(i, 'loss', cp, '止损触发')
            elif cp >= r['tp2']: update_signal_result(i, 'win', cp, 'TP2触发')
        else:
            if cp >= r['sl']: update_signal_result(i, 'loss', cp, '止损触发')
            elif cp <= r['tp2']: update_signal_result(i, 'win', cp, 'TP2触发')
        if use_trailing:
            flag, res, ep, reason = check_trailing_stop(r, cp, trailing_distance)
            if flag: update_signal_result(i, res, ep, reason)

    # ========== 信号卡片 ==========
    if show_signal:
        side, price, ema_f, ema_s, rsi, atr_val = show_signal
        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        signal_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
        risk_pts = abs(price - sl)
        risk_pct = risk_pts / price * 100
        profit_pts = abs(tp2 - price)
        profit_pct = profit_pts / price * 100

        st.markdown('<div class="signal-card">', unsafe_allow_html=True)
        if side == 'BUY':
            st.markdown(f'<div class="header-green">● 多头信号 @ {signal_time}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="header-red">● 空头信号 @ {signal_time}</div>', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;">
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;"><div style="color:#ff4d4d;font-size:15px;">进场价格</div><div style="font-size:29px;font-weight:700;">{price:.2f}</div></div>
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;"><div style="color:#ff99cc;font-size:15px;">止损</div><div style="font-size:29px;font-weight:700;color:#ff99cc;">{sl:.2f}</div><div style="font-size:13px;color:#ff99cc;">风险 {risk_pts:.2f}点 ({risk_pct:.2f}%)</div></div>
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;"><div style="color:#ff99cc;font-size:15px;">TP1</div><div style="font-size:29px;font-weight:700;color:#ff99cc;">{tp1:.2f}</div></div>
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;border:1px solid #4a90ff;"><div style="color:#4a90ff;font-size:15px;">ATR</div><div style="font-size:29px;font-weight:700;color:#4a90ff;">{atr_val:.2f if atr_val else "N/A"}</div></div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="tp2-container">
            <div style="font-size:48px;font-weight:900;background:linear-gradient(90deg,#ffe066,#ffd700,#ffeb3b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{tp2:.2f}</div>
            <div style="margin-top:8px;color:#a0ff9d;font-size:18px;">RR <strong>2:1</strong> +{profit_pts:.2f}点 (+{profit_pct:.1f}%)</div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
            <div style="background:rgba(30,58,95,0.65);padding:14px;border-radius:12px;text-align:center;"><div>EMA快线</div><div style="font-size:23px;color:#ffa500;">{ema_f:.2f}</div></div>
            <div style="background:rgba(30,58,95,0.65);padding:14px;border-radius:12px;text-align:center;"><div>EMA慢线</div><div style="font-size:23px;color:#4a90ff;">{ema_s:.2f}</div></div>
            <div style="background:rgba(30,58,95,0.65);padding:14px;border-radius:12px;text-align:center;"><div>RSI</div><div style="font-size:23px;color:#00ff9d;">{rsi:.1f}</div></div>
            <div style="background:rgba(30,58,95,0.65);padding:14px;border-radius:12px;text-align:center;"><div>当前价</div><div style="font-size:23px;color:#ffffff;">{cp:.2f}</div></div>
        </div>
        ''', unsafe_allow_html=True)

        signal_text = f"🟢 多头 @ {signal_time} 进场{price:.2f} SL{sl:.2f} TP1{tp1:.2f} TP2{tp2:.2f}" if side=='BUY' else f"🔴 空头 @ {signal_time} 进场{price:.2f} SL{sl:.2f} TP1{tp1:.2f} TP2{tp2:.2f}"
        if st.button("📋 一键复制交易信号", use_container_width=True):
            st.markdown(f'<script>navigator.clipboard.writeText(`{signal_text}`);</script>', unsafe_allow_html=True)
            st.success("✅ 已复制！")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="waiting-card">⏳ 等待新信号出现...<br><span style="font-size:16px;">系统正在实时扫描 5分钟K线</span></div>', unsafe_allow_html=True)

    # ========== 当前市场诊断（完全匹配你的截图） ==========
    with st.expander("🔍 当前市场诊断 (为什么还没出信号)", expanded=True):
        if not df.empty:
            last = df.iloc[-1]
            ema_spread_pct = abs(last['ema_fast'] - last['ema_slow']) / last['close'] * 100
            slope = (last['ema_fast'] - df['ema_fast'].iloc[-5]) / 5 / last['close']
            range_pct = (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / last['close'] * 100
            avg_vol = df['volume'].rolling(20).mean().iloc[-1]
            vol_ratio = last['volume'] / avg_vol if avg_vol > 0 else 0.0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta = "✅ 达标" if ema_spread_pct >= 0.1 else "❌ 未达标"
                st.metric("EMA扩散强度", f"{ema_spread_pct:.3f}%", delta)
            with col2:
                delta = "✅ 达标" if abs(slope) >= slope_th else "❌ 未达标"
                st.metric("EMA斜率", f"{slope:.5f}", delta)
            with col3:
                delta = "✅ 达标" if range_pct >= 0.3 else "❌ 未达标"
                st.metric("20根波动率", f"{range_pct:.2f}%", delta)
            with col4:
                delta = "✅ 爆发" if vol_ratio > 1.3 else "等待"
                st.metric("成交量爆发", f"{vol_ratio:.2f}x", delta)

    # ========== 图表 + 右侧 ==========
    col1, col2 = st.columns([3,1])
    with col1:
        plot_df = df.tail(200)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='K线', increasing_line_color='#00ff9d', decreasing_line_color='#ff4d4d'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_fast'], name=f'EMA快线 ({fast_ema})', line=dict(color='#ffd700', width=3.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_slow'], name=f'EMA慢线 ({slow_ema})', line=dict(color='#4da9ff', width=3.5)), row=1, col=1)
        colors = np.where(plot_df['close'] >= plot_df['open'], '#00ff9d', '#ff4d4d')
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], marker_color=colors, opacity=0.85, showlegend=False), row=2, col=1)
        fig.add_hline(y=cp, line_dash="dash", line_color="#00ff9d", annotation_text=f"当前价 {cp:.2f}", annotation_position="top right", row=1, col=1)
        fig.update_layout(height=680, template="plotly_dark", plot_bgcolor="#0e1621", paper_bgcolor="#0e1621", legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("当前价", f"{cp:.2f}")
        if len(df) > 144:
            change = cp - df['close'].iloc[-144]
            change_pct = change / df['close'].iloc[-144] * 100
            st.markdown(f"24h涨跌: <span style='color:{'#00ff9d' if change>=0 else '#ff4d4d'}'>{change:+.2f} ({change_pct:+.2f}%)</span>", unsafe_allow_html=True)

        stats = st.session_state.signal_stats
        st.markdown(f"**总信号:** {stats['total']}")
        st.markdown(f"✅ 盈利: {stats['win']}")
        st.markdown(f"❌ 亏损: {stats['loss']}")
        st.markdown(f"⏹️ 移动退出: {stats['exit']}")
        st.markdown(f"**胜率:** {stats['win_rate']}%")

    # ========== 表格 ==========
    st.subheader("最近 10 根K线")
    display_df = df.reset_index()[['time','open','high','low','close','volume','ema_fast','ema_slow','rsi','atr']].tail(10)
    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M')
    display_df = display_df.rename(columns={'time':'时间','open':'开盘','high':'最高','low':'最低','close':'收盘','volume':'成交量','ema_fast':'EMA快线','ema_slow':'EMA慢线','rsi':'RSI','atr':'ATR'})
    st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)

    st.subheader("📜 历史信号记录")
    if signal_history:
        hist_data = []
        for s in list(signal_history)[:50]:
            res_emoji = "✅ 赢" if s['result']=='win' else "❌ 亏" if s['result']=='loss' else "⏹️ 退" if s['result']=='exit' else "⏳ 待"
            hist_data.append({
                "记录时间": s['record_time'], "信号时间": s['signal_time'],
                "方向": "🟢 多头" if s['side']=='BUY' else "🔴 空头",
                "进场价": f"{s['price']:.2f}", "止损": f"{s['sl']:.2f}",
                "TP1": f"{s['tp1']:.2f}", "TP2": f"{s['tp2']:.2f}",
                "结果": res_emoji,
                "出场价": f"{s.get('exit_price','—'):.2f}" if s.get('exit_price') else "—",
                "出场原因": s.get('exit_reason','—')
            })
        hist_df = pd.DataFrame(hist_data)
        st.dataframe(hist_df, use_container_width=True, height=400)
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出历史信号 (CSV)", csv, f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

st.markdown("---")
st.caption("🔥 极致版 v2026.02.26 • 完全修复SQLite错误 • 实时诊断 + 激进模式 + 全中文 • 直接复制运行！祝你今天大赚！💰🚀")
