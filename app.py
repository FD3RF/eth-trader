import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import sqlite3
import ccxt

# ---------- 配置 ----------
SYMBOL = "ETH/USDT"
TIMEFRAME = "5m"          # 5分钟K线（可改为1h、4h等）
LIMIT = 200               # 获取K线数量
DB_FILE = "signals.db"

# ---------- 初始化币安交易所（只读） ----------
@st.cache_resource
def get_exchange():
    return ccxt.binance({
        'apiKey': st.secrets["BINANCE_API_KEY"],
        'secret': st.secrets["BINANCE_API_SECRET"],
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

# ---------- 获取K线数据 ----------
def fetch_klines():
    exchange = get_exchange()
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
        # ohlcv格式: [[timestamp, open, high, low, close, volume], ...]
        return ohlcv
    except Exception as e:
        st.error(f"获取币安K线失败: {e}")
        return []

# ---------- SQLite ----------
def get_db_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
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
        )""")

def save_signal_to_db(record):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO signals (record_time, signal_time, side, price, ema_fast, ema_slow, rsi, atr, sl, tp1, tp2, result, peak, note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (record['record_time'], record['signal_time'], record['side'], record['price'],
         record['ema_fast'], record['ema_slow'], record['rsi'], record['atr'],
         record['sl'], record['tp1'], record['tp2'], record['result'], record['peak'], record.get('note','')))
        return cur.lastrowid

def update_signal_in_db(signal_id, result=None, exit_price=None, exit_time=None, exit_reason=None, peak=None, note=None):
    with get_db_conn() as conn:
        fields = []
        params = []
        if result is not None:
            fields.append("result=?")
            params.append(result)
        if exit_price is not None:
            fields.append("exit_price=?")
            params.append(exit_price)
        if exit_time is not None:
            fields.append("exit_time=?")
            params.append(exit_time)
        if exit_reason is not None:
            fields.append("exit_reason=?")
            params.append(exit_reason)
        if peak is not None:
            fields.append("peak=?")
            params.append(peak)
        if note is not None:
            fields.append("note=?")
            params.append(note)
        if not fields:
            return
        params.append(signal_id)
        conn.execute(f"UPDATE signals SET {', '.join(fields)} WHERE id=?", params)

def fetch_recent_signals(limit=50, as_dict=False):
    with get_db_conn() as conn:
        df = pd.read_sql("SELECT * FROM signals ORDER BY id DESC LIMIT ?", conn, params=(limit,))
        if as_dict:
            return df.to_dict('records')
        return df.values.tolist()

def load_recent_to_deque(limit=200):
    try:
        rows = fetch_recent_signals(limit, as_dict=True)
        return deque(rows, maxlen=limit)
    except:
        return deque(maxlen=limit)

def clear_all_signals():
    with get_db_conn() as conn:
        conn.execute("DELETE FROM signals")

init_db()

# ---------- Session ----------
if 'api_fail_count' not in st.session_state: st.session_state.api_fail_count = 0
if 'signal_history' not in st.session_state: st.session_state.signal_history = load_recent_to_deque(200)
if 'candle_buffer' not in st.session_state: st.session_state.candle_buffer = deque(maxlen=500)
if 'signal_stats' not in st.session_state: st.session_state.signal_stats = {'total':0,'win':0,'loss':0,'exit':0,'pending':0,'win_rate':0}
if 'last_signal_time' not in st.session_state: st.session_state.last_signal_time = None

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
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
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

# ---------- 信号评分系统（含双档量能） ----------
def calculate_signal_score(side, df, idx=-1):
    row = df.iloc[idx]
    close = row['close']
    ema_f = row['ema_fast']
    ema_s = row['ema_slow']
    rsi = row['rsi']
    atr = row['atr'] if not pd.isna(row['atr']) else 0
    volume = row['volume']

    if len(df) >= 20 and idx >= 0:
        avg_vol = df['volume'].iloc[max(0, idx-20):idx].mean() if idx > 0 else df['volume'].iloc[-20:].mean()
    else:
        avg_vol = volume
    vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

    ema_spread = abs(ema_f - ema_s) / close * 100

    if len(df) >= 5 and idx >= 4:
        slope = (ema_f - df['ema_fast'].iloc[idx-4]) / 5 / close * 100
        slope_abs = abs(slope)
    else:
        slope_abs = 0

    atr_pct = (atr / close * 100) if atr > 0 else 0

    def score_by_threshold(value, thresholds):
        for low, high, s in thresholds:
            if low <= value <= high:
                return s
        return 0

    ema_thresholds = [(0.2, 100, 20), (0.15, 0.2, 15), (0.1, 0.15, 10), (0.05, 0.1, 5), (0, 0.05, 0)]
    score_ema = score_by_threshold(ema_spread, ema_thresholds)

    slope_thresholds = [(0.04, 100, 20), (0.03, 0.04, 15), (0.02, 0.03, 10), (0.01, 0.02, 5), (0, 0.01, 0)]
    score_slope = score_by_threshold(slope_abs, slope_thresholds)

    # 量能双档触发
    if vol_ratio >= 1.3:
        score_vol = 20
    elif vol_ratio >= 0.8:
        score_vol = 10
    else:
        score_vol = 0

    atr_thresholds = [(0.2, 100, 20), (0.16, 0.2, 15), (0.12, 0.16, 10), (0.08, 0.12, 5), (0, 0.08, 0)]
    score_atr = score_by_threshold(atr_pct, atr_thresholds)

    if side == 'BUY':
        if 55 <= rsi <= 70:
            score_rsi = 20
        elif 50 <= rsi < 55:
            score_rsi = 15
        elif 45 <= rsi < 50:
            score_rsi = 10
        elif 40 <= rsi < 45:
            score_rsi = 5
        elif 70 < rsi <= 75:
            score_rsi = 15
        elif 75 < rsi <= 80:
            score_rsi = 10
        elif 80 < rsi <= 85:
            score_rsi = 5
        else:
            score_rsi = 0
    else:
        if 30 <= rsi <= 45:
            score_rsi = 20
        elif 25 <= rsi < 30:
            score_rsi = 15
        elif 20 <= rsi < 25:
            score_rsi = 10
        elif 15 <= rsi < 20:
            score_rsi = 5
        elif 45 < rsi <= 50:
            score_rsi = 15
        elif 50 < rsi <= 55:
            score_rsi = 10
        elif 55 < rsi <= 60:
            score_rsi = 5
        else:
            score_rsi = 0

    total = score_ema + score_slope + score_vol + score_atr + score_rsi
    return total, {'EMA': score_ema, '斜率': score_slope, '量能': score_vol, 'ATR': score_atr, 'RSI': score_rsi}

# ---------- 核心信号检测（方向优先） ----------
def detect_signal_pro(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max,
                      higher_trend, use_volume_filter, use_slope_filter, use_atr_filter, atr_threshold, slope_threshold=0.00022,
                      use_scoring=False, score_threshold=80):
    if df.empty or len(df) < slow + rsi_period + 10: return None
    last = df.iloc[-1]
    ema_f_now = df['ema_fast'].iloc[-1]
    ema_s_now = df['ema_slow'].iloc[-1]
    rsi_now = df['rsi'].iloc[-1]
    atr_now = df['atr'].iloc[-1]

    # 保留基本的波动范围检查
    if len(df) >= 20:
        range_pct = (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / last['close']
        if range_pct < 0.003: return None
        prev_high = df['high'].iloc[-6:-1].max() if len(df) >= 6 else df['high'].iloc[-2]
        prev_low = df['low'].iloc[-6:-1].min() if len(df) >= 6 else df['low'].iloc[-2]
        if ema_f_now > ema_s_now and last['close'] <= prev_high: return None
        if ema_f_now < ema_s_now and last['close'] >= prev_low: return None

    if abs(ema_f_now - ema_s_now) / last['close'] < 0.0005: return None

    lookback = min(5, len(df)-1)
    slope = (ema_f_now - df['ema_fast'].iloc[-lookback]) / lookback / last['close']

    prev_close = df['close'].iloc[-2] if len(df) >= 2 else last['close']
    is_bullish = (ema_f_now > ema_s_now) and (last['close'] > ema_f_now * 0.999)
    is_bearish = (ema_f_now < ema_s_now) and (last['close'] < ema_f_now * 1.001)

    if is_bullish:
        if slope <= 0: return None
        if last['close'] <= prev_close: return None
        if higher_trend == 'down': return None
        if use_scoring:
            total_score, subscores = calculate_signal_score('BUY', df)
            if total_score >= score_threshold:
                return ('BUY', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now, total_score, subscores)
            else:
                return None
        else:
            if not (buy_min < rsi_now < buy_max): return None
            vol_ok = volume_surge(df) if use_volume_filter else True
            if not vol_ok: return None
            if use_slope_filter and abs(slope) <= slope_threshold: return None
            if use_atr_filter and (pd.isna(atr_now) or atr_now <= last['close'] * atr_threshold): return None
            return ('BUY', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)

    if is_bearish:
        if slope >= 0: return None
        if last['close'] >= prev_close: return None
        if higher_trend == 'up': return None
        if use_scoring:
            total_score, subscores = calculate_signal_score('SELL', df)
            if total_score >= score_threshold:
                return ('SELL', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now, total_score, subscores)
            else:
                return None
        else:
            if not (sell_min < rsi_now < sell_max): return None
            vol_ok = volume_surge_bearish(df) if use_volume_filter else True
            if not vol_ok: return None
            if use_slope_filter and abs(slope) <= slope_threshold: return None
            if use_atr_filter and (pd.isna(atr_now) or atr_now <= last['close'] * atr_threshold): return None
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
    if len(signal) > 6:
        side, price, ema_f, ema_s, rsi, atr_val, total_score, subscores = signal
        note = f"评分:{total_score} EMA:{subscores['EMA']} 斜率:{subscores['斜率']} 量能:{subscores['量能']} ATR:{subscores['ATR']} RSI:{subscores['RSI']}"
    else:
        side, price, ema_f, ema_s, rsi, atr_val = signal
        note = ""
    record = {'record_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'signal_time':signal_time_str,
              'side':side,'price':price,'ema_fast':ema_f,'ema_slow':ema_s,'rsi':rsi,
              'atr':atr_val,'sl':sl,'tp1':tp1,'tp2':tp2,'result':'pending','peak':price, 'note':note}
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
st.set_page_config(page_title="ETH 5分钟极致版 (币安数据)", layout="wide")

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
    .score-badge {display:flex;justify-content:space-around;background:rgba(0,255,157,0.1);padding:12px;border-radius:12px;margin:12px 0;}
    .score-item {text-align:center;}
    .score-value {font-size:24px;font-weight:bold;color:#00ff9d;}
    .stat-row {display:flex;justify-content:space-between;background:#1e293b;padding:10px 15px;border-radius:12px;margin:10px 0;}
</style>
""", unsafe_allow_html=True)

st.title("📈 ETH 5分钟 EMA 剥头皮策略 (币安数据·三层松绑)")

candle_buffer = st.session_state.candle_buffer
signal_history = st.session_state.signal_history

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("策略参数")
    fast_ema = st.number_input("快线 EMA", 1, 50, 8, 1)
    slow_ema = st.number_input("慢线 EMA", 2, 100, 21, 1)
    rsi_period = st.number_input("RSI 周期", 2, 50, 14, 1)
    col1, col2 = st.columns(2)
    with col1: 
        st.caption("多头下限")
        buy_min = st.number_input("多头下限", 0, 100, 57, 1, label_visibility="collapsed")
    with col2: 
        st.caption("多头上限")
        buy_max = st.number_input("多头上限", 0, 100, 70, 1, label_visibility="collapsed")
    col3, col4 = st.columns(2)
    with col3: 
        st.caption("空头下限")
        sell_min = st.number_input("空头下限", 0, 100, 30, 1, label_visibility="collapsed")
    with col4: 
        st.caption("空头上限")
        sell_max = st.number_input("空头上限", 0, 100, 43, 1, label_visibility="collapsed")
    refresh_interval = st.number_input("刷新间隔(秒)", 5, 300, 60, 5)
    st.caption(f"⏳ 下次刷新: {refresh_interval} 秒后")

    with st.expander("📊 信号分析系统", expanded=True):
        use_scoring = st.checkbox("启用评分系统（综合评分≥阈值才触发）", value=True)
        score_threshold = st.slider("评分阈值", 0, 100, 80, 1)

    with st.expander("✨ 高级过滤", expanded=True):
        use_slope_filter = st.checkbox("启用斜率过滤", value=True)
        use_volume_filter = st.checkbox("启用成交量滤波", value=True)
        use_atr_filter = st.checkbox("启用波动率过滤", value=True)
        atr_threshold = st.slider("ATR 阈值 (%)", 0.05, 0.5, 0.12, 0.01) / 100
        use_higher_tf_filter = st.checkbox("启用高周期趋势过滤", value=True)

    with st.expander("📊 动态止损", expanded=True):
        use_atr_sl = st.checkbox("启用ATR动态止损", value=True)
        atr_mult_sl = st.slider("ATR止损倍数", 0.5, 4.0, 2.2, 0.05) if use_atr_sl else 2.2
        atr_mult_tp1 = st.slider("ATR TP1倍数", 0.2, 3.0, 0.8, 0.05) if use_atr_sl else 0.8
        atr_mult_tp2 = st.slider("ATR TP2倍数", 0.5, 5.0, 1.6, 0.05) if use_atr_sl else 1.6

    with st.expander("💰 风险仓位计算器（含分级）", expanded=True):
        account = st.number_input("账户余额 (USDT)", value=10000.0, step=100.0)
        risk_pct = st.slider("单笔风险 (%)", 0.5, 5.0, 1.0, 0.1) / 100
        
        if signal_history and signal_history[0]['result'] == 'pending':
            r = signal_history[0]
            score = None
            note = r.get('note', '')
            if '评分:' in note:
                try:
                    score = int(note.split('评分:')[1].split()[0])
                except: pass
            if score is not None:
                if score >= 85: level, factor = "🔴 强信号", 1.0
                elif score >= 75: level, factor = "🟡 普通信号", 0.7
                elif score >= 65: level, factor = "🟢 轻仓试单", 0.4
                else: level, factor = "⚪ 信号偏弱", 0
            else:
                level, factor = "信号（无评分）", 1.0
            risk_points = abs(r['price'] - r['sl'])
            if risk_points > 0 and factor > 0:
                base_qty = (account * risk_pct) / risk_points
                suggested_qty = base_qty * factor
                st.success(f"**{level}** 建议开仓 ≈ **{suggested_qty:,.2f} ETH** (基准 {base_qty:,.2f} × {factor})")
            elif risk_points > 0 and factor == 0:
                st.info("信号偏弱，不建议开仓")
        else:
            st.info("无待处理信号")

    sound_enabled = st.checkbox("🔊 启用信号声音提醒", value=True)
    st.metric("📡 API 失败次数", st.session_state.api_fail_count)

    if st.button("立即刷新数据", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    if st.button("🗑 清空历史信号", width='stretch'):
        clear_signal_history(clear_db=st.checkbox("同时清空数据库", value=False))
        st.rerun()

# ---------- 数据获取 ----------
klines = fetch_klines()
if klines:
    # 更新candle_buffer
    for k in klines:
        st.session_state.candle_buffer.append(k)

st_autorefresh(interval=refresh_interval * 1000, key="final")

# 高周期趋势（仍使用CryptoCompare，也可改用币安日线数据，为保持简单，保留原函数但可替换）
def get_higher_trend():
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym=ETH&tsym=USD&limit=200"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        if result.get('Response') == 'Success':
            closes = [d['close'] for d in result['Data']['Data']]
            if len(closes) >= 200:
                ema200 = pd.Series(closes).ewm(span=200).mean().iloc[-1]
                return 'up' if closes[-1] > ema200 else 'down'
        return 'neutral'
    except:
        return 'neutral'

higher_trend = get_higher_trend() if use_higher_tf_filter else 'neutral'
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 长线: {slow_ema} | 周期: {higher_trend.upper()}")

if len(st.session_state.candle_buffer) < 30:
    st.warning(f"⏳ 数据积累中... {len(st.session_state.candle_buffer)}/30 根")
else:
    df = pd.DataFrame(list(st.session_state.candle_buffer), columns=['ts','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)
    df['ema_fast'] = calculate_ema(df['close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['close'], slow_ema)
    df['rsi'] = calculate_rsi_wilder(df['close'], rsi_period)
    df['atr'] = calculate_atr(df, 14)

    slope_th = 0.00015 if False else 0.00022  # 激进模式已移除
    atr_th = atr_threshold

    signal = detect_signal_pro(df, fast_ema, slow_ema, rsi_period, buy_min, buy_max, sell_min, sell_max,
                               higher_trend, use_volume_filter, use_slope_filter, use_atr_filter, atr_th, slope_th,
                               use_scoring, score_threshold)

    cp = df['close'].iloc[-1]

    # 处理历史pending信号的止损止盈
    for i, r in enumerate(signal_history):
        if r['result'] != 'pending': continue
        if r['side'] == 'BUY':
            if cp <= r['sl']: update_signal_result(i, 'loss', cp, '止损触发')
            elif cp >= r['tp2']: update_signal_result(i, 'win', cp, 'TP2触发')
        else:
            if cp >= r['sl']: update_signal_result(i, 'loss', cp, '止损触发')
            elif cp <= r['tp2']: update_signal_result(i, 'win', cp, 'TP2触发')

    # 新信号处理
    show_signal = signal
    if not show_signal and signal_history and signal_history[0]['result'] == 'pending':
        rec = signal_history[0]
        show_signal = (rec['side'], rec['price'], rec['ema_fast'], rec['ema_slow'], rec['rsi'], rec['atr'])

    if signal and st.session_state.last_signal_time != df.index[-1]:
        if len(signal) > 6:
            side, price, ema_f, ema_s, rsi, atr_val, total_score, subscores = signal
        else:
            side, price, ema_f, ema_s, rsi, atr_val = signal
        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        signal_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M')
        st.session_state.last_signal_time = df.index[-1]
        add_signal_to_history(signal, sl, tp1, tp2, signal_time_str)
        if sound_enabled:
            st.markdown("""<script>var ctx=new AudioContext();var o=ctx.createOscillator();o.type="sine";o.frequency.value=880;o.connect(ctx.destination);o.start();o.stop(0.2);</script>""", unsafe_allow_html=True)

    # 显示信号卡片
    if show_signal:
        if len(show_signal) > 6:
            side, price, ema_f, ema_s, rsi, atr_val, total_score, subscores = show_signal
        else:
            side, price, ema_f, ema_s, rsi, atr_val = show_signal
            total_score, subscores = None, None

        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        signal_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
        risk_pts = abs(price - sl)
        profit_pts = abs(tp2 - price)

        st.markdown('<div class="signal-card">', unsafe_allow_html=True)
        if side == 'BUY':
            st.markdown(f'<div class="header-green">● 多头信号 @ {signal_time}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="header-red">● 空头信号 @ {signal_time}</div>', unsafe_allow_html=True)

        if total_score is not None:
            if total_score >= 85: level_tag = "🔴 强信号"
            elif total_score >= 75: level_tag = "🟡 普通信号"
            elif total_score >= 65: level_tag = "🟢 轻仓试单"
            else: level_tag = "⚪ 偏弱"
            st.markdown(f"""
            <div class="score-badge">
                <div class="score-item"><span>总分</span><br><span class="score-value">{total_score}</span><br><span style="font-size:14px;">{level_tag}</span></div>
                <div class="score-item"><span>EMA</span><br>{subscores['EMA']}</div>
                <div class="score-item"><span>斜率</span><br>{subscores['斜率']}</div>
                <div class="score-item"><span>量能</span><br>{subscores['量能']}</div>
                <div class="score-item"><span>ATR</span><br>{subscores['ATR']}</div>
                <div class="score-item"><span>RSI</span><br>{subscores['RSI']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'''
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;">
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;"><div style="color:#ff4d4d;font-size:15px;">进场价格</div><div style="font-size:29px;font-weight:700;">{price:.2f}</div></div>
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;"><div style="color:#ff99cc;font-size:15px;">止损</div><div style="font-size:29px;font-weight:700;color:#ff99cc;">{sl:.2f}</div><div style="font-size:13px;color:#ff99cc;">风险 {risk_pts:.2f}点</div></div>
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;"><div style="color:#ff99cc;font-size:15px;">TP1</div><div style="font-size:29px;font-weight:700;color:#ff99cc;">{tp1:.2f}</div></div>
            <div style="background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;text-align:center;border:1px solid #4a90ff;"><div style="color:#4a90ff;font-size:15px;">ATR</div><div style="font-size:29px;font-weight:700;color:#4a90ff;">{atr_val:.2f}</div></div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'<div class="tp2-container"><div style="font-size:48px;font-weight:900;">{tp2:.2f}</div><div>RR 目标</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="waiting-card">⏳ 等待新信号出现...</div>', unsafe_allow_html=True)

    # 图表
    plot_df = df.tail(200)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='K线', increasing_line_color='#00ff9d', decreasing_line_color='#ff4d4d'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_fast'], name=f'EMA{fast_ema}', line=dict(color='#ffd700')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_slow'], name=f'EMA{slow_ema}', line=dict(color='#4da9ff')), row=1, col=1)
    colors = np.where(plot_df['close'] >= plot_df['open'], '#00ff9d', '#ff4d4d')
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_hline(y=cp, line_dash="dash", line_color="#00ff9d", annotation_text=f"{cp:.2f}", row=1, col=1)
    fig.update_layout(height=600, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 市场诊断
    with st.expander("🔍 当前市场诊断", expanded=True):
        last = df.iloc[-1]
        ema_spread = abs(last['ema_fast']-last['ema_slow'])/last['close']*100
        slope_val = (last['ema_fast']-df['ema_fast'].iloc[-5])/5/last['close'] if len(df)>=5 else 0
        range_pct = (df['high'].iloc[-20:].max()-df['low'].iloc[-20:].min())/last['close']*100
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        vol_ratio = last['volume']/avg_vol if avg_vol>0 else 0
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**EMA扩散度** {ema_spread:.3f}% {'✅' if ema_spread>=0.1 else '❌'}")
            st.markdown(f"**EMA斜率** {slope_val:.5f} {'✅' if abs(slope_val)>=0.00022 else '❌'}")
        with col2:
            st.markdown(f"**20根波动率** {range_pct:.2f}% {'✅' if range_pct>=0.3 else '❌'}")
            st.markdown(f"**成交量倍数** {vol_ratio:.2f}x {'✅' if vol_ratio>1.3 else '⏳'}")

    # 价格统计
    if len(df) >= 288:
        change_24h = cp - df['close'].iloc[-288]
        change_pct = change_24h / df['close'].iloc[-288] * 100
        change_display = f"{change_24h:+.2f} ({change_pct:+.2f}%)"
        color = '#00ff9d' if change_24h>=0 else '#ff4d4d'
    else:
        change_display, color = "N/A", "#888"

    st.markdown(f"""
    <div class="stat-row">
        <span style="font-size:28px;">{cp:.2f}</span>
        <span>24h涨跌: <span style="color:{color}">{change_display}</span></span>
        <span>盈利: {st.session_state.signal_stats['win']}</span>
        <span>亏损: {st.session_state.signal_stats['loss']}</span>
        <span>胜率: {st.session_state.signal_stats['win_rate']}%</span>
    </div>
    """, unsafe_allow_html=True)

    # 最近K线表格
    st.subheader("最近10根K线")
    display_df = df.reset_index()[['time','open','high','low','close','volume','ema_fast','ema_slow','rsi','atr']].tail(10)
    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M')
    display_df.columns = ['时间','开盘','最高','最低','收盘','成交量','EMA快线','EMA慢线','RSI','ATR']
    st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)

    # 历史信号记录
    st.subheader("📜 历史信号")
    if signal_history:
        hist_df = pd.DataFrame(list(signal_history)[:50])
        st.dataframe(hist_df[['signal_time','side','price','result','exit_price','exit_reason','note']], use_container_width=True, height=400)
        csv = hist_df.to_csv(index=False).encode()
        st.download_button("导出CSV", csv, "signals.csv")

st.markdown("---")
st.caption("🔥 策略基于币安实时数据 · 请确保API仅开启读取权限 · 已暴露的密钥请立即删除")
