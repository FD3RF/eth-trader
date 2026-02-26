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
SYMBOL = "ETH-USDT"                     # KuCoin 使用 ETH-USDT
INTERVAL = "5min"                        # KuCoin 5分钟为 "5min"
LIMIT = 200
RETRIES = 3
BASE_DELAY = 0.5
REQUEST_DELAY = 0.5
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
LOG_RETENTION_DAYS = 30
DB_FILE = "signals.db"
INTERVAL_MINUTES = 5                     # 5分钟K线

# ---------- SQLite 持久化 ----------
def get_db_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

def init_db():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_time TEXT,
            signal_time TEXT,
            side TEXT,
            price REAL,
            ema_fast REAL,
            ema_slow REAL,
            rsi REAL,
            atr REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            result TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            peak REAL,
            note TEXT
        )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_record_time ON signals(record_time);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_result ON signals(result);")
        conn.commit()
    finally:
        conn.close()
    try:
        delete_old_signals(30)
    except:
        pass

def save_signal_to_db(record):
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("""
        INSERT INTO signals
        (record_time, signal_time, side, price, ema_fast, ema_slow, rsi, atr,
         sl, tp1, tp2, result, peak, note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record.get('record_time'),
            record.get('signal_time'),
            record.get('side'),
            record.get('price'),
            record.get('ema_fast'),
            record.get('ema_slow'),
            record.get('rsi'),
            record.get('atr'),
            record.get('sl'),
            record.get('tp1'),
            record.get('tp2'),
            record.get('result'),
            record.get('peak'),
            record.get('note', '')
        ))
        signal_id = c.lastrowid
        conn.commit()
        return signal_id
    finally:
        conn.close()

def update_signal_in_db(signal_id, result=None, exit_price=None, exit_time=None,
                        exit_reason=None, peak=None, note=None):
    conn = get_db_conn()
    try:
        c = conn.cursor()
        fields = []
        values = []
        if result is not None:
            fields.append("result=?")
            values.append(result)
        if exit_price is not None:
            fields.append("exit_price=?")
            values.append(exit_price)
        if exit_time is not None:
            fields.append("exit_time=?")
            values.append(exit_time)
        if exit_reason is not None:
            fields.append("exit_reason=?")
            values.append(exit_reason)
        if peak is not None:
            fields.append("peak=?")
            values.append(peak)
        if note is not None:
            fields.append("note=?")
            values.append(note)
        if not fields:
            return
        values.append(signal_id)
        sql = f"UPDATE signals SET {', '.join(fields)} WHERE id=?"
        c.execute(sql, values)
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
    finally:
        conn.close()

def load_recent_to_deque(limit=200):
    rows = fetch_recent_signals(limit, as_dict=True)
    valid_records = []
    for record in rows:
        if (record.get('record_time') and
            re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', record['record_time']) and
            record.get('signal_time') and
            re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', record['signal_time'])):
            valid_records.append(record)
    return deque(valid_records, maxlen=limit)

def auto_clean_invalid_signals():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("""
        DELETE FROM signals 
        WHERE record_time NOT LIKE '____-__-__ __:__:__'
           OR signal_time NOT LIKE '____-__-__ __:__'
        """)
        deleted = c.rowcount
        if deleted > 0:
            log_event(f"自动清理了 {deleted} 条时间格式异常的信号记录")
        conn.commit()
    finally:
        conn.close()

def delete_old_signals(days=30):
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM signals WHERE record_time < ?", (cutoff,))
        deleted = c.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()

def clear_all_signals():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM signals")
        deleted = c.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()

# ---------- 日志与计数器 ----------
def log_event(msg):
    today = datetime.now().strftime("%Y%m%d")
    filename = f"strategy_log_{today}.txt"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass
    cleanup_old_logs()

def cleanup_old_logs(days=LOG_RETENTION_DAYS):
    cutoff = datetime.now() - timedelta(days=days)
    for log_file in glob.glob("strategy_log_*.txt"):
        try:
            date_str = log_file.replace("strategy_log_", "").replace(".txt", "")
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < cutoff:
                os.remove(log_file)
        except:
            pass

if 'api_fail_count' not in st.session_state:
    st.session_state.api_fail_count = 0

def reset_fail_count():
    st.session_state.api_fail_count = 0

# ---------- 统一重试 ----------
def fetch_with_retry(func, retries=RETRIES, base_delay=BASE_DELAY):
    for i in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            return func()
        except requests.exceptions.RequestException as e:
            wait = base_delay * (2 ** i)
            if i < retries - 1:
                msg = f"请求失败，{wait:.1f}秒后重试 ({i+1}/{retries})：{e}"
                st.warning(msg)
                log_event(msg)
                time.sleep(wait)
            else:
                msg = f"请求失败，已达最大重试次数：{e}"
                st.error(msg)
                log_event(msg)
                st.session_state.api_fail_count += 1
                if st.session_state.api_fail_count > 20:
                    st.error("API 持续失败超过20次，请检查网络或限速策略")
                return None
        except Exception as e:
            wait = base_delay * (2 ** i)
            if i < retries - 1:
                msg = f"未知错误，{wait:.1f}秒后重试 ({i+1}/{retries})：{e}"
                st.warning(msg)
                log_event(msg)
                time.sleep(wait)
            else:
                msg = f"未知错误，已达最大重试次数：{e}"
                st.error(msg)
                log_event(msg)
                st.session_state.api_fail_count += 1
                return None
    return None

# ---------- 数据获取（KuCoin）----------
@st.cache_data(ttl=5, show_spinner=False)
def fetch_klines():
    def _fetch():
        url = f"https://api.kucoin.com/api/v1/market/candles?type={INTERVAL}&symbol={SYMBOL}&limit={LIMIT}"
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, timeout=10, headers=headers)
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait = int(retry_after)
            st.warning(f"服务器要求等待 {wait} 秒")
            time.sleep(wait)
        resp.raise_for_status()
        return resp.json()
    result = fetch_with_retry(_fetch)
    if result:
        reset_fail_count()
    if result and result.get('code') == '200000' and result.get('data'):
        data = result['data']
        candles = []
        for item in data:
            # KuCoin 返回: [time, open, close, high, low, volume, turnover]
            # time 是秒级时间戳
            ts = int(item[0]) * 1000
            open_price = float(item[1])
            close = float(item[2])
            high = float(item[3])
            low = float(item[4])
            volume = float(item[5])
            candles.append([ts, open_price, high, low, close, volume])
        candles.sort(key=lambda x: x[0])
        return candles
    st.warning("KuCoin 返回数据异常")
    return []

@st.cache_data(ttl=5, show_spinner=False)
def fetch_latest_candle():
    def _fetch():
        url = f"https://api.kucoin.com/api/v1/market/candles?type={INTERVAL}&symbol={SYMBOL}&limit=1"
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, timeout=5, headers=headers)
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait = int(retry_after)
            st.warning(f"服务器要求等待 {wait} 秒")
            time.sleep(wait)
        resp.raise_for_status()
        return resp.json()
    result = fetch_with_retry(_fetch)
    if result:
        reset_fail_count()
    if result and result.get('code') == '200000' and result.get('data'):
        data = result['data']
        if data:
            item = data[0]
            ts = int(item[0]) * 1000
            return [ts, float(item[1]), float(item[3]), float(item[4]), float(item[2]), float(item[5])]
        st.warning("最新 K 线数据为空")
        return None
    return None

@st.cache_data(ttl=60, show_spinner=False)
def get_higher_trend():
    def _fetch():
        url = f"https://api.kucoin.com/api/v1/market/candles?type=1hour&symbol={SYMBOL}&limit=200"
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, timeout=5, headers=headers)
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait = int(retry_after)
            st.warning(f"服务器要求等待 {wait} 秒")
            time.sleep(wait)
        resp.raise_for_status()
        return resp.json()
    result = fetch_with_retry(_fetch)
    if result:
        reset_fail_count()
    if result and result.get('code') == '200000' and result.get('data'):
        data = result['data']
        if not data:
            return 'neutral'
        # 注意 KuCoin 返回顺序：时间、开盘、收盘、最高、最低、成交量，取收盘价
        closes = [float(item[2]) for item in data]
        if len(closes) < 200:
            return 'neutral'
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
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def volume_surge(df, vol_period=20, surge_mult=1.5):
    if len(df) < vol_period + 1:
        return True
    avg_vol = df['volume'].rolling(window=vol_period).mean().iloc[-1]
    return df['volume'].iloc[-1] > avg_vol * surge_mult

# ========== 核心信号检测 ==========
def detect_signal_pro(df, fast, slow, rsi_period, buy_min, buy_max, sell_min, sell_max,
                      higher_trend, use_volume_filter, use_slope_filter, use_atr_filter, atr_threshold=0.001):
    if df.empty:
        return None
    if len(df) < slow + rsi_period + 10:
        return None
    if pd.isna(df['ema_fast'].iloc[-1]) or pd.isna(df['ema_slow'].iloc[-1]):
        return None

    last = df.iloc[-1]
    ema_f_now = df['ema_fast'].iloc[-1]
    ema_s_now = df['ema_slow'].iloc[-1]
    rsi_now = df['rsi'].iloc[-1]
    atr_now = df['atr'].iloc[-1]

    is_bullish = (ema_f_now > ema_s_now) and (last['close'] > ema_f_now * 0.999)
    is_bearish = (ema_f_now < ema_s_now) and (last['close'] < ema_f_now * 1.001)

    slope_fast = (ema_f_now - df['ema_fast'].iloc[-4]) / 3 if len(df) >= 4 else 0

    if pd.isna(atr_now):
        atr_ok = True
    else:
        atr_ok = atr_now > last['close'] * atr_threshold if use_atr_filter else True

    vol_ok = volume_surge(df) if use_volume_filter else True

    if is_bullish and buy_min < rsi_now < buy_max:
        if use_slope_filter and slope_fast <= 0:
            return None
        if higher_trend == 'down':
            return None
        if not atr_ok or not vol_ok:
            return None
        return ('BUY', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)

    if is_bearish and sell_min < rsi_now < sell_max:
        if use_slope_filter and slope_fast >= 0:
            return None
        if higher_trend == 'up':
            return None
        if not atr_ok or not vol_ok:
            return None
        return ('SELL', last['close'], ema_f_now, ema_s_now, rsi_now, atr_now)
    return None

# ========== ATR动态止损/止盈 ==========
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

# ========== 移动止损 ==========
def check_trailing_stop(record, current_price, trailing_dist):
    if record is None or not isinstance(record, dict):
        return False, None, None, None
    if record.get('result') != 'pending':
        return False, None, None, None
    if 'peak' not in record or 'price' not in record or 'side' not in record:
        return False, None, None, None
    peak = record['peak']
    if peak is None or peak <= 0:
        return False, None, None, None
    side = record['side']
    if side not in ('BUY', 'SELL'):
        return False, None, None, None

    if side == 'BUY':
        trailing_sl = peak * (1 - trailing_dist / 100)
        if current_price <= trailing_sl:
            return True, 'exit', trailing_sl, '移动止损触发'
    else:
        trailing_sl = peak * (1 + trailing_dist / 100)
        if current_price >= trailing_sl:
            return True, 'exit', trailing_sl, '移动止损触发'
    return False, None, None, None

# ---------- 缺失K线补全 ----------
def fill_missing_candles(buffer, new_candle):
    if len(buffer) == 0:
        return [new_candle]
    last_ts = buffer[-1][0]
    new_ts = new_candle[0]
    expected_ts = last_ts + INTERVAL_MINUTES * 60 * 1000
    if new_ts > expected_ts:
        missing_candles = []
        ts = expected_ts
        while ts < new_ts:
            prev_close = buffer[-1][4]
            missing_candle = [ts, prev_close, prev_close, prev_close, prev_close, 0]
            missing_candles.append(missing_candle)
            ts += INTERVAL_MINUTES * 60 * 1000
        return missing_candles + [new_candle]
    else:
        return [new_candle]

# ---------- 信号历史管理 ----------
def update_signal_stats():
    history = st.session_state.signal_history
    total = len(history)
    win = sum(1 for s in history if s.get('result') == 'win')
    loss = sum(1 for s in history if s.get('result') == 'loss')
    exit_count = sum(1 for s in history if s.get('result') == 'exit')
    pending = sum(1 for s in history if s.get('result') == 'pending')
    win_rate = round(win / (win + loss) * 100, 1) if (win + loss) > 0 else 0
    st.session_state.signal_stats = {
        'total': total,
        'win': win,
        'loss': loss,
        'exit': exit_count,
        'pending': pending,
        'win_rate': win_rate
    }

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
    signal_id = save_signal_to_db(record)
    record['id'] = signal_id
    st.session_state.signal_history.appendleft(record)
    update_signal_stats()

def update_signal_result(index, result, exit_price=None, exit_reason='', note=''):
    history = st.session_state.signal_history
    if 0 <= index < len(history):
        record = history[index]
        if 'id' not in record:
            st.warning("记录缺少id，无法同步数据库")
            return
        record['result'] = result
        if exit_price is not None and exit_price > 0:
            record['exit_price'] = round(exit_price, 2)
            record['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if exit_reason:
            record['exit_reason'] = exit_reason
        if note:
            record['note'] = note
        update_signal_in_db(
            signal_id=record['id'],
            result=result,
            exit_price=record.get('exit_price'),
            exit_time=record.get('exit_time'),
            exit_reason=exit_reason,
            peak=record.get('peak'),
            note=record.get('note')
        )
        history[index] = record
        update_signal_stats()

def clear_signal_history(clear_db=False):
    st.session_state.signal_history = deque(maxlen=200)
    if clear_db:
        clear_all_signals()
    update_signal_stats()

# ---------- Streamlit 页面配置 ----------
st.set_page_config(page_title="ETH 5分钟策略 (职业终极版)", layout="wide")

# ----- 全局CSS优化 -----
st.markdown("""
<style>
    * {
        font-family: 'Inter', 'Microsoft YaHei', sans-serif;
    }
    .stApp {
        background-color: #0f172a;
    }
    /* 卡片悬停效果 */
    .finished-signal-card {
        transition: transform 0.2s;
        margin-bottom: 10px;
    }
    .finished-signal-card:hover {
        transform: scale(1.02);
    }
    /* 表格文字居中 */
    .dataframe th, .dataframe td {
        text-align: center !important;
    }
    /* 侧边栏滑块标签加粗 */
    .stSlider label {
        font-weight: 600;
    }
    /* metric数值放大 */
    .metric-value {
        font-size: 2rem !important;
    }
    /* 侧边栏统计彩色文字 */
    .stat-win {
        color: #00ff9d;
        font-weight: bold;
    }
    .stat-loss {
        color: #ff4d4d;
        font-weight: bold;
    }
    .stat-exit {
        color: #ffd966;
        font-weight: bold;
    }
    /* 信号卡片样式 */
    .signal-card {
        background: linear-gradient(135deg, #0a3d2a 0%, #112233 100%);
        border-radius: 18px;
        padding: 24px;
        margin: 15px 0 25px 0;
        border: 3px solid #00ff9d;
        box-shadow: 0 10px 30px rgba(0, 255, 157, 0.2);
        position: relative;
        overflow: hidden;
    }
    .header-green, .header-red {
        padding: 16px 28px;
        border-radius: 12px;
        font-size: 27px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 22px;
        letter-spacing: 1px;
    }
    .header-green { background: #0a3d2a; color: #00ff9d; border-left: 8px solid #00ff9d; box-shadow: 0 0 15px #00ff9d; }
    .header-red { background: #3d0a0a; color: #ff4d4d; border-left: 8px solid #ff4d4d; box-shadow: 0 0 15px #ff4d4d; }
    .price-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 20px;
    }
    .price-item, .atr-box {
        background: rgba(255,255,255,0.06);
        padding: 16px 12px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.12);
        transition: all 0.3s ease;
    }
    .price-item:hover, .atr-box:hover { transform: scale(1.04); box-shadow: 0 0 20px rgba(255,255,255,0.15); }
    .price-label { font-size: 15px; opacity: 0.85; margin-bottom: 6px; }
    .price-value { font-size: 29px; font-weight: 700; line-height: 1.05; }
    .risk-text { font-size: 13px; color: #ff99cc; margin-top: 4px; }
    .atr-box { border-color: #4a90ff; }
    .atr-box .price-value { color: #4a90ff; }
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .metric-item {
        background: rgba(30,58,95,0.65);
        padding: 14px 10px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(74,144,255,0.3);
    }
    .metric-item div:first-child { font-size: 13.5px; opacity: 0.8; margin-bottom: 3px; }
    .metric-value { font-size: 23px; font-weight: 600; }
    .copy-btn {
        background: linear-gradient(90deg, #00ff9d, #00cc7a) !important;
        color: #000 !important;
        font-weight: 700;
        font-size: 17px;
        padding: 14px 32px;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .copy-btn:hover { transform: scale(1.05); box-shadow: 0 0 25px #00ff9d; }
    .op-guide {
        background: #1e3a5f;
        padding: 22px;
        border-radius: 14px;
        color: #a0d8ff;
        border-left: 6px solid #4a90ff;
        box-shadow: 0 4px 15px rgba(74,144,255,0.15);
    }
    .waiting-card {
        background: #1e3a5f;
        padding: 32px;
        border-radius: 18px;
        text-align: center;
        font-size: 23px;
        color: #89c2ff;
        border: 2px dashed #4a90ff;
        box-shadow: 0 8px 25px rgba(74,144,255,0.1);
        margin: 15px 0 25px 0;
    }
    .tp2-container {
        background: linear-gradient(135deg, #2c2200 0%, #4a3a00 50%, #2c2200 100%);
        border: 3px solid #ffd700;
        border-radius: 18px;
        padding: 22px 18px;
        margin: 22px 0 26px 0;
        box-shadow: 0 0 25px rgba(255,215,0,0.7), 0 0 45px rgba(255,170,0,0.4), inset 0 0 25px rgba(255,215,0,0.25);
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: tp2-pulse 2.5s infinite ease-in-out;
    }
    .tp2-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 60%; height: 300%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
        animation: tp2-shine 4s infinite linear;
        pointer-events: none;
    }
    @keyframes tp2-pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.015); } }
    @keyframes tp2-shine { 0% { transform: translateX(-150%) skewX(-15deg); } 100% { transform: translateX(400%) skewX(-15deg); } }
    .finished-signal-card {
        background: linear-gradient(135deg, #1a2a3a 0%, #0f1a24 100%);
        border-radius: 12px;
        padding: 15px;
        border-left: 6px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .finished-signal-card.win { border-left-color: #90ee90; }
    .finished-signal-card.loss { border-left-color: #ffcccb; }
    .finished-signal-card.exit { border-left-color: #ffd966; }
</style>
""", unsafe_allow_html=True)

st.title("📈 ETH 5分钟 EMA 剥头皮策略 (职业整合版)")

init_db()
auto_clean_invalid_signals()

if 'signal_history' not in st.session_state:
    st.session_state.signal_history = load_recent_to_deque(200)

if 'candle_buffer' not in st.session_state:
    st.session_state.candle_buffer = deque(maxlen=500)
if 'signal_stats' not in st.session_state:
    st.session_state.signal_stats = {'total': 0, 'win': 0, 'loss': 0, 'exit': 0, 'pending': 0, 'win_rate': 0}
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = None

candle_buffer = st.session_state.candle_buffer
signal_history = st.session_state.signal_history

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("策略参数")
    fast_ema = st.number_input("快线 EMA", 1, 50, 9, 1)
    slow_ema = st.number_input("慢线 EMA", 2, 100, 21, 1)
    rsi_period = st.number_input("RSI 周期", 2, 50, 14, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        buy_min = st.number_input("多头下限", 0, 100, 50, 1)
    with col2:
        buy_max = st.number_input("多头上限", 0, 100, 70, 1)
    
    col3, col4 = st.columns(2)
    with col3:
        sell_min = st.number_input("空头下限", 0, 100, 30, 1)
    with col4:
        sell_max = st.number_input("空头上限", 0, 100, 50, 1)
    
    refresh_interval = st.number_input("刷新间隔(秒)", 5, 300, 60, 5)

    with st.expander("✨ 高级过滤", expanded=True):
        use_slope_filter = st.checkbox("启用斜率过滤", value=True)
        use_volume_filter = st.checkbox("启用成交量爆发过滤", value=True)
        use_atr_filter = st.checkbox("启用波动率过滤", value=True)
        atr_threshold = st.slider(
            "ATR 阈值 (相对于价格的百分比)",
            min_value=0.05, max_value=0.5, value=0.1, step=0.05,
            help="ATR 必须大于价格 × 此百分比。若信号太少，可适当降低此值（例如 0.05%）。"
        ) / 100
        use_higher_tf_filter = st.checkbox("启用高周期趋势过滤", value=True)

    with st.expander("📊 动态止损", expanded=True):
        use_atr_sl = st.checkbox("启用ATR动态止损", value=True)
        if use_atr_sl:
            atr_mult_sl = st.slider("ATR止损倍数", 0.5, 4.0, 2.55, 0.05)
            atr_mult_tp1 = st.slider("ATR TP1倍数 (RR 1:1)", 0.5, 3.0, 1.0, 0.05)
            atr_mult_tp2 = st.slider("ATR TP2倍数 (RR 2:1)", 1.0, 5.0, 2.0, 0.05)
        else:
            atr_mult_sl = atr_mult_tp1 = atr_mult_tp2 = 1.2

    with st.expander("✨ 移动止损", expanded=False):
        use_trailing = st.checkbox("启用移动止损", value=False)
        trailing_distance = st.slider("回调距离 (%)", 0.1, 2.0, 0.3, 0.1) if use_trailing else 0.3

    sound_enabled = st.checkbox("🔊 启用信号声音提醒", value=True)

    st.markdown("---")
    st.metric("📡 API 失败次数", st.session_state.api_fail_count)

    if st.button("立即刷新数据", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    if st.button("🔄 重置所有状态", width='stretch'):
        for key in ['candle_buffer', 'signal_history', 'signal_stats', 'last_signal_time']:
            try:
                if key in st.session_state:
                    del st.session_state[key]
            except Exception as e:
                st.warning(f"清除状态 {key} 时发生错误: {e}")
        st.cache_data.clear()
        st.rerun()

    if st.checkbox("🧪 DB 调试"):
        rows = fetch_recent_signals(10, as_dict=True)
        st.write("最近10条数据库记录：")
        for row in rows:
            st.write(row)

    clear_db_option = st.checkbox("同时清空数据库", value=False, key="clear_db_option")
    if st.button("🗑 清空历史信号", width='stretch'):
        clear_signal_history(clear_db=clear_db_option)
        st.rerun()

# ---------- 获取K线数据 ----------
candles = fetch_klines()
if isinstance(candles, list) and len(candles) > 0:
    if len(candle_buffer) == 0:
        for c in candles:
            candle_buffer.append(c)
    else:
        for c in candles:
            if c[0] > candle_buffer[-1][0]:
                missing = fill_missing_candles(candle_buffer, c)
                for mc in missing:
                    candle_buffer.append(mc)
else:
    st.warning("无法获取最新K线，使用缓存数据。")

latest = fetch_latest_candle()
if latest and isinstance(latest, list) and len(latest) == 6:
    if len(candle_buffer) == 0 or latest[0] > candle_buffer[-1][0]:
        missing = fill_missing_candles(candle_buffer, latest)
        for mc in missing:
            candle_buffer.append(mc)

st_autorefresh(interval=refresh_interval * 1000, key="final")

higher_trend = get_higher_trend() if use_higher_tf_filter else 'neutral'
trend_icon = "🟢" if higher_trend == "up" else "🔴" if higher_trend == "down" else "⚪"
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | K线数量: {len(candle_buffer)} | 高周期趋势: {trend_icon} {higher_trend.upper()}")

if len(candle_buffer) < 30:
    st.warning(f"⏳ 数据积累中... 当前 {len(candle_buffer)}/30 根")
else:
    df = pd.DataFrame(list(candle_buffer), columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty:
        st.warning("DataFrame 为空，无法计算指标")
        st.stop()
    df['time'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('time', inplace=True)
    df['ema_fast'] = calculate_ema(df['close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['close'], slow_ema)
    df['rsi'] = calculate_rsi_wilder(df['close'], rsi_period)
    df['atr'] = calculate_atr(df, 14)

    signal = detect_signal_pro(df, fast_ema, slow_ema, rsi_period, buy_min, buy_max, sell_min, sell_max,
                               higher_trend, use_volume_filter, use_slope_filter, use_atr_filter, atr_threshold)

    # 显示信号卡片（新信号或待定）
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

    # 更新峰值
    if signal_history and len(signal_history) > 0 and signal_history[0]['result'] == 'pending':
        rec = signal_history[0]
        cp = df['close'].iloc[-1]
        updated = False
        if rec['side'] == 'BUY' and cp > rec['peak']:
            rec['peak'] = cp
            updated = True
        elif rec['side'] == 'SELL' and cp < rec['peak']:
            rec['peak'] = cp
            updated = True
        if updated:
            update_signal_in_db(signal_id=rec['id'], peak=rec['peak'], note="峰值更新")

    cp = df['close'].iloc[-1]
    # 检查止损/止盈
    for i, r in enumerate(signal_history):
        if r['result'] != 'pending':
            continue
        s = r['side']
        sl = r['sl']
        tp2 = r['tp2']
        if s == 'BUY':
            if cp <= sl:
                update_signal_result(i, 'loss', cp, '止损触发')
            elif cp >= tp2:
                update_signal_result(i, 'win', cp, 'TP2触发')
        else:
            if cp >= sl:
                update_signal_result(i, 'loss', cp, '止损触发')
            elif cp <= tp2:
                update_signal_result(i, 'win', cp, 'TP2触发')
        if use_trailing:
            exit_flag, res, ep, reason = check_trailing_stop(r, cp, trailing_distance)
            if exit_flag:
                update_signal_result(i, res, ep, reason)

    # ========== 信号卡片区域（在图表上方）==========
    if show_signal:
        side, price, ema_f, ema_s, rsi, atr_val = show_signal
        sl, tp1, tp2 = calculate_sltp(price, side, atr_val, use_atr_sl, atr_mult_sl, atr_mult_tp1, atr_mult_tp2)
        signal_time = df.index[-1].strftime('%Y-%m-%d %H:%M') if signal else (signal_history[0]['signal_time'] if signal_history else '')

        risk_pts = abs(price - sl)
        risk_pct = (risk_pts / price * 100)
        profit_pts = abs(tp2 - price)
        profit_pct = (profit_pts / price * 100)

        st.markdown('<div class="signal-card">', unsafe_allow_html=True)

        if side == 'BUY':
            st.markdown(f'<div class="header-green">● 多头信号 @ {signal_time}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="header-red">● 空头信号 @ {signal_time}</div>', unsafe_allow_html=True)

        atr_display = f"{atr_val:.2f}" if atr_val is not None else "N/A"

        st.markdown(f'''
            <div class="price-grid">
                <div class="price-item"><div class="price-label" style="color:#ff4d4d;">★ 进场价格</div><div class="price-value" style="color:#ffffff;">{price:.2f}</div></div>
                <div class="price-item"><div class="price-label" style="color:#ff99cc;">● 止损价格</div><div class="price-value" style="color:#ff99cc;">{sl:.2f}</div><div class="risk-text">风险 {risk_pts:.2f}点 ({risk_pct:.2f}%)</div></div>
                <div class="price-item"><div class="price-label" style="color:#ff99cc;">● 第一目标 (TP1)</div><div class="price-value" style="color:#ff99cc;">{tp1:.2f}</div><div style="font-size:13px;color:#ff99cc;">RR 1:1</div></div>
                <div class="atr-box"><div style="font-size:15px;color:#4a90ff;">ATR</div><div class="price-value" style="margin-top:2px;">{atr_display}</div></div>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="tp2-container">
            <div style="display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:6px;">
                <span style="font-size:26px;color:#ffd700;">✨</span>
                <span style="font-size:19px;color:#ffd700;font-weight:600;letter-spacing:1px;">第二目标 (TP2) · 全平仓</span>
                <span style="font-size:26px;color:#ffd700;">✨</span>
            </div>
            <div style="font-size:48px;font-weight:900;line-height:1;background:linear-gradient(90deg,#ffe066,#ffd700,#ffeb3b,#ffd700,#ffe066);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-shadow:0 0 20px #ffd700,0 0 40px #ffaa00,0 0 60px rgba(255,215,0,0.6);">{tp2:.2f}</div>
            <div style="margin-top:8px;display:flex;justify-content:center;gap:20px;font-size:15px;">
                <span style="background:rgba(255,215,0,0.15);color:#ffd700;padding:4px 14px;border-radius:20px;border:1px solid #ffd700;">RR <strong>2:1</strong></span>
                <span style="color:#a0ff9d;font-weight:600;">潜在盈利 +{profit_pts:.2f}点 (+{profit_pct:.1f}%)</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
            <div class="metrics-row">
                <div class="metric-item"><div>EMA快线</div><div class="metric-value" style="color:#ffa500;">{ema_f:.2f}</div></div>
                <div class="metric-item"><div>EMA慢线</div><div class="metric-value" style="color:#4a90ff;">{ema_s:.2f}</div></div>
                <div class="metric-item"><div>RSI</div><div class="metric-value" style="color:#00ff9d;">{rsi:.1f}</div></div>
                <div class="metric-item"><div>当前价格</div><div class="metric-value" style="color:#ffffff;">{cp:.2f}</div></div>
            </div>
        ''', unsafe_allow_html=True)

        signal_text = f"🟢 多头信号 @ {signal_time} 进场{price:.2f} SL{sl:.2f} TP1{tp1:.2f} TP2{tp2:.2f} 风险{risk_pts:.2f}点" if side == 'BUY' else f"🔴 空头信号 @ {signal_time} 进场{price:.2f} SL{sl:.2f} TP1{tp1:.2f} TP2{tp2:.2f} 风险{risk_pts:.2f}点"
        if st.button("📋 一键复制交易信号", key="copy_btn", width='stretch'):
            st.markdown(f'<script>navigator.clipboard.writeText(`{signal_text}`);</script>', unsafe_allow_html=True)
            st.success("✅ 已复制到剪贴板！直接粘贴到OKX即可下单")

        if use_trailing and signal_history and signal_history[0]['result'] == 'pending':
            latest = signal_history[0]
            peak = latest['peak']
            if side == 'BUY':
                trailing_sl = max(sl, peak * (1 - trailing_distance / 100))
                st.success(f"📈 当前最高 **{peak:.2f}** → 建议移动止损 **{trailing_sl:.2f}**（已上移）")
            else:
                trailing_sl = min(sl, peak * (1 + trailing_distance / 100))
                st.error(f"📉 当前最低 **{peak:.2f}** → 建议移动止损 **{trailing_sl:.2f}**（已下移）")

        st.markdown(f"""
        <div class="op-guide">
            <strong>📌 操作指引：</strong>
            <ul style="margin:12px 0 0 22px;padding:0;line-height:1.7;">
                <li>进场：在 <strong>{price:.2f}</strong> 附近买入/卖出。</li>
                <li>止损：立即挂单 <strong>{sl:.2f}</strong>。</li>
                <li>止盈：TP1 <strong>{tp1:.2f}</strong>（平半仓），TP2 <strong>{tp2:.2f}</strong>（全平）。</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="waiting-card">⏳ 等待新信号出现...<br><span style="font-size:16px;opacity:0.7;">系统正在实时扫描 5分钟K线</span></div>', unsafe_allow_html=True)

    # ========== 主内容区三列布局 ==========
    col1, col2 = st.columns([3, 1])

    with col1:
        # 图表区域
        st.markdown("""
        <style>
        .chart-container {
            background: #0e1621;
            border-radius: 16px;
            padding: 12px;
            border: 2px solid rgba(74,144,255,0.2);
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        plot_df = df.tail(200)
        colors = np.where(plot_df['close'] >= plot_df['open'], '#00ff9d', '#ff4d4d')
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28],
                            subplot_titles=("<b>价格 & EMA</b>", "<b>成交量</b>"))

        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'],
                                     name='K线', increasing_line_color='#00ff9d', decreasing_line_color='#ff4d4d',
                                     line=dict(width=1.8), increasing_fillcolor='#00ff9d', decreasing_fillcolor='#ff4d4d', whiskerwidth=0.6), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_fast'], name=f'EMA快线 ({fast_ema})',
                                 line=dict(color='#ffd700', width=3.5), hovertemplate='EMA快线: %{y:.2f}<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_slow'], name=f'EMA慢线 ({slow_ema})',
                                 line=dict(color='#4da9ff', width=3.5), hovertemplate='EMA慢线: %{y:.2f}<extra></extra>'), row=1, col=1)
        # 成交量trace，隐藏图例
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name='成交量', marker_color=colors, opacity=0.85,
                             hovertemplate='成交量: %{y:,.0f}<extra></extra>', showlegend=False), row=2, col=1)

        fig.add_hline(y=cp, line_dash="dash", line_color="#00ff9d", line_width=1.5, row=1, col=1,
                      annotation_text=f"当前价 {cp:.2f}", annotation_position="top right",
                      annotation_font=dict(size=13, color="#00ff9d"))

        fig.update_layout(height=680, template="plotly_dark", plot_bgcolor="#0e1621", paper_bgcolor="#0e1621",
                          font=dict(family="Microsoft YaHei, Arial, sans-serif", size=13, color="#e0e0e0"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                      bgcolor="rgba(15,23,42,0.85)", bordercolor="#334155", borderwidth=1,
                                      font=dict(size=12.5, color="#e0e0e0")),
                          margin=dict(l=10, r=30, t=50, b=30), hovermode="x unified",
                          hoverlabel=dict(bgcolor="#1e2937", font_size=13, font_family="Microsoft YaHei"))

        fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, showline=True, linewidth=1, linecolor="#334155")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, showline=True, linewidth=1, linecolor="#334155")

        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # 右侧信息区域
        st.metric("当前价", f"{cp:.2f}")

        # 计算24h涨跌
        if len(df) > 1:
            now = df.index[-1]
            target_time = now - timedelta(hours=24)
            time_diffs = np.abs((df.index - target_time).total_seconds())
            closest_idx = time_diffs.argmin()
            if time_diffs[closest_idx] <= 5 * 60:
                prev_close = df.iloc[closest_idx]['close']
                change = cp - prev_close
                change_pct = (change / prev_close) * 100
                color = "#00ff9d" if change >= 0 else "#ff4d4d"
                arrow = "▲" if change >= 0 else "▼"
                st.markdown(f"24h涨跌: <span style='color:{color}'>{arrow} {change:+.2f} ({change_pct:+.2f}%)</span>", unsafe_allow_html=True)
            else:
                st.markdown("24h涨跌: --")
        else:
            st.markdown("24h涨跌: --")

        st.markdown("---")

        stats = st.session_state.signal_stats
        st.markdown(f"**总信号:** {stats['total']}")
        st.markdown(f"<span class='stat-win'>✅ 盈利: {stats['win']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='stat-loss'>❌ 亏损: {stats['loss']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='stat-exit'>⏹️ 移动退出: {stats['exit']}</span>", unsafe_allow_html=True)
        st.markdown(f"**胜率:** {stats['win_rate']}%")
        st.markdown("---")

        st.subheader("📋 最近信号")
        if signal_history:
            finished = [s for s in list(signal_history) if s.get('result') in ('win', 'loss', 'exit')][:3]
            if finished:
                for s in finished:
                    if s['exit_price'] and s['exit_price'] > 0:
                        if s['side'] == 'BUY':
                            points = s['exit_price'] - s['price']
                        else:
                            points = s['price'] - s['exit_price']
                    else:
                        points = 0.0
                    card_class = f"finished-signal-card {s['result']}"
                    st.markdown(f"""
                    <div class="{card_class}" style="background: linear-gradient(135deg, #1a2a3a 0%, #0f1a24 100%); border-radius: 12px; padding: 15px; border-left: 6px solid {'#90ee90' if s['result']=='win' else '#ffcccb' if s['result']=='loss' else '#ffd966'};">
                        <div style="font-size:18px; font-weight:bold;">{"🟢 多头" if s['side']=='BUY' else "🔴 空头"}</div>
                        <div style="font-size:12px; opacity:0.8;">{s['signal_time']}</div>
                        <div>盈亏: <span style="font-weight:bold; {'color:#90ee90' if points>0 else '#ff4d4d' if points<0 else '#ffffff'}">{points:+.2f}点</span></div>
                        <div style="font-size:12px; opacity:0.7;">{s['exit_reason']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("暂无已完成信号")
        else:
            st.info("暂无信号记录")

    # ---------- 下方表格 ----------
    st.markdown("---")
    st.subheader("最近 10 根K线")
    display_df = df.reset_index()[['time', 'open', 'high', 'low', 'close', 'volume', 'ema_fast', 'ema_slow', 'rsi', 'atr']].tail(10)
    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df = display_df.rename(columns={
        'time': '时间',
        'open': '开盘',
        'high': '最高',
        'low': '最低',
        'close': '收盘',
        'volume': '成交量',
        'ema_fast': 'EMA快线',
        'ema_slow': 'EMA慢线',
        'rsi': 'RSI',
        'atr': 'ATR'
    })
    st.dataframe(display_df.round(2), width='stretch', hide_index=True)

    # ---------- 历史信号记录 ----------
    st.markdown("---")
    st.subheader("📜 历史信号记录")
    if signal_history:
        hist_data = []
        for s in list(signal_history)[:50]:
            if s['result'] == 'win':
                result_emoji = "✅ 赢"
            elif s['result'] == 'loss':
                result_emoji = "❌ 亏"
            elif s['result'] == 'exit':
                result_emoji = "⏹️ 退"
            else:
                result_emoji = "⏳ 待"
            hist_data.append({
                "记录时间": s['record_time'],
                "信号时间": s['signal_time'],
                "方向": "🟢 多头" if s['side'] == 'BUY' else "🔴 空头",
                "进场价": f"{s['price']:.2f}",
                "止损": f"{s['sl']:.2f}",
                "TP1": f"{s['tp1']:.2f}",
                "TP2": f"{s['tp2']:.2f}",
                "结果": result_emoji,
                "出场价": f"{s['exit_price']:.2f}" if s['exit_price'] else "—",
                "出场原因": s['exit_reason'] if s['exit_reason'] else "—",
                "峰值": f"{s['peak']:.2f}" if s['peak'] else "—"
            })
        hist_df = pd.DataFrame(hist_data)

        def color_result(val):
            if '✅' in val:
                return 'background-color: #90ee90; color: black'
            elif '❌' in val:
                return 'background-color: #ffcccb; color: black'
            elif '⏹️' in val:
                return 'background-color: #ffd966; color: black'
            elif '⏳' in val:
                return 'background-color: #fffacd; color: black'
            return ''

        styled = hist_df.style.map(color_result, subset=['结果'])
        styled.set_properties(**{'text-align': 'center'})
        styled.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        st.dataframe(styled, width='stretch', height=400)

        with st.expander("✏️ 手动标记信号结果", expanded=False):
            col1, col2, col3, col4 = st.columns([3,1,1,2])
            with col1:
                options = [f"{i}. {s['record_time']} {s['side']} @{s['price']:.2f}" for i, s in enumerate(list(signal_history)[:50])]
                idx = st.selectbox("选择信号", range(len(options)), format_func=lambda x: options[x])
            with col2:
                res = st.selectbox("结果", ["pending", "win", "loss", "exit"])
            with col3:
                ep = st.number_input("出场价", value=0.0, step=0.01)
            with col4:
                if st.button("✅ 更新结果", width='stretch'):
                    rec = signal_history[idx]
                    update_signal_result(idx, res, ep if ep > 0 else None,
                                         exit_reason="手动标记",
                                         note="用户手动更新")
                    st.rerun()
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出历史信号 (CSV)", csv, f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    else:
        st.info("暂无历史信号，等待新信号出现...")

st.markdown("---")
st.caption("🔥 终极职业版 • 持续信号 + ATR动态止损 + 豪华卡片 + 专业图表 + SQLite持久化 + 缺失K线补全 + 峰值同步 + 自动数据清洗 + 卡片复盘 + 24h涨跌 • 祝交易大赚！💰")
