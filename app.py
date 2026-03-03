"""
5分钟趋势合约系统（生产级最终完美版V2）
- 依赖版本锁定：streamlit==1.41.0 等
- 省电模式开关（5秒/15秒刷新）
- 信号防抖：避免重复叠加标记
- 注记位置优化：不遮挡K线
- 异步数据线程 + 心跳 + 自动重启
- 回测模块：夏普比率、最大回撤
- 模拟账户 + 信号历史 + 实时UI
- 日志轮转 + 文件备份 + 防空
"""

import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import os
import json
import numpy as np
import threading
import logging
from logging.handlers import RotatingFileHandler
import shutil
from streamlit_autorefresh import st_autorefresh

# --------------------------
# 日志配置
# --------------------------
log_handler = RotatingFileHandler(
    'error.log', maxBytes=5 * 1024 * 1024, backupCount=3
)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(layout="wide")
st.title("📈 5分钟趋势合约系统（最终完美版V2）")

SYMBOL = "ETH-USDT-SWAP"
CACHE_FILE_5M = "market_data_5m_cache.csv"
CACHE_FILE_15M = "market_data_15m_cache.csv"
SIGNAL_HISTORY_FILE = "signal_history.json"
ACCOUNT_FILE = "account.json"
BACKUP_DIR = "backup"

# --------------------------
# 省电模式开关
# --------------------------
st.sidebar.header("⚙️ 系统设置")
power_saving = st.sidebar.checkbox("省电模式", value=False, help="开启后刷新间隔延长至15秒")
refresh_interval = 15000 if power_saving else 5000  # 毫秒
st_autorefresh(interval=refresh_interval, key="refresh")

# --------------------------
# 文件备份与截断
# --------------------------
def ensure_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

def trim_file(file_path, max_items=200):
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if len(data) > max_items:
            data = data[-max_items:]
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
    except Exception:
        logging.exception(f"截断失败: {file_path}")

def daily_backup():
    ensure_backup_dir()
    today = datetime.now().strftime("%Y%m%d")
    for f in [SIGNAL_HISTORY_FILE, ACCOUNT_FILE, CACHE_FILE_5M, CACHE_FILE_15M]:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(BACKUP_DIR, f"{today}_{f}"))

if "last_backup_day" not in st.session_state:
    st.session_state.last_backup_day = datetime.now().day

if datetime.now().day != st.session_state.last_backup_day:
    daily_backup()
    st.session_state.last_backup_day = datetime.now().day

trim_file(SIGNAL_HISTORY_FILE, 500)
trim_file(ACCOUNT_FILE, 100)

# --------------------------
# 异步数据获取线程（带心跳）
# --------------------------
class DataFetcher:
    def __init__(self):
        self.lock = threading.Lock()
        self._5m_data = None
        self._15m_data = None
        self._ticker = None
        self._stop = threading.Event()
        self.fail_count = 0
        self.last_heartbeat = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logging.info("数据线程启动")

    def _run(self):
        while not self._stop.is_set():
            try:
                start = time.time()

                data_5m = self._fetch_okx("5m", 300)
                if data_5m:
                    df = self._to_df(data_5m)
                    df.to_csv(CACHE_FILE_5M, index=False)
                    with self.lock:
                        self._5m_data = df

                data_15m = self._fetch_okx("15m", 100)
                if data_15m:
                    df = self._to_df(data_15m)
                    df.to_csv(CACHE_FILE_15M, index=False)
                    with self.lock:
                        self._15m_data = df

                ticker = self._fetch_ticker()
                if ticker is not None:
                    with self.lock:
                        self._ticker = ticker

                self.fail_count = 0
                self.last_heartbeat = time.time()

                if time.time() - start > 10:
                    logging.warning("数据获取超时")

            except Exception:
                self.fail_count += 1
                logging.exception("DataFetcher异常")
                if self.fail_count > 5:
                    logging.error("连续失败超过5次，停止线程")
                    self._stop.set()

            for _ in range(5):
                if self._stop.is_set():
                    break
                time.sleep(1)

    def _to_df(self, data):
        df = pd.DataFrame(data, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "volCcy", "volCcyQuote", "confirm"
        ])
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.sort_values("ts").reset_index(drop=True)

    def _fetch_okx(self, bar, limit):
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": SYMBOL, "bar": bar, "limit": limit}
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=5, params=params)
                if r.status_code == 200:
                    j = r.json()
                    return j.get("data")
            except:
                time.sleep(2 ** attempt)
        return None

    def _fetch_ticker(self):
        url = f"https://www.okx.com/api/v5/market/ticker?instId={SYMBOL}"
        try:
            r = requests.get(url, timeout=3)
            data = r.json()
            if data.get("code") == "0":
                return float(data["data"][0]["last"])
        except:
            return None

    def stop(self):
        self._stop.set()
        logging.info("数据线程停止")

    def is_alive(self):
        return self.thread.is_alive() and not self._stop.is_set()

    def get_5m(self):
        with self.lock:
            return self._5m_data.copy() if self._5m_data is not None else None

    def get_15m(self):
        with self.lock:
            return self._15m_data.copy() if self._15m_data is not None else None

    def get_ticker(self):
        with self.lock:
            return self._ticker


# 初始化或重启数据线程
def init_fetcher():
    if "fetcher" not in st.session_state or not st.session_state.fetcher.is_alive():
        st.session_state.fetcher = DataFetcher()

init_fetcher()
fetcher = st.session_state.fetcher

# 心跳冷却 + 自动重启
if "last_restart" not in st.session_state:
    st.session_state.last_restart = 0

if time.time() - fetcher.last_heartbeat > 30:
    if time.time() - st.session_state.last_restart > 60:
        st.sidebar.warning("数据线程无响应，正在自动重启...")
        fetcher.stop()
        time.sleep(1)
        st.session_state.fetcher = DataFetcher()
        st.session_state.last_restart = time.time()
        st.rerun()

# --------------------------
# 从fetcher获取数据
# --------------------------
df_5m = fetcher.get_5m()
if df_5m is None or df_5m.empty:
    if os.path.exists(CACHE_FILE_5M):
        df_5m = pd.read_csv(CACHE_FILE_5M, parse_dates=["ts"])
    else:
        st.error("无法获取5分钟数据")
        st.stop()

df_15m = fetcher.get_15m()
real_price = fetcher.get_ticker()

# --------------------------
# 侧边栏参数
# --------------------------
st.sidebar.header("⚙️ 策略参数")
adx_threshold = st.sidebar.slider("ADX趋势阈值", 20, 40, 22)
ema_period_fast = st.sidebar.slider("快线EMA周期", 10, 30, 12)
ema_period_slow = st.sidebar.slider("慢线EMA周期", 30, 100, 50)
atr_period = st.sidebar.slider("ATR周期", 7, 30, 14)
lookback_sr = st.sidebar.slider("支撑/阻力周期（根K线）", 10, 50, 15)
callback_atr_mult = st.sidebar.slider("回调距离（ATR倍数）", 0.3, 1.0, 0.7, step=0.1)
risk_reward = st.sidebar.slider("盈亏比", 1.0, 3.0, 2.0, step=0.1)
risk_percent = st.sidebar.slider("单笔风险 (%)", 1.0, 5.0, 2.0, step=0.5)
initial_balance = st.sidebar.number_input("初始资金 (USDT)", value=100.0, step=10.0)
use_tf_filter = st.sidebar.checkbox("启用15分钟EMA验证", value=True)

# 回测模式开关
backtest_mode = st.sidebar.checkbox("🔍 回测模式", value=False)
if backtest_mode:
    days_back = st.sidebar.slider("回测天数", 1, 30, 7)

# 线程状态显示
st.sidebar.divider()
st.sidebar.write("📡 **数据线程状态**")
st.sidebar.write(f"心跳: {datetime.fromtimestamp(fetcher.last_heartbeat).strftime('%H:%M:%S')}")
st.sidebar.write(f"失败计数: {fetcher.fail_count}")
if not fetcher.is_alive():
    st.sidebar.error("线程已停止")
if st.sidebar.button("🔄 手动重启线程"):
    fetcher.stop()
    time.sleep(1)
    st.session_state.fetcher = DataFetcher()
    st.rerun()

# 查看错误日志
if st.sidebar.button("📋 查看错误日志"):
    if os.path.exists("error.log"):
        with open("error.log", "r") as f:
            logs = f.readlines()[-10:]
            st.sidebar.code("".join(logs))
    else:
        st.sidebar.info("无错误日志")

trend_confirm_bars = 3

# --------------------------
# 指标计算与缓存（版本控制）
# --------------------------
if "last_calc_time" not in st.session_state:
    st.session_state.last_calc_time = 0
    st.session_state.last_params = None
    st.session_state.last_ts = None
    st.session_state.cached_df = None
    st.session_state.cached_15m = None

current_params = (
    adx_threshold, ema_period_fast, ema_period_slow, atr_period,
    lookback_sr, callback_atr_mult, risk_reward, risk_percent,
    use_tf_filter, trend_confirm_bars
)

latest_ts = df_5m["ts"].iloc[-1] if not df_5m.empty else None

need_recalc = False
now = time.time()
if st.session_state.last_params != current_params:
    need_recalc = True
elif latest_ts and st.session_state.last_ts != latest_ts:
    need_recalc = True
elif now - st.session_state.last_calc_time > 30:
    need_recalc = True

if need_recalc:
    df = df_5m.copy()
    df["EMA_fast"] = ta.trend.ema_indicator(df["close"], window=ema_period_fast)
    df["EMA_slow"] = ta.trend.ema_indicator(df["close"], window=ema_period_slow)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)
    df["resistance_roll"] = df["high"].rolling(window=lookback_sr).max()
    df["support_roll"] = df["low"].rolling(window=lookback_sr).min()
    df["EMA_fast_up"] = df["EMA_fast"] > df["EMA_fast"].shift(1)
    df["EMA_fast_down"] = df["EMA_fast"] < df["EMA_fast"].shift(1)
    df["trend_up_confirm"] = (df["EMA_fast"] > df["EMA_slow"]) & df["EMA_fast_up"]
    df["trend_down_confirm"] = (df["EMA_fast"] < df["EMA_slow"]) & df["EMA_fast_down"]
    df["trend_up_streak"] = df["trend_up_confirm"].rolling(trend_confirm_bars).sum() == trend_confirm_bars
    df["trend_down_streak"] = df["trend_down_confirm"].rolling(trend_confirm_bars).sum() == trend_confirm_bars
    df = df.dropna().reset_index(drop=True)
    st.session_state.cached_df = df

    if use_tf_filter and df_15m is not None and not df_15m.empty:
        df_15m_calc = df_15m.copy()
        df_15m_calc["EMA20"] = ta.trend.ema_indicator(df_15m_calc["close"], window=20)
        df_15m_calc = df_15m_calc.dropna().reset_index(drop=True)
        st.session_state.cached_15m = df_15m_calc
    else:
        st.session_state.cached_15m = None

    st.session_state.last_calc_time = now
    st.session_state.last_params = current_params
    st.session_state.last_ts = latest_ts

df = st.session_state.cached_df
if df is None or df.empty:
    st.error("指标计算失败")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# --------------------------
# 15分钟EMA方向判断
# --------------------------
tf_ok = True
if use_tf_filter and st.session_state.cached_15m is not None:
    df_15m_calc = st.session_state.cached_15m
    if len(df_15m_calc) > 1:
        latest_15m = df_15m_calc.iloc[-1]
        prev_15m = df_15m_calc.iloc[-2]
        if latest["trend_up_streak"]:
            tf_ok = latest_15m["EMA20"] > prev_15m["EMA20"]
        elif latest["trend_down_streak"]:
            tf_ok = latest_15m["EMA20"] < prev_15m["EMA20"]
        else:
            tf_ok = False
    else:
        tf_ok = False

# --------------------------
# 信号计算函数
# --------------------------
def calc_signal_from_row(latest, prev, params):
    trend = None
    if latest["trend_up_streak"] and latest["ADX"] > params['adx_threshold'] and params['tf_ok']:
        trend = "多"
    elif latest["trend_down_streak"] and latest["ADX"] > params['adx_threshold'] and params['tf_ok']:
        trend = "空"

    signal = None
    if trend == "多":
        if latest["close"] > latest["EMA_slow"] and \
           abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * params['callback_atr_mult'] and \
           prev["close"] < prev["EMA_fast"]:
            body = abs(latest["close"] - latest["open"])
            lower_shadow = min(latest["close"], latest["open"]) - latest["low"]
            if lower_shadow > body * 1.5 or latest["close"] > latest["open"]:
                signal = "多"
    elif trend == "空":
        if latest["close"] < latest["EMA_slow"] and \
           abs(latest["close"] - latest["EMA_fast"]) < latest["ATR"] * params['callback_atr_mult'] and \
           prev["close"] > prev["EMA_fast"]:
            body = abs(latest["close"] - latest["open"])
            upper_shadow = latest["high"] - max(latest["close"], latest["open"])
            if upper_shadow > body * 1.5 or latest["close"] < latest["open"]:
                signal = "空"
    return signal, trend

# 实时信号
signal_params = {
    'adx_threshold': adx_threshold,
    'callback_atr_mult': callback_atr_mult,
    'tf_ok': tf_ok
}
signal, trend = calc_signal_from_row(latest, prev, signal_params)

# --------------------------
# 回测模块（独立分支）
# --------------------------
def run_backtest(df_full, days, params):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df_backtest = df_full[df_full["ts"] >= start_date].copy().reset_index(drop=True)
    if len(df_backtest) < 50:
        return None

    class BacktestAccount:
        def __init__(self, initial):
            self.balance = initial
            self.position = None
            self.trades = []
            self.equity_curve = []

        def step(self, row, signal):
            self.equity_curve.append(self.balance)
            if self.position:
                entry = self.position['entry']
                stop = self.position['stop']
                tp = self.position['tp']
                qty = self.position['qty']
                direction = self.position['direction']
                if direction == "多":
                    if row['low'] <= stop:
                        close_price = stop
                        pnl = (close_price - entry) * qty
                        self.balance += pnl
                        self.trades.append({'entry': entry, 'exit': close_price, 'pnl': pnl, 'reason': '止损'})
                        self.position = None
                    elif row['high'] >= tp:
                        close_price = tp
                        pnl = (close_price - entry) * qty
                        self.balance += pnl
                        self.trades.append({'entry': entry, 'exit': close_price, 'pnl': pnl, 'reason': '止盈'})
                        self.position = None
                else:
                    if row['high'] >= stop:
                        close_price = stop
                        pnl = (entry - close_price) * qty
                        self.balance += pnl
                        self.trades.append({'entry': entry, 'exit': close_price, 'pnl': pnl, 'reason': '止损'})
                        self.position = None
                    elif row['low'] <= tp:
                        close_price = tp
                        pnl = (entry - close_price) * qty
                        self.balance += pnl
                        self.trades.append({'entry': entry, 'exit': close_price, 'pnl': pnl, 'reason': '止盈'})
                        self.position = None

            if signal and not self.position:
                atr = row['ATR']
                if signal == "多":
                    stop = row['close'] - atr * 1.5
                    tp = row['close'] + (row['close'] - stop) * risk_reward
                else:
                    stop = row['close'] + atr * 1.5
                    tp = row['close'] - (stop - row['close']) * risk_reward
                risk_amount = self.balance * (risk_percent / 100)
                stop_dist = abs(row['close'] - stop)
                if stop_dist > 0:
                    qty = risk_amount / stop_dist
                    qty = round(qty / 0.01) * 0.01
                    if qty < 0.01:
                        qty = 0.01
                    self.position = {
                        'direction': signal,
                        'entry': row['close'],
                        'qty': qty,
                        'stop': stop,
                        'tp': tp
                    }

    bt_account = BacktestAccount(initial_balance)
    for i in range(1, len(df_backtest)):
        row = df_backtest.iloc[i]
        prev_row = df_backtest.iloc[i-1]
        sig_params = {
            'adx_threshold': adx_threshold,
            'callback_atr_mult': callback_atr_mult,
            'tf_ok': True  # 回测简化，忽略多时间框架
        }
        sig, _ = calc_signal_from_row(row, prev_row, sig_params)
        bt_account.step(row, sig)

    if bt_account.position:
        last_row = df_backtest.iloc[-1]
        entry = bt_account.position['entry']
        qty = bt_account.position['qty']
        direction = bt_account.position['direction']
        if direction == "多":
            pnl = (last_row['close'] - entry) * qty
        else:
            pnl = (entry - last_row['close']) * qty
        bt_account.balance += pnl
        bt_account.trades.append({'entry': entry, 'exit': last_row['close'], 'pnl': pnl, 'reason': '平仓'})
        bt_account.position = None

    # 绩效指标
    trades_df = pd.DataFrame(bt_account.trades)
    wins = len(trades_df[trades_df['pnl'] > 0]) if not trades_df.empty else 0
    losses = len(trades_df[trades_df['pnl'] < 0]) if not trades_df.empty else 0
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0
    net_pnl = bt_account.balance - initial_balance

    # 夏普比率
    if len(bt_account.equity_curve) > 1:
        returns = np.diff(bt_account.equity_curve) / bt_account.equity_curve[:-1]
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 12)  # 5分钟K线年化
        else:
            sharpe = 0
    else:
        sharpe = 0

    # 最大回撤
    peak = np.maximum.accumulate(bt_account.equity_curve)
    drawdown = np.where(peak > 0, (peak - bt_account.equity_curve) / peak, 0)
    max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

    return {
        'initial': initial_balance,
        'final': bt_account.balance,
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': bt_account.trades
    }

if backtest_mode:
    st.subheader("📊 回测结果")
    with st.spinner("回测进行中..."):
        result = run_backtest(df, days_back, {})
    if result:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("初始资金", f"{result['initial']:.2f}")
        with col2:
            st.metric("最终资金", f"{result['final']:.2f}")
        with col3:
            st.metric("总交易", result['total_trades'])
        with col4:
            st.metric("胜率", f"{result['win_rate']:.1f}%")
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("净盈亏", f"{result['net_pnl']:+.2f}")
        with col6:
            st.metric("夏普比率", f"{result['sharpe']:.2f}")
        with col7:
            st.metric("最大回撤", f"{result['max_drawdown']:.2f}%")
        if result['trades']:
            st.dataframe(pd.DataFrame(result['trades']).tail(10))
    else:
        st.warning("回测数据不足，请减小天数")
    st.stop()

# --------------------------
# 实时模式：信号持久化与模拟账户
# --------------------------
def load_signal_history():
    if os.path.exists(SIGNAL_HISTORY_FILE):
        try:
            with open(SIGNAL_HISTORY_FILE, 'r') as f:
                data = json.load(f)
                for item in data:
                    item['time'] = datetime.fromisoformat(item['time'])
                return data
        except:
            logging.exception("信号历史文件损坏，重置")
            return []
    return []

def save_signal_history(history):
    if len(history) > 500:
        history = history[-500:]
    serializable = []
    for item in history:
        copy = item.copy()
        copy['time'] = item['time'].isoformat()
        serializable.append(copy)
    with open(SIGNAL_HISTORY_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)

if "signal_history" not in st.session_state:
    st.session_state.signal_history = load_signal_history()

if signal:
    last_signal_time = st.session_state.signal_history[-1]["time"] if st.session_state.signal_history else None
    if last_signal_time != latest["ts"]:
        st.session_state.signal_history.append({
            "time": latest["ts"],
            "direction": signal,
            "price": latest["close"],
            "ema_fast": latest["EMA_fast"],
            "atr": latest["ATR"]
        })
        save_signal_history(st.session_state.signal_history)
        logging.info(f"新信号: {signal} @ {latest['close']}")

# --------------------------
# 线程安全模拟账户
# --------------------------
class TradingAccount:
    def __init__(self, initial_balance):
        self.initial = initial_balance
        self.file = ACCOUNT_FILE
        self.lock = threading.Lock()
        self.load()

    def load(self):
        if os.path.exists(self.file):
            with self.lock:
                try:
                    with open(self.file, 'r') as f:
                        data = json.load(f)
                        self.balance = data.get('balance', self.initial)
                        self.position = data.get('position', None)
                        self.trades = data.get('trades', [])
                except:
                    logging.exception("账户文件损坏，重置")
                    self.reset()
        else:
            self.reset()

    def save(self):
        if len(self.trades) > 100:
            self.trades = self.trades[-100:]
        with self.lock:
            data = {
                'balance': self.balance,
                'position': self.position,
                'trades': self.trades
            }
            with open(self.file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

    def reset(self):
        self.balance = self.initial
        self.position = None
        self.trades = []
        self.save()
        logging.info("账户重置")

    def can_open(self):
        return self.position is None

    def open_position(self, direction, price, atr):
        if direction == "多":
            stop_loss = price - atr * 1.5
            take_profit = price + (price - stop_loss) * risk_reward
        else:
            stop_loss = price + atr * 1.5
            take_profit = price - (stop_loss - price) * risk_reward

        risk_amount = self.balance * (risk_percent / 100)
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            return False, "止损距离为零"

        raw_qty = risk_amount / stop_distance
        qty = round(raw_qty / 0.01) * 0.01
        if qty < 0.01:
            qty = 0.01

        fee = price * qty * 0.0005
        if self.balance < fee:
            return False, "余额不足"

        self.balance -= fee
        self.position = {
            'direction': direction,
            'entry': price,
            'qty': qty,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'open_time': datetime.now().isoformat()
        }
        self.save()
        logging.info(f"开仓 {direction} @ {price}, 数量 {qty}")
        return True, "开仓成功"

    def check_stop_take(self, current_price):
        if not self.position:
            return None
        pos = self.position
        direction = pos['direction']
        entry = pos['entry']
        stop = pos['stop_loss']
        tp = pos['take_profit']
        qty = pos['qty']

        if direction == "多":
            if current_price <= stop:
                reason, close_price = "止损", stop
            elif current_price >= tp:
                reason, close_price = "止盈", tp
            else:
                return None
        else:
            if current_price >= stop:
                reason, close_price = "止损", stop
            elif current_price <= tp:
                reason, close_price = "止盈", tp
            else:
                return None

        pnl = (close_price - entry) * qty if direction == "多" else (entry - close_price) * qty
        fee = close_price * qty * 0.0005
        net_pnl = pnl - fee

        self.balance += net_pnl
        self.trades.append({
            **pos,
            'close_price': close_price,
            'close_time': datetime.now().isoformat(),
            'reason': reason,
            'pnl': pnl,
            'fee': fee,
            'net_pnl': net_pnl
        })
        self.position = None
        self.save()
        logging.info(f"平仓 {reason}, 盈亏 {net_pnl:.2f}")
        return reason, net_pnl

    def get_equity(self, current_price):
        if not self.position:
            return self.balance
        pos = self.position
        if pos['direction'] == "多":
            return self.balance + (current_price - pos['entry']) * pos['qty']
        else:
            return self.balance + (pos['entry'] - current_price) * pos['qty']

account = TradingAccount(initial_balance)

# 开仓与止损止盈检查
if signal and account.can_open():
    success, msg = account.open_position(signal, latest["close"], latest["ATR"])
    if success:
        st.sidebar.success(f"✅ 开仓 {signal}")
    else:
        st.sidebar.warning(f"开仓失败: {msg}")

if account.position:
    result = account.check_stop_take(latest["close"])
    if result:
        reason, net = result
        st.sidebar.info(f"📌 平仓: {reason}, 盈亏: {net:+.2f} USDT")

# --------------------------
# 胜率统计
# --------------------------
def calculate_performance(trades):
    if not trades:
        return {}
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t['net_pnl'] > 0]
    losses = df_t[df_t['net_pnl'] < 0]
    total = len(df_t)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    profit_factor = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf')
    return {
        'total_trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'net_pnl_total': df_t['net_pnl'].sum()
    }

# --------------------------
# 绘图（含防抖和优化注记位置）
# --------------------------
def find_swing_highs(df, window=3):
    highs = []
    for i in range(window, len(df)-window):
        left = df['high'].iloc[i-window:i].max()
        right = df['high'].iloc[i+1:i+window+1].max()
        if df['high'].iloc[i] > left and df['high'].iloc[i] > right:
            highs.append((df['ts'].iloc[i], df['high'].iloc[i]))
    return highs

def find_swing_lows(df, window=3):
    lows = []
    for i in range(window, len(df)-window):
        left = df['low'].iloc[i-window:i].min()
        right = df['low'].iloc[i+1:i+window+1].min()
        if df['low'].iloc[i] < left and df['low'].iloc[i] < right:
            lows.append((df['ts'].iloc[i], df['low'].iloc[i]))
    return lows

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["ts"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_fast"], line=dict(color="blue", width=1), name=f"EMA{ema_period_fast}"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["EMA_slow"], line=dict(color="orange", width=1), name=f"EMA{ema_period_slow}"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["support_roll"], line=dict(color="green", width=1, dash="dash"), name="支撑(滚动)"))
fig.add_trace(go.Scatter(x=df["ts"], y=df["resistance_roll"], line=dict(color="red", width=1, dash="dash"), name="阻力(滚动)"))

swing_highs = find_swing_highs(df, window=3)
swing_lows = find_swing_lows(df, window=3)
if swing_highs:
    sh_x, sh_y = zip(*swing_highs)
    fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='结构高点'))
if swing_lows:
    sl_x, sl_y = zip(*swing_lows)
    fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='结构低点'))

# 防抖：仅当信号时间戳变化时才添加新标记
if signal and st.session_state.get("last_signal_ts") != latest["ts"]:
    st.session_state.last_signal_ts = latest["ts"]
    # 计算注记位置：多单放高点上方，空单放低点下方，避免遮挡K线
    if signal == "多":
        y_pos = latest["high"] * 1.002
        symbol = "arrow-up"
    else:
        y_pos = latest["low"] * 0.998
        symbol = "arrow-down"
    fig.add_trace(go.Scatter(
        x=[latest["ts"]],
        y=[y_pos],
        mode="markers+text",
        marker=dict(symbol=symbol, size=15, color="yellow"),
        text=signal,
        textposition="top center" if signal == "多" else "bottom center",
        name="新信号"
    ))

fig.update_layout(
    title=f"{SYMBOL} 5分钟图（最终完美版V2）",
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, width='stretch')

# --------------------------
# 状态面板
# --------------------------
st.subheader("📊 当前状态")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("趋势方向", trend if trend else "无")
with col2:
    st.metric("ADX", f"{latest['ADX']:.1f}")
with col3:
    st.metric("K线收盘价", f"{latest['close']:.2f}")
with col4:
    if real_price:
        st.metric("实时价格", f"{real_price:.2f}", delta=f"{real_price - latest['close']:.2f}")
    else:
        st.metric("实时价格", "N/A")
with col5:
    st.metric("ATR", f"{latest['ATR']:.4f}")

st.write(f"**滚动支撑** (最近{lookback_sr}根): {latest['support_roll']:.2f}")
st.write(f"**滚动阻力** (最近{lookback_sr}根): {latest['resistance_roll']:.2f}")
if use_tf_filter and st.session_state.cached_15m is not None:
    st.write(f"**15分钟EMA方向**: {'上升' if tf_ok else '下降或持平'}")
if signal:
    st.success(f"📈 当前信号: {signal}")
else:
    st.info("⏳ 无信号")

st.caption(f"数据更新于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --------------------------
# 账户面板
# --------------------------
st.subheader("💰 模拟账户")
col_acc1, col_acc2, col_acc3 = st.columns(3)
with col_acc1:
    st.metric("余额 (USDT)", f"{account.balance:.2f}")
with col_acc2:
    equity = account.get_equity(real_price if real_price else latest["close"])
    st.metric("权益 (USDT)", f"{equity:.2f}")
with col_acc3:
    if account.position:
        st.metric("持仓", f"{account.position['direction']} {account.position['qty']:.2f}张")
    else:
        st.metric("持仓", "无")

if account.position:
    pos = account.position
    st.write(f"开仓价: {pos['entry']:.2f} | 止损: {pos['stop_loss']:.2f} | 止盈: {pos['take_profit']:.2f}")

if st.button("重置账户"):
    account.reset()
    st.rerun()

# --------------------------
# 信号历史与绩效
# --------------------------
st.subheader("📜 信号历史")
if st.session_state.signal_history:
    hist_df = pd.DataFrame(st.session_state.signal_history)
    hist_df['time'] = hist_df['time'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(hist_df[['time', 'direction', 'price', 'ema_fast', 'atr']], width='stretch')

st.subheader("📈 交易绩效统计")
if account.trades:
    perf = calculate_performance(account.trades)
    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
    with col_perf1:
        st.metric("总交易", perf['total_trades'])
    with col_perf2:
        st.metric("胜率", f"{perf['win_rate']:.1f}%")
    with col_perf3:
        st.metric("总盈亏", f"{perf['net_pnl_total']:+.2f}")
    with col_perf4:
        pf = perf['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        st.metric("盈亏比", pf_str)
    st.dataframe(pd.DataFrame(account.trades).tail(10)[['open_time', 'direction', 'entry', 'close_price', 'reason', 'net_pnl']])
else:
    st.info("暂无交易记录")

# 程序退出时停止数据线程
import atexit
atexit.register(lambda: st.session_state.fetcher.stop() if "fetcher" in st.session_state else None)
