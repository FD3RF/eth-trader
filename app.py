# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 超神终极版 36.2
==================================================
更新：优化回测引擎，支持完整止损止盈、部分止盈、超时平仓、自适应仓位计算
==================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import warnings
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import functools
import hashlib
import csv
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==================== 日志文件持久化配置 ====================
LOG_DIR = "logs"
TRADE_LOG_FILE = "trade_log.csv"
PERF_LOG_FILE = "performance_log.csv"
os.makedirs(LOG_DIR, exist_ok=True)

def append_to_csv(file_path: str, row: dict):
    file_exists = os.path.isfile(file_path)
    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"写入CSV失败: {e}")

def append_to_log(file_name: str, message: str):
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{file_name}_{date_str}.log")
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    except Exception as e:
        print(f"写入日志失败: {e}")

# ==================== 配置与常量 ====================
class SignalStrength(Enum):
    EXTREME = 0.85
    STRONG = 0.75
    HIGH = 0.65
    MEDIUM = 0.55
    WEAK = 0.50      # 低于此值的信号强制忽略
    NONE = 0.0

class MarketRegime(Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    PANIC = "PANIC"
    CALM = "CALM"

@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"])
    base_risk_per_trade: float = 0.02
    risk_budget_ratio: float = 0.10
    daily_loss_limit: float = 300.0
    max_drawdown_pct: float = 20.0
    min_atr_pct: float = 0.5
    tp_min_ratio: float = 2.0
    partial_tp_ratio: float = 0.5
    partial_tp_r_multiple: float = 1.2
    trailing_stop_pct: float = 0.3
    breakeven_trigger_pct: float = 1.5
    max_hold_hours: int = 36
    max_consecutive_losses: int = 3
    cooldown_losses: int = 3
    cooldown_hours: int = 24
    max_daily_trades: int = 5
    leverage_modes: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "保守 (1-2x)": (1, 2),
        "稳健 (3-5x)": (3, 5),
        "进取 (5-8x)": (5, 8),
        "极限 (8-10x)": (8, 10)
    })
    exchanges: Dict[str, Any] = field(default_factory=lambda: {
        "Binance合约": ccxt.binance,
        "Bybit合约": ccxt.bybit,
        "OKX合约": ccxt.okx
    })
    data_sources: List[str] = field(default_factory=lambda: ["binance", "bybit", "okx", "mexc", "kucoin"])
    timeframes: List[str] = field(default_factory=lambda: ['15m', '1h', '4h', '1d'])
    timeframe_weights: Dict[str, int] = field(default_factory=lambda: {'1d': 10, '4h': 7, '1h': 5, '15m': 3})
    fetch_limit: int = 1000
    auto_refresh_ms: int = 30000
    anti_duplicate_seconds: int = 180
    kelly_fraction: float = 0.25
    atr_multiplier_base: float = 1.5
    max_leverage_global: float = 10.0
    circuit_breaker_atr: float = 5.0
    circuit_breaker_fg_extreme: Tuple[int, int] = (10, 90)
    slippage_base: float = 0.0003
    fee_rate: float = 0.0004
    ic_window: int = 80
    mc_simulations: int = 500
    sim_volatility: float = 0.06
    sim_trend_strength: float = 0.2
    # 自适应参数窗口
    adapt_window: int = 20
    # 因子权重学习率
    factor_learning_rate: float = 0.3
    # VaR置信水平
    var_confidence: float = 0.95

CONFIG = TradingConfig()

# ==================== 全局变量（用于在线学习）====================
factor_weights = {
    'trend': 1.0,
    'rsi': 1.0,
    'macd': 1.0,
    'bb': 1.0,
    'volume': 1.0,
    'adx': 1.0
}
# 因子到实际列名的映射（用于IC计算）
factor_to_col = {
    'trend': 'trend_factor',
    'rsi': 'rsi',
    'macd': 'macd_diff',
    'bb': 'bb_factor',
    'volume': 'volume_ratio',
    'adx': 'adx'
}
factor_performance = deque(maxlen=100)  # 存储 (factor_name, ic) 用于学习

# ==================== 日志系统 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UltimateTrader")

# ==================== 辅助函数 ====================
def safe_request(max_retries: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(2 ** attempt)
            return None
        return wrapper
    return decorator

def init_session_state():
    defaults = {
        'account_balance': 10000.0,
        'daily_pnl': 0.0,
        'peak_balance': 10000.0,
        'consecutive_losses': 0,
        'daily_trades': 0,
        'trade_log': [],
        'position': None,
        'auto_enabled': True,
        'pause_until': None,
        'exchange': None,
        'net_value_history': [],
        'last_signal_time': None,
        'current_symbol': 'ETH/USDT',
        'telegram_token': None,
        'telegram_chat_id': None,
        'backtest_results': None,
        'circuit_breaker': False,
        'cooldown_until': None,
        'mc_results': None,
        'use_simulated_data': False,
        'data_source_failed': False,
        'error_log': deque(maxlen=20),
        'execution_log': deque(maxlen=50),
        'last_trade_date': None,
        'exchange_choice': 'Binance合约',
        'testnet': True,
        'use_real': False,
        'binance_api_key': '',
        'binance_secret_key': '',
        'fear_greed': 50,
        'market_regime': MarketRegime.RANGE,
        'multi_df': {},
        'performance_metrics': {'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0, 'sharpe': 0.0},
        'mode': 'live',  # 'live' or 'backtest'
        'backtest_data': None,
        'backtest_index': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_error(msg: str):
    st.session_state.error_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    append_to_log("error", msg)
    logger.error(msg)

def log_execution(msg: str):
    st.session_state.execution_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    append_to_log("execution", msg)

def send_telegram(msg: str):
    token = st.session_state.get('telegram_token')
    chat_id = st.session_state.get('telegram_chat_id')
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": msg}, timeout=3)
        except:
            pass

def update_performance_metrics():
    """从交易日志计算近期绩效指标"""
    trades = st.session_state.trade_log[-50:]
    if len(trades) < 5:
        return
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    win_rate = len(wins) / len(df) if len(df) > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 1
    returns = df['pnl'].values / st.session_state.account_balance
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() != 0 else 0
    st.session_state.performance_metrics = {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe
    }

# ==================== 自适应ATR倍数 ====================
def adaptive_atr_multiplier(price_series: pd.Series, window: int = 20) -> float:
    if len(price_series) < window:
        return CONFIG.atr_multiplier_base
    returns = price_series.pct_change().dropna()
    vol = returns.std() * np.sqrt(365 * 24 * 4)
    base_vol = 0.5
    ratio = base_vol / max(vol, 0.1)
    new_mult = CONFIG.atr_multiplier_base * np.clip(ratio, 0.5, 2.0)
    return new_mult

# ==================== 在线学习因子权重 ====================
def update_factor_weights(ic_dict: Dict[str, float]):
    """根据IC值更新因子权重（指数加权移动平均）"""
    global factor_weights
    lr = CONFIG.factor_learning_rate
    for factor, ic in ic_dict.items():
        if factor in factor_weights and not np.isnan(ic):
            adjustment = 1 + lr * ic
            factor_weights[factor] = max(0.1, factor_weights[factor] * adjustment)

# ==================== 超真实模拟数据生成器 ====================
def generate_simulated_data(symbol: str, limit: int = 1500) -> Dict[str, pd.DataFrame]:
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % 2**32
    np.random.seed(seed)
    end = datetime.now()
    timestamps = pd.date_range(end=end, periods=limit, freq='15min')
    
    if 'BTC' in symbol:
        base = 40000
        volatility = CONFIG.sim_volatility * 0.6
        trend_factor = 0.1
    elif 'ETH' in symbol:
        base = 2100
        volatility = CONFIG.sim_volatility
        trend_factor = 0.15
    else:
        base = 100
        volatility = CONFIG.sim_volatility * 1.2
        trend_factor = 0.2
    
    t = np.linspace(0, 6*np.pi, limit)
    trend_direction = np.random.choice([-1, 1], p=[0.3, 0.7])
    trend = trend_direction * CONFIG.sim_trend_strength * np.linspace(0, 1, limit) * base * trend_factor
    cycle1 = 0.03 * base * np.sin(t * 1)
    cycle2 = 0.015 * base * np.sin(t * 3)
    cycle3 = 0.007 * base * np.sin(t * 7)
    random_step = np.random.randn(limit) * volatility * base
    random_walk = np.cumsum(random_step) * 0.15
    price_series = base + trend + cycle1 + cycle2 + cycle3 + random_walk
    price_series = np.maximum(price_series, base * 0.3)
    
    opens = price_series * (1 + np.random.randn(limit) * 0.0015)
    closes = price_series * (1 + np.random.randn(limit) * 0.0025)
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(limit)) * volatility * price_series * 0.5
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(limit)) * volatility * price_series * 0.5
    volume_base = np.random.randint(800, 8000, limit)
    volume_factor = 1 + 3 * np.abs(np.diff(price_series, prepend=price_series[0])) / price_series
    volumes = (volume_base * volume_factor).astype(int)
    
    df_15m = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    df_15m = add_indicators(df_15m)
    
    data_dict = {'15m': df_15m}
    for tf in ['1h', '4h', '1d']:
        resampled = df_15m.resample(tf, on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        if len(resampled) >= 30:
            resampled = add_indicators(resampled)
            data_dict[tf] = resampled
    return data_dict

# ==================== 技术指标计算（新增连续因子列）====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    if len(df) >= 14:
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr
        df['atr_ma'] = atr.rolling(20).mean()
    else:
        df['rsi'] = np.nan
        df['atr'] = np.nan
        df['atr_ma'] = np.nan
    if len(df) >= 26:
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = df['macd'] - df['macd_signal']
    else:
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_diff'] = np.nan
    if len(df) >= 14:
        try:
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        except:
            df['adx'] = np.nan
    else:
        df['adx'] = np.nan
    if len(df) >= 20:
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        # 布林带相对位置因子（0-1之间）
        df['bb_factor'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_width'] = np.nan
        df['bb_factor'] = np.nan
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    # 趋势因子：价格与EMA20的差值（归一化）
    df['trend_factor'] = (df['close'] - df['ema20']) / df['close']
    if len(df) >= 6:
        df['future_ret'] = df['close'].pct_change(5).shift(-5)
    else:
        df['future_ret'] = np.nan
    return df

# ==================== 因子IC计算 ====================
_ic_cache = {}
def calculate_ic(df: pd.DataFrame, factor_name: str) -> float:
    """计算指定因子列与未来收益的相关系数"""
    key = (id(df), factor_name)
    if key in _ic_cache:
        return _ic_cache[key]
    window = min(CONFIG.ic_window, len(df) - 6)
    if window < 20:
        return 0.0
    factor = df[factor_name].iloc[-window:-5]
    future = df['future_ret'].iloc[-window:-5]
    valid = factor.notna() & future.notna()
    if valid.sum() < 10:
        return 0.0
    ic = factor[valid].corr(future[valid])
    ic = 0.0 if pd.isna(ic) else ic
    _ic_cache[key] = ic
    return ic

# ==================== 独立缓存函数 ====================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_fear_greed() -> int:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        return int(r.json()['data'][0]['value'])
    except Exception:
        return 50

# ==================== 并行数据获取器 ====================
@st.cache_resource
def get_fetcher() -> 'AggregatedDataFetcher':
    return AggregatedDataFetcher()

class AggregatedDataFetcher:
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        for name in CONFIG.data_sources:
            try:
                cls = getattr(ccxt, name)
                ex = cls({'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'future'}})
                self.exchanges[name] = ex
            except Exception:
                pass

    @safe_request()
    def _fetch_kline_single(self, ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) >= 50:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.astype({col: float for col in ['open','high','low','close','volume']})
                return df
        except Exception as e:
            log_error(f"{ex.id} 获取失败: {e}")
        return None

    def _fetch_kline_parallel(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        futures = []
        for name in ["binance"] + [n for n in CONFIG.data_sources if n != "binance"]:
            if name in self.exchanges:
                ex = self.exchanges[name]
                futures.append(self.executor.submit(self._fetch_kline_single, ex, symbol, timeframe, limit))
        for future in as_completed(futures):
            result = future.result(timeout=10)
            if result is not None:
                for f in futures:
                    f.cancel()
                return result
        return None

    def fetch_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        for tf in CONFIG.timeframes:
            df = self._fetch_kline_parallel(symbol, tf, CONFIG.fetch_limit)
            if df is not None and len(df) >= 50:
                df = add_indicators(df)
                data_dict[tf] = df
        return data_dict

    def fetch_funding_rate(self, symbol: str) -> float:
        rates = []
        for name in ["binance"] + [n for n in CONFIG.data_sources if n != "binance"]:
            if name in self.exchanges:
                try:
                    rates.append(self.exchanges[name].fetch_funding_rate(symbol)['fundingRate'])
                except Exception:
                    continue
        return float(np.mean(rates)) if rates else 0.0

    def fetch_orderbook_imbalance(self, symbol: str, depth: int = 10) -> float:
        for name in ["binance"] + [n for n in CONFIG.data_sources if n != "binance"]:
            if name in self.exchanges:
                try:
                    ob = self.exchanges[name].fetch_order_book(symbol, limit=depth)
                    bid_vol = sum(b[1] for b in ob['bids'])
                    ask_vol = sum(a[1] for a in ob['asks'])
                    total = bid_vol + ask_vol
                    return (bid_vol - ask_vol) / total if total > 0 else 0.0
                except Exception:
                    continue
        return 0.0

    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        if st.session_state.use_simulated_data:
            sim_data = generate_simulated_data(symbol)
            return {
                "data_dict": sim_data,
                "current_price": sim_data['15m']['close'].iloc[-1],
                "fear_greed": 50,
                "funding_rate": 0.0,
                "orderbook_imbalance": 0.0,
            }
        data_dict = self.fetch_all_timeframes(symbol)
        if '15m' not in data_dict or data_dict['15m'].empty:
            log_error("所有数据源获取失败，自动切换模拟模式")
            st.session_state.use_simulated_data = True
            sim_data = generate_simulated_data(symbol)
            return {
                "data_dict": sim_data,
                "current_price": sim_data['15m']['close'].iloc[-1],
                "fear_greed": 50,
                "funding_rate": 0.0,
                "orderbook_imbalance": 0.0,
            }
        current_price = float(data_dict['15m']['close'].iloc[-1])
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "fear_greed": fetch_fear_greed(),
            "funding_rate": self.fetch_funding_rate(symbol),
            "orderbook_imbalance": self.fetch_orderbook_imbalance(symbol),
        }

# ==================== 信号引擎（带在线学习）====================
class SignalEngine:
    def __init__(self):
        pass

    def detect_market_regime(self, df_dict: Dict[str, pd.DataFrame]) -> MarketRegime:
        if '1h' not in df_dict or '4h' not in df_dict:
            return MarketRegime.RANGE
        df1h = df_dict['1h']
        df4h = df_dict['4h']
        adx1h = df1h['adx'].iloc[-1] if not pd.isna(df1h['adx'].iloc[-1]) else 25
        adx4h = df4h['adx'].iloc[-1] if not pd.isna(df4h['adx'].iloc[-1]) else 25
        avg_adx = (adx1h + adx4h) / 2
        close1h = df1h['close'].iloc[-1]
        ema20_1h = df1h['ema20'].iloc[-1] if not pd.isna(df1h['ema20'].iloc[-1]) else close1h
        close4h = df4h['close'].iloc[-1]
        ema20_4h = df4h['ema20'].iloc[-1] if not pd.isna(df4h['ema20'].iloc[-1]) else close4h
        trend_up = (close1h > ema20_1h) and (close4h > ema20_4h)
        trend_down = (close1h < ema20_1h) and (close4h < ema20_4h)
        if avg_adx > 30:
            if trend_up or trend_down:
                return MarketRegime.TREND
            else:
                return MarketRegime.RANGE
        elif st.session_state.fear_greed <= 20:
            return MarketRegime.PANIC
        else:
            return MarketRegime.RANGE

    def calc_signal(self, df_dict: Dict[str, pd.DataFrame]) -> Tuple[int, float]:
        global factor_weights
        total_score = 0
        total_weight = 0
        tf_votes = []
        regime = st.session_state.market_regime
        ic_dict = {}  # 存储各因子的平均IC

        for tf, df in df_dict.items():
            if df.empty or len(df) < 2:
                continue
            last = df.iloc[-1]
            weight = CONFIG.timeframe_weights.get(tf, 1)
            if regime == MarketRegime.TREND:
                if tf in ['4h', '1d']:
                    weight *= 1.5
            elif regime == MarketRegime.RANGE:
                if tf in ['15m', '1h']:
                    weight *= 1.3
            if pd.isna(last.get('ema20', np.nan)):
                continue

            # 各因子得分（使用当前权重）
            factor_scores = {}
            # 趋势因子
            if last['close'] > last['ema20']:
                factor_scores['trend'] = 1 * factor_weights['trend']
            elif last['close'] < last['ema20']:
                factor_scores['trend'] = -1 * factor_weights['trend']
            else:
                factor_scores['trend'] = 0

            # RSI
            if last['rsi'] > 70:
                factor_scores['rsi'] = -0.7 * factor_weights['rsi']
            elif last['rsi'] < 30:
                factor_scores['rsi'] = 0.7 * factor_weights['rsi']
            else:
                factor_scores['rsi'] = 0

            # MACD
            if last['macd_diff'] > 0:
                factor_scores['macd'] = 0.8 * factor_weights['macd']
            elif last['macd_diff'] < 0:
                factor_scores['macd'] = -0.8 * factor_weights['macd']
            else:
                factor_scores['macd'] = 0

            # 布林带
            if not pd.isna(last.get('bb_upper')):
                if last['close'] > last['bb_upper']:
                    factor_scores['bb'] = -0.5 * factor_weights['bb']
                elif last['close'] < last['bb_lower']:
                    factor_scores['bb'] = 0.5 * factor_weights['bb']
                else:
                    factor_scores['bb'] = 0
            else:
                factor_scores['bb'] = 0

            # 成交量
            if not pd.isna(last.get('volume_ratio')):
                factor_scores['volume'] = (1.2 if last['volume_ratio'] > 1.5 else 0) * factor_weights['volume']
            else:
                factor_scores['volume'] = 0

            # ADX
            adx = last.get('adx', 25)
            if pd.isna(adx):
                factor_scores['adx'] = 0
            else:
                factor_scores['adx'] = (0.3 if adx > 30 else -0.2 if adx < 20 else 0) * factor_weights['adx']

            # 计算当前周期各因子的IC（用于后续权重更新）
            for fname in factor_scores.keys():
                col = factor_to_col.get(fname)
                if col and col in df.columns:
                    ic = calculate_ic(df, col)
                    if fname not in ic_dict:
                        ic_dict[fname] = []
                    ic_dict[fname].append(ic)

            # 加权组合得到本周期得分
            tf_score = sum(factor_scores.values()) * weight
            total_score += tf_score
            total_weight += weight
            if tf_score > 0:
                tf_votes.append(1)
            elif tf_score < 0:
                tf_votes.append(-1)

        # 更新因子权重（使用各因子在所有周期上的平均IC）
        avg_ic = {}
        for fname, ic_list in ic_dict.items():
            avg_ic[fname] = np.nanmean(ic_list) if ic_list else 0.0
        update_factor_weights(avg_ic)

        if total_weight == 0:
            return 0, 0.0
        max_possible = sum(CONFIG.timeframe_weights.values()) * 3.5
        prob_raw = min(1.0, abs(total_score) / max_possible) if max_possible > 0 else 0.5
        prob = 0.5 + 0.45 * prob_raw

        if prob < SignalStrength.WEAK.value:
            return 0, prob

        if prob >= SignalStrength.WEAK.value:
            direction = 1 if total_score > 0 else -1 if total_score < 0 else 0
        else:
            if tf_votes:
                direction = 1 if sum(tf_votes) > 0 else -1 if sum(tf_votes) < 0 else 0
            else:
                direction = 0
        if direction == 0:
            prob = 0.0
        return direction, prob

# ==================== 风险管理（含VaR）====================
class RiskManager:
    def __init__(self):
        pass

    def check_daily_limit(self) -> bool:
        today = datetime.now().date()
        if st.session_state.get('last_trade_date') != today:
            st.session_state.daily_trades = 0
            st.session_state.last_trade_date = today
        return st.session_state.daily_trades >= CONFIG.max_daily_trades

    def check_cooldown(self) -> bool:
        until = st.session_state.get('cooldown_until')
        return until is not None and datetime.now() < until

    def update_losses(self, win: bool):
        if not win:
            st.session_state.consecutive_losses += 1
            if st.session_state.consecutive_losses >= CONFIG.cooldown_losses:
                st.session_state.cooldown_until = datetime.now() + timedelta(hours=CONFIG.cooldown_hours)
        else:
            st.session_state.consecutive_losses = 0
            st.session_state.cooldown_until = None

    def check_circuit_breaker(self, atr_pct: float, fear_greed: int) -> bool:
        return atr_pct > CONFIG.circuit_breaker_atr or fear_greed <= CONFIG.circuit_breaker_fg_extreme[0] or fear_greed >= CONFIG.circuit_breaker_fg_extreme[1]

    def check_max_drawdown(self) -> bool:
        drawdown = (st.session_state.peak_balance - st.session_state.account_balance) / st.session_state.peak_balance * 100
        return drawdown > CONFIG.max_drawdown_pct

    def calc_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        if len(returns) < 10:
            return 0.02
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)

    def calc_position_size(self, balance: float, prob: float, atr: float, price: float, recent_returns: np.ndarray) -> float:
        if price <= 0 or prob < 0.5:
            return 0.0
        edge = max(0.05, prob - 0.5) * 2
        var = self.calc_var(recent_returns, CONFIG.var_confidence)
        risk_amount = balance * CONFIG.base_risk_per_trade * edge * CONFIG.kelly_fraction * (1 / max(var, 0.01))
        if atr == 0 or np.isnan(atr) or atr < price * CONFIG.min_atr_pct / 100:
            stop_distance = price * 0.01
        else:
            stop_distance = atr * adaptive_atr_multiplier(pd.Series(recent_returns))
        leverage_mode = st.session_state.get('leverage_mode', '稳健 (3-5x)')
        min_lev, max_lev = CONFIG.leverage_modes.get(leverage_mode, (3,5))
        max_size_by_leverage = balance * max_lev / price
        size_by_risk = risk_amount / stop_distance
        size = min(size_by_risk, max_size_by_leverage)
        return max(size, 0.001)

# ==================== 持仓管理（增强移动止损）====================
@dataclass
class Position:
    direction: int
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    take_profit: float
    initial_atr: float
    partial_taken: bool = False
    real: bool = False
    highest_price: float = 0.0
    lowest_price: float = 1e9
    atr_mult: float = CONFIG.atr_multiplier_base

    def __post_init__(self):
        if self.direction == 1:
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price

    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size * self.direction

    def stop_distance(self) -> float:
        if self.direction == 1:
            return self.entry_price - self.stop_loss
        else:
            return self.stop_loss - self.entry_price

    def update_stops(self, current_price: float, atr: float):
        self.atr_mult = adaptive_atr_multiplier(pd.Series([self.entry_price, current_price]))
        if self.direction == 1:
            if current_price > self.highest_price:
                self.highest_price = current_price
            new_stop = current_price - atr * self.atr_mult
            self.stop_loss = max(self.stop_loss, new_stop)
            new_tp = current_price + atr * self.atr_mult * CONFIG.tp_min_ratio
            self.take_profit = max(self.take_profit, new_tp)
            if current_price >= self.entry_price + self.stop_distance() * CONFIG.breakeven_trigger_pct:
                self.stop_loss = max(self.stop_loss, self.entry_price)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            new_stop = current_price + atr * self.atr_mult
            self.stop_loss = min(self.stop_loss, new_stop)
            new_tp = current_price - atr * self.atr_mult * CONFIG.tp_min_ratio
            self.take_profit = min(self.take_profit, new_tp)
            if current_price <= self.entry_price - self.stop_distance() * CONFIG.breakeven_trigger_pct:
                self.stop_loss = min(self.stop_loss, self.entry_price)

    def should_close(self, high: float, low: float, current_time: datetime) -> Tuple[bool, str, float]:
        if self.direction == 1:
            if low <= self.stop_loss:
                return True, "止损", self.stop_loss
            if high >= self.take_profit:
                return True, "止盈", self.take_profit
        else:
            if high >= self.stop_loss:
                return True, "止损", self.stop_loss
            if low <= self.take_profit:
                return True, "止盈", self.take_profit
        hold_hours = (current_time - self.entry_time).total_seconds() / 3600
        if hold_hours > CONFIG.max_hold_hours:
            return True, "超时", (high + low) / 2
        if not self.partial_taken:
            if self.direction == 1 and high >= self.entry_price + self.stop_distance() * CONFIG.partial_tp_r_multiple:
                self.partial_taken = True
                return True, "部分止盈", self.entry_price + self.stop_distance() * CONFIG.partial_tp_r_multiple
            if self.direction == -1 and low <= self.entry_price - self.stop_distance() * CONFIG.partial_tp_r_multiple:
                self.partial_taken = True
                return True, "部分止盈", self.entry_price - self.stop_distance() * CONFIG.partial_tp_r_multiple
        return False, "", 0

# ==================== 下单执行 ====================
def execute_order(symbol: str, direction: int, size: float, price: float, stop: float, take: float):
    dir_str = "多" if direction == 1 else "空"
    st.session_state.position = Position(
        direction=direction,
        entry_price=price,
        entry_time=datetime.now(),
        size=size,
        stop_loss=stop,
        take_profit=take,
        initial_atr=0,
        real=False
    )
    st.session_state.daily_trades += 1
    log_execution(f"开仓 {symbol} {dir_str} 仓位 {size:.4f} @ {price:.2f} 止损 {stop:.2f} 止盈 {take:.2f}")
    send_telegram(f"🔔 开仓 {dir_str} {symbol}\n价格: {price:.2f}\n仓位: {size:.4f}\n止损: {stop:.2f}\n止盈: {take:.2f}")

def close_position(symbol: str, exit_price: float, reason: str):
    pos = st.session_state.position
    if pos is None:
        return
    pnl = pos.pnl(exit_price)
    st.session_state.daily_pnl += pnl
    st.session_state.account_balance += pnl
    if st.session_state.account_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = st.session_state.account_balance
    st.session_state.net_value_history.append({'time': datetime.now(), 'value': st.session_state.account_balance})
    
    trade_record = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'direction': '多' if pos.direction == 1 else '空',
        'entry': pos.entry_price,
        'exit': exit_price,
        'size': pos.size,
        'pnl': pnl,
        'reason': reason
    }
    st.session_state.trade_log.append(trade_record)
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log.pop(0)
    
    append_to_csv(TRADE_LOG_FILE, trade_record)
    
    win = pnl > 0
    RiskManager().update_losses(win)
    log_execution(f"平仓 {symbol} {reason} 盈亏 {pnl:.2f} 余额 {st.session_state.account_balance:.2f}")
    send_telegram(f"🔔 平仓 {reason}\n盈亏: {pnl:.2f}\n余额: {st.session_state.account_balance:.2f}")
    st.session_state.position = None
    update_performance_metrics()

# ==================== 优化后的回测引擎 ====================
def run_backtest(data_dict: Dict[str, pd.DataFrame], initial_balance: float = 10000) -> Dict[str, Any]:
    """
    回测引擎（15分钟为基础周期），支持止损、止盈、ATR止损、自适应仓位。
    返回：
        - equity: pd.DataFrame，时间序列余额
        - trades: pd.DataFrame，交易记录
        - performance: dict，绩效指标
    """
    df_15m = data_dict['15m'].copy()
    balance = initial_balance
    peak_balance = initial_balance
    position = None
    equity_curve = []
    trades = []
    recent_returns = deque(maxlen=50)

    engine = SignalEngine()
    risk_manager = RiskManager()

    for i in range(50, len(df_15m)):
        row = df_15m.iloc[i]
        price = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr'] if not pd.isna(row['atr']) else 0
        dummy_dict = {tf: data_dict[tf].iloc[:i+1] for tf in data_dict}

        direction, prob = engine.calc_signal(dummy_dict)

        # 持仓管理
        if position is None and direction != 0 and prob >= SignalStrength.WEAK.value:
            stop_dist = atr * CONFIG.atr_multiplier_base if atr > 0 else price * 0.01
            stop = price - stop_dist if direction == 1 else price + stop_dist
            take = price + stop_dist * CONFIG.tp_min_ratio if direction == 1 else price - stop_dist * CONFIG.tp_min_ratio
            size = risk_manager.calc_position_size(balance, prob, atr, price, np.array(recent_returns))
            position = {
                'direction': direction,
                'entry': price,
                'size': size,
                'stop': stop,
                'take': take,
                'entry_time': row['timestamp'],
                'partial_taken': False
            }

        elif position is not None:
            close_flag = False
            exit_price = price
            reason = ""
            # 检查止损/止盈/部分止盈/超时
            hold_hours = (row['timestamp'] - position['entry_time']).total_seconds() / 3600

            if position['direction'] == 1:
                if low <= position['stop']:
                    close_flag, exit_price, reason = True, position['stop'], '止损'
                elif high >= position['take']:
                    close_flag, exit_price, reason = True, position['take'], '止盈'
                elif not position['partial_taken'] and high >= position['entry'] + (position['take'] - position['entry']) * CONFIG.partial_tp_r_multiple:
                    close_flag, exit_price, reason = True, position['entry'] + (position['take'] - position['entry']) * CONFIG.partial_tp_r_multiple, '部分止盈'
                    position['partial_taken'] = True
            else:
                if high >= position['stop']:
                    close_flag, exit_price, reason = True, position['stop'], '止损'
                elif low <= position['take']:
                    close_flag, exit_price, reason = True, position['take'], '止盈'
                elif not position['partial_taken'] and low <= position['entry'] - (position['entry'] - position['take']) * CONFIG.partial_tp_r_multiple:
                    close_flag, exit_price, reason = True, position['entry'] - (position['entry'] - position['take']) * CONFIG.partial_tp_r_multiple, '部分止盈'
                    position['partial_taken'] = True

            if hold_hours > CONFIG.max_hold_hours:
                close_flag, exit_price, reason = True, (high + low) / 2, '超时'

            if close_flag:
                pnl = (exit_price - position['entry']) * position['size'] * position['direction']
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': row['timestamp'],
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'reason': reason
                })
                recent_returns.append(pnl / max(1, balance))
                peak_balance = max(peak_balance, balance)
                position = None

        equity_curve.append({'time': row['timestamp'], 'balance': balance})

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['entry_time','exit_time','direction','entry','exit','size','pnl','reason'])

    # 简单绩效指标
    if not trades_df.empty:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        win_rate = len(wins)/len(trades_df)
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 1
        returns = trades_df['pnl'].values / initial_balance
        sharpe = (returns.mean()/returns.std()*np.sqrt(252)) if len(returns) > 1 and returns.std() != 0 else 0
        max_drawdown = (peak_balance - equity_df['balance'].min()) / peak_balance * 100
    else:
        win_rate = avg_win = avg_loss = sharpe = max_drawdown = 0

    performance = {
        'final_balance': balance,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'max_drawdown_pct': max_drawdown
    }

    return {
        'equity': equity_df,
        'trades': trades_df,
        'performance': performance
    }

# ==================== UI渲染器（含回测结果显示）====================
class UIRenderer:
    def __init__(self):
        self.fetcher = get_fetcher()

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ 配置")
            mode = st.radio("模式", ['实盘', '回测'], index=0)
            st.session_state.mode = 'live' if mode == '实盘' else 'backtest'

            symbol = st.selectbox("品种", CONFIG.symbols, index=CONFIG.symbols.index(st.session_state.current_symbol))
            st.session_state.current_symbol = symbol

            use_sim = st.checkbox("使用模拟数据（离线模式）", value=st.session_state.use_simulated_data)
            if use_sim != st.session_state.use_simulated_data:
                st.session_state.use_simulated_data = use_sim
                st.cache_data.clear()
                st.rerun()

            if st.session_state.use_simulated_data:
                st.info("📡 当前数据源：模拟数据")
            else:
                if st.session_state.data_source_failed:
                    st.error("📡 真实数据获取失败，已回退到模拟")
                else:
                    st.success("📡 当前数据源：币安实时数据")

            mode_lev = st.selectbox("杠杆模式", list(CONFIG.leverage_modes.keys()))
            st.session_state.leverage_mode = mode_lev

            st.number_input("余额 USDT", value=st.session_state.account_balance, disabled=True)

            if st.button("🔄 同步实盘余额"):
                if st.session_state.exchange and not st.session_state.use_simulated_data:
                    try:
                        bal = st.session_state.exchange.fetch_balance()
                        st.session_state.account_balance = float(bal['total'].get('USDT', 0))
                        st.success(f"同步成功: {st.session_state.account_balance:.2f} USDT")
                    except Exception as e:
                        st.error(f"同步失败: {e}")

            st.markdown("---")
            st.subheader("实盘")
            exchange_choice = st.selectbox("交易所", list(CONFIG.exchanges.keys()), key='exchange_choice')
            api_key = st.text_input("API Key", value=st.session_state.binance_api_key, type="password")
            secret_key = st.text_input("Secret Key", value=st.session_state.binance_secret_key, type="password")
            passphrase = st.text_input("Passphrase (仅OKX需要)", type="password") if "OKX" in exchange_choice else None
            testnet = st.checkbox("测试网", value=st.session_state.testnet)
            use_real = st.checkbox("实盘交易", value=st.session_state.use_real)

            if st.button("🔌 测试连接"):
                try:
                    ex_class = CONFIG.exchanges[exchange_choice]
                    params = {
                        'apiKey': api_key,
                        'secret': secret_key,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    }
                    if passphrase:
                        params['password'] = passphrase
                    ex = ex_class(params)
                    if testnet:
                        ex.set_sandbox_mode(True)
                    ticker = ex.fetch_ticker(symbol)
                    st.success(f"连接成功！{symbol} 价格: {ticker['last']}")
                    st.session_state.exchange = ex
                    st.session_state.binance_api_key = api_key
                    st.session_state.binance_secret_key = secret_key
                    st.session_state.testnet = testnet
                    st.session_state.use_real = use_real
                except Exception as e:
                    st.error(f"连接失败: {e}")

            st.session_state.auto_enabled = st.checkbox("自动交易", value=True)

            with st.expander("📱 Telegram通知"):
                token = st.text_input("Bot Token", type="password")
                chat_id = st.text_input("Chat ID")
                if token and chat_id:
                    st.session_state.telegram_token = token
                    st.session_state.telegram_chat_id = chat_id

            if st.button("🚨 一键紧急平仓"):
                if st.session_state.position:
                    close_position(st.session_state.current_symbol,
                                   st.session_state.multi_df['15m']['close'].iloc[-1],
                                   "紧急平仓")
                st.rerun()

            if st.button("🖐️ 手动开仓测试"):
                if 'multi_df' in st.session_state and st.session_state.multi_df:
                    df = st.session_state.multi_df['15m']
                    price = df['close'].iloc[-1]
                    atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 0
                    if atr == 0:
                        stop_dist = price * 0.01
                    else:
                        stop_dist = atr * CONFIG.atr_multiplier_base
                    stop = price - stop_dist
                    take = price + stop_dist * CONFIG.tp_min_ratio
                    recent_returns = df['close'].pct_change().dropna().values[-20:]
                    size = RiskManager().calc_position_size(st.session_state.account_balance, 0.7, atr, price, recent_returns)
                    if size > 0:
                        execute_order(symbol, 1, size, price, stop, take)
                        st.rerun()

            if st.button("📂 查看历史交易记录"):
                if os.path.exists(TRADE_LOG_FILE):
                    df_trades = pd.read_csv(TRADE_LOG_FILE)
                    st.dataframe(df_trades.tail(20))
                else:
                    st.info("暂无历史交易记录")

            if st.session_state.error_log:
                with st.expander("⚠️ 错误日志（实时）"):
                    for err in list(st.session_state.error_log)[-10:]:
                        st.text(err)

            if st.session_state.execution_log:
                with st.expander("📋 执行日志（实时）"):
                    for log in list(st.session_state.execution_log)[-10:]:
                        st.text(log)

            if st.button("🗑️ 重置所有状态"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        return symbol, mode_lev, use_real

    def render_main_panel(self, symbol, mode, use_real, data, engine, risk):
        if not data:
            st.error("❌ 数据获取失败")
            return
        df_dict = data['data_dict']
        st.session_state.multi_df = df_dict
        st.session_state.fear_greed = data['fear_greed']
        st.session_state.market_regime = engine.detect_market_regime(df_dict)

        df_15m = df_dict['15m']
        current_price = data['current_price']
        atr = df_15m['atr'].iloc[-1] if not pd.isna(df_15m['atr'].iloc[-1]) else 0
        atr_pct = (atr / current_price * 100) if atr > 0 else 0
        st.session_state.circuit_breaker = risk.check_circuit_breaker(atr_pct, data['fear_greed'])

        if st.session_state.mode == 'backtest':
            if st.button("▶️ 运行回测"):
                with st.spinner("回测中..."):
                    results = run_backtest(df_dict, st.session_state.account_balance)
                    st.session_state.backtest_results = results
            if st.session_state.backtest_results:
                eq = st.session_state.backtest_results['equity']
                trades = st.session_state.backtest_results['trades']
                perf = st.session_state.backtest_results['performance']
                st.subheader("回测结果")
                col1, col2, col3 = st.columns(3)
                col1.metric("最终余额", f"{perf['final_balance']:.2f}")
                col2.metric("胜率", f"{perf['win_rate']:.2%}")
                col3.metric("夏普比率", f"{perf['sharpe']:.2f}")
                col1.metric("平均盈利", f"{perf['avg_win']:.2f}")
                col2.metric("平均亏损", f"{perf['avg_loss']:.2f}")
                col3.metric("最大回撤", f"{perf['max_drawdown_pct']:.2f}%")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eq['time'], y=eq['balance'], mode='lines', name='净值'))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                if not trades.empty:
                    st.dataframe(trades.tail(10))
            return

        # 实盘模式
        direction, prob = engine.calc_signal(df_dict)
        recent_returns = df_15m['close'].pct_change().dropna().values[-20:]
        size = risk.calc_position_size(st.session_state.account_balance, prob, atr, current_price, recent_returns)

        with st.expander("🔍 开仓调试信息", expanded=True):
            st.write(f"方向: {direction}, 概率: {prob:.2%}")
            st.write(f"ATR: {atr:.2f}, ATR%: {atr_pct:.2f}%, 计算仓位: {size:.4f}")
            st.write(f"信号阈值: {SignalStrength.WEAK.value:.2%}")
            st.write(f"市场状态: {st.session_state.market_regime.value}")
            st.write(f"恐惧贪婪: {data['fear_greed']}")
            st.write(f"风控状态: 熔断={st.session_state.circuit_breaker}, 冷却={risk.check_cooldown()}, 日内限制={risk.check_daily_limit()}, 超回撤={risk.check_max_drawdown()}")
            st.write(f"是否满足开仓条件: {direction != 0 and prob >= SignalStrength.WEAK.value and size > 0}")

        if not (st.session_state.circuit_breaker or risk.check_cooldown() or risk.check_daily_limit() or risk.check_max_drawdown()):
            if st.session_state.position:
                pos = st.session_state.position
                high = df_15m['high'].iloc[-1]
                low = df_15m['low'].iloc[-1]
                should_close, reason, exit_price = pos.should_close(high, low, datetime.now())
                if should_close:
                    close_position(symbol, exit_price, reason)
                else:
                    if not pd.isna(atr) and atr > 0:
                        pos.update_stops(current_price, atr)
            else:
                if direction != 0 and prob >= SignalStrength.WEAK.value and size > 0:
                    if st.session_state.last_signal_time and (datetime.now() - st.session_state.last_signal_time).total_seconds() < CONFIG.anti_duplicate_seconds:
                        st.write("⏳ 防重机制阻止开仓")
                    else:
                        if atr == 0 or np.isnan(atr):
                            stop_dist = current_price * 0.01
                        else:
                            stop_dist = atr * adaptive_atr_multiplier(df_15m['close'])
                        stop = current_price - stop_dist if direction == 1 else current_price + stop_dist
                        take = current_price + stop_dist * CONFIG.tp_min_ratio if direction == 1 else current_price - stop_dist * CONFIG.tp_min_ratio
                        execute_order(symbol, direction, size, current_price, stop, take)
                        st.session_state.last_signal_time = datetime.now()
                        st.rerun()

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("### 📊 市场状态")
            c1, c2, c3 = st.columns(3)
            c1.metric("恐惧贪婪", data['fear_greed'])
            c2.metric("信号概率", f"{prob:.1%}")
            c3.metric("当前价格", f"{current_price:.2f}")

            if st.session_state.position:
                pos = st.session_state.position
                pnl = pos.pnl(current_price)
                st.markdown(f"### 持仓 {'多' if pos.direction==1 else '空'}")
                st.info(f"入场 {pos.entry_price:.2f} | 数量 {pos.size:.4f}")
                st.info(f"止损 {pos.stop_loss:.2f} | 止盈 {pos.take_profit:.2f}")
                st.metric("浮动盈亏", f"{pnl:.2f} USDT", delta=f"{(pnl/pos.size):.2f}")
            else:
                st.markdown("### 无持仓")
                st.info("等待信号...")

            with st.expander("🔍 多周期信号详情"):
                for tf, df in df_dict.items():
                    last = df.iloc[-1]
                    st.write(f"{tf}: 价格 {last['close']:.2f}, EMA20 {last['ema20']:.2f}, RSI {last['rsi']:.1f}, ADX {last['adx']:.1f}")

            st.markdown("### 📉 风险监控")
            st.metric("实时盈亏", f"{st.session_state.daily_pnl:.2f} USDT")
            drawdown = (st.session_state.peak_balance - st.session_state.account_balance) / st.session_state.peak_balance * 100
            st.metric("最大回撤", f"{drawdown:.2f}%")
            st.metric("连亏次数", st.session_state.consecutive_losses)
            st.metric("日内交易", f"{st.session_state.daily_trades}/{CONFIG.max_daily_trades}")

            if st.session_state.cooldown_until:
                st.warning(f"冷却至 {st.session_state.cooldown_until.strftime('%H:%M')}")

            perf = st.session_state.performance_metrics
            st.markdown("### 📈 绩效指标 (近50笔)")
            st.metric("胜率", f"{perf['win_rate']:.2%}")
            st.metric("平均盈利", f"{perf['avg_win']:.2f}")
            st.metric("平均亏损", f"{perf['avg_loss']:.2f}")
            st.metric("夏普比率", f"{perf['sharpe']:.2f}")

            if st.session_state.net_value_history:
                hist_df = pd.DataFrame(st.session_state.net_value_history[-200:])
                fig_nv = go.Figure()
                fig_nv.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['value'], mode='lines', name='净值', line=dict(color='cyan')))
                fig_nv.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), template='plotly_dark')
                st.plotly_chart(fig_nv, use_container_width=True)

        with col2:
            df_plot = df_15m.tail(120)
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5,0.15,0.15,0.2], vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(x=df_plot['timestamp'], open=df_plot['open'], high=df_plot['high'],
                                          low=df_plot['low'], close=df_plot['close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema20'], line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema50'], line=dict(color="blue")), row=1, col=1)
            if st.session_state.position:
                pos = st.session_state.position
                fig.add_hline(y=pos.entry_price, line_dash="dot", line_color="yellow", annotation_text=f"入场 {pos.entry_price:.2f}")
                fig.add_hline(y=pos.stop_loss, line_dash="dash", line_color="red", annotation_text=f"止损 {pos.stop_loss:.2f}")
                fig.add_hline(y=pos.take_profit, line_dash="dash", line_color="green", annotation_text=f"止盈 {pos.take_profit:.2f}")
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['rsi'], line=dict(color="purple")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['macd'], line=dict(color="cyan")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['macd_signal'], line=dict(color="orange")), row=3, col=1)
            fig.add_bar(x=df_plot['timestamp'], y=df_plot['macd_diff'], marker_color="gray", row=3, col=1)
            colors_vol = np.where(df_plot['close'] >= df_plot['open'], 'green', 'red')
            fig.add_trace(go.Bar(x=df_plot['timestamp'], y=df_plot['volume'], marker_color=colors_vol), row=4, col=1)
            fig.update_layout(height=800, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            if st.button("运行 Monte Carlo 风险模拟"):
                sim_df = self._monte_carlo_sim(df_15m['close'])
                if not sim_df.empty:
                    fig_mc = go.Figure()
                    for i in range(min(30, sim_df.shape[1])):
                        fig_mc.add_trace(go.Scatter(y=sim_df.iloc[:, i], mode='lines', line=dict(color='rgba(0,200,0,0.1)'), showlegend=False))
                    fig_mc.add_trace(go.Scatter(y=sim_df.mean(axis=1), mode='lines', line=dict(color='red', width=2), name='均值'))
                    fig_mc.update_layout(height=300, template='plotly_dark')
                    st.plotly_chart(fig_mc, use_container_width=True)

    def _monte_carlo_sim(self, price_series: pd.Series, n_sim: int = 500) -> pd.DataFrame:
        returns = price_series.pct_change().dropna().values
        if len(returns) == 0:
            return pd.DataFrame()
        last_price = price_series.iloc[-1]
        sim = np.zeros((n_sim, min(200, len(price_series))))
        for i in range(n_sim):
            sim[i, 0] = last_price
            for t in range(1, sim.shape[1]):
                sim[i, t] = sim[i, t-1] * (1 + np.random.choice(returns))
        return pd.DataFrame(sim.T)

# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="终极量化终端 36.2", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 超神终极版 36.2")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北 · 多源并行 · 在线学习 · 自适应风控 · 优化回测")

    init_session_state()
    renderer = UIRenderer()
    symbol, mode, use_real = renderer.render_sidebar()

    data = renderer.fetcher.get_symbol_data(symbol)
    if not data:
        st.error("❌ 数据获取失败，请检查网络或API配置")
        st.stop()

    if st.session_state.use_simulated_data:
        st.warning("⚠️ 当前使用模拟数据（币安获取失败自动回退）")
    else:
        st.success("🟢 实时币安数据同步成功 · K线图完全真实")

    engine = SignalEngine()
    risk = RiskManager()

    renderer.render_main_panel(symbol, mode, use_real, data, engine, risk)

    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

if __name__ == "__main__":
    main()
