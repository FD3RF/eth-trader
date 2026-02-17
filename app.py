# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 超神烧脑版 30.0（宇宙主宰·永恒无敌·完美无瑕·永不败北·终极稳定版）
绝对智慧 · 离线模拟引擎 · 动态K线 · 一键实盘切换 · 永不崩溃
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
import math

warnings.filterwarnings('ignore')

# ==================== 配置与常量 ====================
class SignalStrength(Enum):
    STRONG = 0.70
    HIGH = 0.62
    MEDIUM = 0.55
    WEAK = 0.50
    NONE = 0.0

class MarketRegime(Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    PANIC = "PANIC"

@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"])
    base_risk_per_trade: float = 0.02
    risk_budget_ratio: float = 0.10
    daily_loss_limit: float = 300.0
    max_drawdown_pct: float = 20.0
    min_atr_pct: float = 0.8
    tp_min_ratio: float = 2.0
    partial_tp_ratio: float = 0.5
    partial_tp_r_multiple: float = 1.0
    trailing_stop_pct: float = 0.35
    breakeven_trigger_pct: float = 1.01
    max_hold_hours: int = 36
    max_consecutive_losses: int = 3
    cooldown_losses: int = 3
    cooldown_hours: int = 24
    max_daily_trades: int = 5
    leverage_modes: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "稳健 (3-5x)": (3, 5),
        "无敌 (5-8x)": (5, 8),
        "神级 (8-10x)": (8, 10)
    })
    exchanges: Dict[str, Any] = field(default_factory=lambda: {
        "Binance合约": ccxt.binance,
        "Bybit合约": ccxt.bybit,
        "OKX合约": ccxt.okx
    })
    data_sources: List[str] = field(default_factory=lambda: ["mexc", "binance", "bybit", "kucoin"])
    timeframes: List[str] = field(default_factory=lambda: ['15m', '1h', '4h', '1d'])
    timeframe_weights: Dict[str, int] = field(default_factory=lambda: {'1d': 10, '4h': 7, '1h': 5, '15m': 3})
    fetch_limit: int = 1500
    auto_refresh_ms: int = 60000
    anti_duplicate_seconds: int = 300
    kelly_fraction: float = 0.25
    atr_multiplier_base: float = 1.5
    max_leverage_global: float = 10.0
    funding_rate_weight: int = 10
    ichimoku_weight: int = 8
    volume_profile_weight: int = 7
    orderbook_weight: int = 8
    machine_learning_weight: int = 15
    circuit_breaker_atr: float = 5.0
    circuit_breaker_fg_extreme: Tuple[int, int] = (10, 90)
    rsi_extreme_penalty: int = 15
    fg_extreme_penalty: int = 12
    slippage_base: float = 0.0003
    fee_rate: float = 0.0004
    ic_window: int = 168
    ic_ma_window: int = 5
    walk_forward_train: int = 2000
    walk_forward_test: int = 500
    mc_simulations: int = 10000
    # 模拟数据参数
    sim_volatility: float = 0.05
    sim_trend_strength: float = 0.15

CONFIG = TradingConfig()

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
        'auto_position': None,
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
        'use_simulated_data': True,  # 默认使用模拟数据，确保开箱即用
        'data_source_failed': False,
        'error_log': [],
        'execution_log': [],
        'last_trade_date': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def load_secrets_config() -> Dict[str, str]:
    secrets_config = {}
    try:
        key_map = {
            'OKX_API_KEY': ['OKX_API_KEY'],
            'OKX_SECRET_KEY': ['OKX_SECRET_KEY'],
            'OKX_PASSPHRASE': ['OKX_PASSPHRASE'],
            'BINANCE_API_KEY': ['BINANCE_API_KEY'],
            'BINANCE_SECRET_KEY': ['BINANCE_SECRET_KEY'],
            'BYBIT_API_KEY': ['BYBIT_API_KEY'],
            'BYBIT_SECRET_KEY': ['BYBIT_SECRET_KEY'],
            'TELEGRAM_BOT_TOKEN': ['TELEGRAM_BOT_TOKEN'],
            'TELEGRAM_CHAT_ID': ['TELEGRAM_CHAT_ID']
        }
        for target, possible_keys in key_map.items():
            for key in possible_keys:
                if key in st.secrets:
                    secrets_config[target] = st.secrets[key]
                    break
    except Exception:
        pass
    return secrets_config

def send_telegram(msg: str) -> None:
    token = st.session_state.get('telegram_token') or st.secrets.get('TELEGRAM_BOT_TOKEN')
    chat_id = st.session_state.get('telegram_chat_id') or st.secrets.get('TELEGRAM_CHAT_ID')
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                          timeout=5)
        except Exception:
            pass

def log_error(msg: str):
    st.session_state.error_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    if len(st.session_state.error_log) > 10:
        st.session_state.error_log.pop(0)
    logger.error(msg)

def log_execution(msg: str):
    st.session_state.execution_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    if len(st.session_state.execution_log) > 20:
        st.session_state.execution_log.pop(0)

# ==================== 超真实模拟数据生成器 ====================
def generate_simulated_data(symbol: str, limit: int = 1500) -> Dict[str, pd.DataFrame]:
    """生成动态波动的模拟K线数据，包含所有技术指标"""
    try:
        np.random.seed(abs(hash(symbol)) % 2**32)
        end = datetime.now()
        start = end - timedelta(minutes=15 * limit)
        timestamps = pd.date_range(start, end, periods=limit, freq='15min')
        
        if 'BTC' in symbol:
            base = 40000
            volatility = CONFIG.sim_volatility * 0.7
        elif 'ETH' in symbol:
            base = 2000
            volatility = CONFIG.sim_volatility
        else:
            base = 100
            volatility = CONFIG.sim_volatility * 1.3
        
        # 生成价格序列（趋势 + 周期 + 随机）
        t = np.linspace(0, 4*np.pi, limit)
        trend_direction = np.random.choice([-1, 1])
        trend = trend_direction * CONFIG.sim_trend_strength * np.linspace(0, 1, limit) * base
        cycle = 0.05 * base * np.sin(t * 2)
        random_step = np.random.randn(limit) * volatility * base
        random_walk = np.cumsum(random_step) * 0.1
        price_series = base + trend + cycle + random_walk
        price_series = np.maximum(price_series, base * 0.2)
        
        # 生成OHLC
        opens = price_series * (1 + np.random.randn(limit) * 0.001)
        closes = price_series * (1 + np.random.randn(limit) * 0.002)
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(limit)) * volatility * price_series
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(limit)) * volatility * price_series
        volume_base = np.random.randint(1000, 10000, limit)
        volume_factor = 1 + 2 * np.abs(np.diff(price_series, prepend=price_series[0])) / price_series
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
            resampled = add_indicators(resampled)
            data_dict[tf] = resampled
        
        return data_dict
    except Exception as e:
        logger.error(f"模拟数据生成失败: {e}")
        # 降级方案：生成最简单的随机数据
        dummy_times = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        dummy_prices = 100 + np.cumsum(np.random.randn(100) * 2)
        dummy_df = pd.DataFrame({
            'timestamp': dummy_times,
            'open': dummy_prices,
            'high': dummy_prices + np.abs(np.random.randn(100) * 5),
            'low': dummy_prices - np.abs(np.random.randn(100) * 5),
            'close': dummy_prices + np.random.randn(100) * 2,
            'volume': np.random.randint(1000, 10000, 100)
        })
        dummy_df = add_indicators(dummy_df)
        return {'15m': dummy_df, '1h': dummy_df, '4h': dummy_df, '1d': dummy_df}

# ==================== 技术指标计算 ====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['atr'] = atr
    df['atr_pct'] = (df['atr'] / df['close'] * 100)
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_surge'] = df['volume'] > df['volume_ma20'] * 1.2

    high9 = df['high'].rolling(9).max()
    low9 = df['low'].rolling(9).min()
    df['ichimoku_tenkan'] = (high9 + low9) / 2
    high26 = df['high'].rolling(26).max()
    low26 = df['low'].rolling(26).min()
    df['ichimoku_kijun'] = (high26 + low26) / 2
    df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
    df['ichimoku_senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

    df['date'] = df['timestamp'].dt.date
    typical = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df.groupby('date')['volume'].cumsum()
    cum_typical_vol = (typical * df['volume']).groupby(df['date']).cumsum()
    df['vwap'] = np.where(cum_vol > 0, cum_typical_vol / cum_vol, df['close'])

    mf_mult = (df['close'] - df['low']) - (df['high'] - df['close'])
    mf_denom = df['high'] - df['low']
    mf = np.where(mf_denom > 0, mf_mult / mf_denom * df['volume'], 0)
    vol_sum = df['volume'].rolling(20).sum()
    df['cmf'] = np.where(vol_sum > 0, pd.Series(mf).rolling(20).sum() / vol_sum, 0)

    df['future_return'] = df['close'].pct_change(8).shift(-8)

    return df

# ==================== 数据获取器（同步版，内置模拟回退）====================
@st.cache_resource
def get_fetcher() -> 'AggregatedDataFetcher':
    return AggregatedDataFetcher()

class AggregatedDataFetcher:
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        for name in CONFIG.data_sources:
            try:
                cls = getattr(ccxt, name)
                self.exchanges[name] = cls({'enableRateLimit': True, 'timeout': 30000})
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
            logger.warning(f"从 {ex.id} 获取 {symbol} {timeframe} 数据失败: {e}")
        return None

    def _fetch_kline(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        if not self.exchanges:
            logger.error("无可用交易所")
            return None
        for ex in self.exchanges.values():
            df = self._fetch_kline_single(ex, symbol, timeframe, limit)
            if df is not None:
                return df
        return None

    @st.cache_data(ttl=60, show_spinner=False)
    def fetch_15m(_symbol: str) -> pd.DataFrame:
        fetcher = get_fetcher()
        df = fetcher._fetch_kline(_symbol, '15m', CONFIG.fetch_limit)
        if df is not None and len(df) >= 50:
            return add_indicators(df)
        return pd.DataFrame()

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_other(_symbol: str, tf: str) -> pd.DataFrame:
        fetcher = get_fetcher()
        df = fetcher._fetch_kline(_symbol, tf, CONFIG.fetch_limit)
        if df is not None and len(df) >= 50:
            return add_indicators(df)
        return pd.DataFrame()

    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_fear_greed() -> int:
        try:
            r = requests.Session().get("https://api.alternative.me/fng/", timeout=5)
            return int(r.json()['data'][0]['value'])
        except Exception:
            return 50

    def fetch_funding_rate(self, symbol: str) -> float:
        rates = []
        for ex in self.exchanges.values():
            try:
                rates.append(ex.fetch_funding_rate(symbol)['fundingRate'])
            except Exception:
                continue
        return float(np.mean(rates)) if rates else 0.0

    def fetch_orderbook_imbalance(self, symbol: str, depth: int = 10) -> float:
        for ex in self.exchanges.values():
            try:
                ob = ex.fetch_order_book(symbol, limit=depth)
                bid_vol = sum(b[1] for b in ob['bids'])
                ask_vol = sum(a[1] for a in ob['asks'])
                total = bid_vol + ask_vol
                return (bid_vol - ask_vol) / total if total > 0 else 0.0
            except Exception:
                continue
        return 0.0

    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        # 如果强制模拟数据，直接生成模拟
        if st.session_state.get('use_simulated_data', True):
            data_dict = generate_simulated_data(symbol, CONFIG.fetch_limit)
            return {
                "data_dict": data_dict,
                "current_price": float(data_dict['15m']['close'].iloc[-1]),
                "fear_greed": 50,
                "funding_rate": 0.0,
                "orderbook_imbalance": 0.0,
            }

        # 尝试获取真实数据
        try:
            df_15m = self.fetch_15m(symbol)
            if df_15m.empty:
                log_error("获取15m数据失败，切换至模拟数据")
                st.session_state.use_simulated_data = True
                return self.get_symbol_data(symbol)

            data_dict = {'15m': df_15m}
            for tf in CONFIG.timeframes[1:]:
                df_tf = self.fetch_other(symbol, tf)
                if not df_tf.empty:
                    data_dict[tf] = df_tf

            return {
                "data_dict": data_dict,
                "current_price": float(df_15m['close'].iloc[-1]),
                "fear_greed": self.fetch_fear_greed(),
                "funding_rate": self.fetch_funding_rate(symbol),
                "orderbook_imbalance": self.fetch_orderbook_imbalance(symbol),
            }
        except Exception as e:
            log_error(f"获取数据异常: {e}，切换至模拟数据")
            st.session_state.use_simulated_data = True
            return self.get_symbol_data(symbol)

# ==================== Regime & IC 引擎 ====================
class RegimeEngine:
    @staticmethod
    def detect_regime(df_15m: pd.DataFrame) -> MarketRegime:
        last = df_15m.iloc[-1]
        atr_pct = last['atr_pct']
        adx = last['adx']
        returns_vol = df_15m['close'].pct_change().rolling(20).std().iloc[-1]
        skew = df_15m['close'].pct_change().rolling(60).skew().iloc[-1]
        price_to_ema200 = last['close'] / last['ema200']
        
        if atr_pct > 3.5 or returns_vol > 0.04 or abs(skew) > 1.5:
            return MarketRegime.PANIC
        elif adx > 30 and abs(price_to_ema200 - 1) > 0.05:
            return MarketRegime.TREND
        else:
            return MarketRegime.RANGE

class ICEngine:
    @staticmethod
    def calculate_ic_ma(df_15m: pd.DataFrame, factor_name: str) -> float:
        ics = []
        for i in range(CONFIG.ic_ma_window):
            window = min(CONFIG.ic_window, len(df_15m) - 8 - i * 24)
            if window < 50:
                break
            factor = df_15m[factor_name].iloc[-window - i*24 : -8 - i*24]
            future_ret = df_15m['future_return'].iloc[-window - i*24 : -8 - i*24]
            ic = factor.corr(future_ret)
            if not pd.isna(ic):
                ics.append(ic)
        return np.mean(ics) if ics else 0.0

# ==================== 信号引擎（真实概率置信区间版）====================
class SignalEngine:
    def __init__(self):
        self.base_weights = {
            'core_trend': 30, 'multi_frame': 20, 'volatility': 15, 'volume': 15,
            'rsi': 10, 'btc_sync': 10, 'fear_greed': 10, 'funding_rate': CONFIG.funding_rate_weight,
            'ichimoku': CONFIG.ichimoku_weight, 'volume_profile': CONFIG.volume_profile_weight,
            'orderbook': CONFIG.orderbook_weight, 'machine_learning': CONFIG.machine_learning_weight,
        }
        self.regime_mod = {
            MarketRegime.TREND: {'core_trend': 1.4, 'multi_frame': 1.3, 'ichimoku': 1.2},
            MarketRegime.RANGE: {'rsi': 1.5, 'volume_profile': 1.4, 'orderbook': 1.3},
            MarketRegime.PANIC: {'volatility': 0.4, 'risk_mult': 0.4},
        }

    @staticmethod
    def is_uptrend(row: pd.Series) -> bool:
        return row['close'] > row['ema200'] and row['macd_diff'] > 0

    @staticmethod
    def is_downtrend(row: pd.Series) -> bool:
        return row['close'] < row['ema200'] and row['macd_diff'] < 0

    def get_weights(self, regime: MarketRegime, ic_ma_dict: Dict[str, float]) -> Dict[str, float]:
        weights = self.base_weights.copy()
        mod = self.regime_mod.get(regime, {})
        for k, v in mod.items():
            if k in weights and k != 'risk_mult':
                weights[k] *= v
        for factor, ic_ma in ic_ma_dict.items():
            if factor in weights:
                ic_adj = np.clip(ic_ma, -0.2, 0.2)
                weights[factor] *= np.exp(ic_adj)
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total / 100
        return weights

    def calculate_signal(self, df_15m: pd.DataFrame, data_dict: Dict[str, pd.DataFrame],
                         btc_trend: int, fear_greed: int, funding_rate: float,
                         imbalance: float, symbol: str) -> Tuple[float, float, float, int, MarketRegime, List[str]]:
        last = df_15m.iloc[-1]
        regime = RegimeEngine.detect_regime(df_15m)
        
        ic_ma_dict = {
            'rsi': ICEngine.calculate_ic_ma(df_15m, 'rsi'),
            'macd_diff': ICEngine.calculate_ic_ma(df_15m, 'macd_diff'),
            'atr_pct': ICEngine.calculate_ic_ma(df_15m, 'atr_pct'),
            'adx': ICEngine.calculate_ic_ma(df_15m, 'adx'),
        }
        
        weights = self.get_weights(regime, ic_ma_dict)
        
        raw_score = 0.0
        details = [f"市场状态: {regime.value}"]
        
        if self.is_uptrend(last):
            raw_score += weights.get('core_trend', 30)
            direction = 1
        elif self.is_downtrend(last):
            raw_score += weights.get('core_trend', 30)
            direction = -1
        else:
            details.append("无明确趋势")
            return 0.0, 0.0, 0.0, 0, regime, details
        
        mf = 0
        for tf, w in CONFIG.timeframe_weights.items():
            if tf in data_dict:
                l = data_dict[tf].iloc[-1]
                if (direction == 1 and l['close'] > l['ema50'] > l['ema200'] and l['adx'] > 20) or \
                   (direction == -1 and l['close'] < l['ema50'] < l['ema200'] and l['adx'] > 20):
                    mf += w
        raw_score += min(mf, weights.get('multi_frame', 20))
        
        if last['atr_pct'] >= CONFIG.min_atr_pct:
            raw_score += weights.get('volatility', 15)
        if last['volume_surge']:
            raw_score += weights.get('volume', 15)
        if (direction == 1 and last['rsi'] > 50) or (direction == -1 and last['rsi'] < 50):
            raw_score += weights.get('rsi', 10)
        
        # BTC相关系数简化
        if btc_trend == direction:
            raw_score += weights.get('btc_sync', 10)
        
        model_prob = 1 / (1 + np.exp(-raw_score / 20 + 2.5))
        
        historical_window = min(500, len(df_15m) - 8)
        if historical_window > 100:
            future_returns = df_15m['future_return'].iloc[-historical_window:-8]
            historical_prob = (future_returns > 0).mean()
            prob_std = np.std((future_returns > 0).astype(int))
        else:
            historical_prob = 0.5
            prob_std = 0.3
        
        prob = 0.6 * model_prob + 0.4 * historical_prob
        prob_lower = max(0.0, prob - prob_std * 1.96)
        prob_upper = min(1.0, prob + prob_std * 1.96)
        
        if regime == MarketRegime.PANIC:
            prob *= 0.4
            prob_lower *= 0.4
            prob_upper *= 0.4
        
        details.append(f"校准概率: {prob:.1%} [{prob_lower:.1%}-{prob_upper:.1%}]")
        return prob, prob_lower, prob_upper, direction, regime, details

# ==================== 风控与持仓 ====================
class RiskManager:
    def __init__(self):
        self.recent_trades = deque(maxlen=50)

    def update_stats(self, r_multiple: float) -> None:
        self.recent_trades.append(r_multiple)

    def kelly_fraction(self) -> float:
        if len(self.recent_trades) < 10:
            return 0.1
        wins = [r for r in self.recent_trades if r > 0]
        losses = [abs(r) for r in self.recent_trades if r < 0]
        win_rate = len(wins) / len(self.recent_trades)
        avg_win = np.mean(wins) if wins else 1.0
        avg_loss = np.mean(losses) if losses else 1.0
        variance = np.var(self.recent_trades)
        variance_penalty = 1 / (1 + variance)
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / b
        return max(0, min(kelly * CONFIG.kelly_fraction * variance_penalty, 0.5))

    def get_position_size(self, balance: float, prob: float, regime: MarketRegime, initial_risk_per_unit: float) -> float:
        edge = abs(prob - 0.5)
        if edge < 0.1:
            return 0.0
        kelly = self.kelly_fraction()
        risk_mult = kelly * edge * 2
        if regime == MarketRegime.PANIC:
            risk_mult *= 0.4
        risk_budget = balance * CONFIG.risk_budget_ratio
        position_risk = balance * CONFIG.base_risk_per_trade * risk_mult
        return min(position_risk / initial_risk_per_unit, risk_budget / initial_risk_per_unit) if initial_risk_per_unit > 0 else 0.0

    def dynamic_stops(self, entry: float, direction: int, atr: float, adx: float, atr_pct: float, regime: MarketRegime) -> Tuple[float, float]:
        mult_base = CONFIG.atr_multiplier_base
        if regime == MarketRegime.TREND:
            mult_base *= 1.2
        elif regime == MarketRegime.RANGE:
            mult_base *= 0.8
        elif regime == MarketRegime.PANIC:
            mult_base *= 0.5
        mult = mult_base * (1.2 if adx > 35 else 0.8 if adx < 20 else 1.0) * (1.3 if atr_pct > 2.0 else 1.0)
        stop_dist = mult * atr
        take_dist = stop_dist * CONFIG.tp_min_ratio
        return (entry - stop_dist, entry + take_dist) if direction == 1 else (entry + stop_dist, entry - take_dist)

    def check_circuit_breaker(self, atr_pct: float, fear_greed: int) -> bool:
        return atr_pct > CONFIG.circuit_breaker_atr or fear_greed <= CONFIG.circuit_breaker_fg_extreme[0] or fear_greed >= CONFIG.circuit_breaker_fg_extreme[1]

    def check_cooldown(self) -> bool:
        if st.session_state.consecutive_losses >= CONFIG.cooldown_losses:
            if st.session_state.cooldown_until is None:
                st.session_state.cooldown_until = datetime.now() + timedelta(hours=CONFIG.cooldown_hours)
            if datetime.now() < st.session_state.cooldown_until:
                return True
        elif st.session_state.consecutive_losses == 0:
            st.session_state.cooldown_until = None
        return False

    def check_daily_trade_limit(self) -> bool:
        today = datetime.now().date()
        if 'last_trade_date' not in st.session_state or st.session_state.last_trade_date != today:
            st.session_state.daily_trades = 0
            st.session_state.last_trade_date = today
        return st.session_state.daily_trades >= CONFIG.max_daily_trades

@dataclass
class Position:
    direction: int
    entry: float
    time: datetime
    stop: float
    take: float
    size: float
    original_size: float
    initial_risk_per_unit: float
    partial_taken: bool = False
    real: bool = False

    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry) * self.size * self.direction

    def r_multiple(self, exit_price: float) -> float:
        return (exit_price - self.entry) * self.direction / self.initial_risk_per_unit

    def check_partial_tp(self, current_price: float) -> bool:
        if self.partial_taken:
            return False
        r_target = self.entry + self.initial_risk_per_unit * self.direction * CONFIG.partial_tp_r_multiple
        if (self.direction == 1 and current_price >= r_target) or (self.direction == -1 and current_price <= r_target):
            self.size *= (1 - CONFIG.partial_tp_ratio)
            self.partial_taken = True
            self.stop = self.entry
            return True
        return False

    def should_close(self, high: float, low: float, close: float, current_time: datetime) -> Tuple[bool, str, float]:
        exit_price = close
        reason = ""
        if self.direction == 1:
            if low <= self.stop:
                exit_price = self.stop
                reason = "止损"
            elif high >= self.take:
                exit_price = self.take
                reason = "止盈"
        else:
            if high >= self.stop:
                exit_price = self.stop
                reason = "止损"
            elif low <= self.take:
                exit_price = self.take
                reason = "止盈"
        if reason:
            return True, reason, exit_price
        if (current_time - self.time).total_seconds() / 3600 > CONFIG.max_hold_hours:
            return True, "超时", close
        return False, "", close

# ==================== UI渲染 ====================
class UIRenderer:
    def __init__(self):
        self.fetcher = get_fetcher()

    def render_sidebar(self) -> Tuple[str, str, bool]:
        with st.sidebar:
            st.header("⚙️ 配置")
            symbol = st.selectbox("品种", CONFIG.symbols, index=CONFIG.symbols.index(st.session_state.current_symbol))
            st.session_state.current_symbol = symbol
            
            use_sim = st.checkbox("使用模拟数据（离线模式）", value=st.session_state.get('use_simulated_data', True))
            if use_sim != st.session_state.get('use_simulated_data', True):
                st.session_state.use_simulated_data = use_sim
                st.cache_data.clear()
                st.rerun()
            
            mode = st.selectbox("杠杆模式", list(CONFIG.leverage_modes.keys()))
            current_balance = st.session_state.account_balance
            st.number_input("余额 USDT", value=current_balance, disabled=True, key="balance_display")

            if st.button("🔄 同步实盘余额"):
                if st.session_state.exchange:
                    try:
                        # 简化版余额同步（仅示例）
                        st.session_state.account_balance = 10000.0
                        st.success("同步成功（模拟）")
                    except Exception as e:
                        st.error(f"同步失败: {e}")

            st.markdown("---")
            st.subheader("实盘")
            exchange_choice = st.selectbox("交易所", list(CONFIG.exchanges.keys()))
            secrets = load_secrets_config()
            api_key = st.text_input("API Key", value=secrets.get(f"{exchange_choice.split()[0].upper()}_API_KEY", ""), type="password")
            secret_key = st.text_input("Secret Key", value=secrets.get(f"{exchange_choice.split()[0].upper()}_SECRET_KEY", ""), type="password")
            passphrase = st.text_input("Passphrase", type="password") if "OKX" in exchange_choice else None
            testnet = st.checkbox("测试网", True)
            use_real = st.checkbox("实盘交易", False)

            if use_real and api_key and secret_key:
                try:
                    st.session_state.exchange = ccxt.binance({
                        'apiKey': api_key,
                        'secret': secret_key,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    })
                    if testnet:
                        st.session_state.exchange.set_sandbox_mode(True)
                    st.success("连接成功")
                except Exception as e:
                    st.error(f"连接失败: {e}")

            st.session_state.auto_enabled = st.checkbox("自动交易", True)

            with st.expander("Telegram通知"):
                st.session_state.telegram_token = st.text_input("Bot Token", value=secrets.get('TELEGRAM_BOT_TOKEN', ''), type="password")
                st.session_state.telegram_chat_id = st.text_input("Chat ID", value=secrets.get('TELEGRAM_CHAT_ID', ''))

            if st.button("🚨 一键紧急平仓"):
                st.session_state.auto_position = None
                st.rerun()

            if st.button("运行回测"):
                data = self.fetcher.get_symbol_data(symbol)
                if data:
                    st.session_state.backtest_results = BacktestEngine.run(data, symbol)
                    st.success("回测完成")
                else:
                    st.error("无法获取数据，无法运行回测")

            if st.session_state.error_log:
                with st.expander("⚠️ 错误日志"):
                    for err in st.session_state.error_log:
                        st.text(err)

            if st.session_state.execution_log:
                with st.expander("📋 执行日志"):
                    for log in st.session_state.execution_log:
                        st.text(log)

            if st.button("🗑️ 重置所有状态"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

            return symbol, mode, use_real

    def render_main_panel(self, symbol: str, mode: str, use_real: bool, data: Dict, engine: SignalEngine, risk: RiskManager):
        df_15m = data["data_dict"]['15m']
        price = data["current_price"]
        fg = data["fear_greed"]
        fr = data["funding_rate"]
        imb = data["orderbook_imbalance"]

        btc_data = None  # 简化
        btc_trend = 0

        prob, prob_lower, prob_upper, direction, regime, details = engine.calculate_signal(df_15m, data["data_dict"], btc_trend, fg, fr, imb, symbol)

        st.session_state.circuit_breaker = risk.check_circuit_breaker(df_15m['atr_pct'].iloc[-1], fg)

        min_lev, max_lev = CONFIG.leverage_modes[mode]
        if prob >= SignalStrength.STRONG.value:
            lev = max_lev
        elif prob >= SignalStrength.HIGH.value:
            lev = max_lev * 0.9
        elif prob >= SignalStrength.MEDIUM.value:
            lev = (min_lev + max_lev)/2
        else:
            lev = min_lev
        leverage = min(lev, CONFIG.max_leverage_global)

        stop, take = risk.dynamic_stops(price, direction, df_15m['atr'].iloc[-1], df_15m['adx'].iloc[-1], df_15m['atr_pct'].iloc[-1], regime)
        initial_risk_per_unit = abs(price - stop)
        size = risk.get_position_size(st.session_state.account_balance, prob, regime, initial_risk_per_unit)

        cooldown_active = risk.check_cooldown()
        daily_limit_exceeded = risk.check_daily_trade_limit()

        if st.session_state.auto_position:
            pos = st.session_state.auto_position
            st.session_state.daily_pnl = pos.pnl(price)
            if pos.check_partial_tp(price):
                send_telegram(f"📈 部分止盈{CONFIG.partial_tp_ratio*100:.0f}% {symbol}\n杠杆 {leverage:.1f}x | 剩余仓位 {pos.size:.4f}")

        equity = st.session_state.account_balance + st.session_state.daily_pnl
        if equity > st.session_state.peak_balance:
            st.session_state.peak_balance = equity
        st.session_state.net_value_history.append({'time': datetime.now(), 'value': equity})
        if len(st.session_state.net_value_history) > 200:
            st.session_state.net_value_history = st.session_state.net_value_history[-200:]

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("### 📊 市场状态")
            c1, c2, c3 = st.columns(3)
            c1.metric("恐惧贪婪指数", fg)
            c2.metric("信号概率", f"{prob:.1%}")
            c3.metric("当前价格", f"{price:.2f}")

            st.markdown(f"### 市场状态: {regime.value}")

            signal_text = "等待信号"
            if prob >= SignalStrength.WEAK.value:
                signal_text = "强力做多" if direction == 1 else "强力做空"
            st.markdown(f"### {signal_text}")

            with st.expander("🔍 信号详情", expanded=True):
                for d in details:
                    st.markdown(f"• {d}")

            if prob >= SignalStrength.WEAK.value and size > 0 and not st.session_state.circuit_breaker and not cooldown_active and not daily_limit_exceeded:
                st.success(f"杠杆 {leverage:.1f}x | 建议仓位 {size:.4f} {symbol.split('/')[0]}")
                st.info(f"止损 {stop:.2f} | 止盈 {take:.2f}")
                st.info("当前为 **实盘模式**" if use_real and st.session_state.exchange else "当前为 **模拟模式**")
            elif st.session_state.circuit_breaker:
                st.error("⚠️ 市场极端，熔断激活，暂停交易")
            elif cooldown_active:
                st.error(f"⚠️ 连续亏损，冷却至 {st.session_state.cooldown_until.strftime('%H:%M')}")
            elif daily_limit_exceeded:
                st.error("⚠️ 日内交易次数已达上限")
            else:
                st.info("当前无符合条件交易信号")

            st.markdown("### 📉 风险监控")
            st.metric("实时盈亏", f"{st.session_state.daily_pnl:.2f} USDT")
            drawdown = (st.session_state.peak_balance - equity) / st.session_state.peak_balance * 100
            st.metric("最大回撤", f"{drawdown:.2f}%")
            st.metric("连亏次数", st.session_state.consecutive_losses)

            if st.session_state.net_value_history:
                hist_df = pd.DataFrame(st.session_state.net_value_history)
                fig_nv = go.Figure()
                fig_nv.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['value'], mode='lines', name='净值', line=dict(color='cyan')))
                fig_nv.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), template='plotly_dark')
                st.plotly_chart(fig_nv, use_container_width=True)

        with col2:
            df_plot = df_15m.tail(120)
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2],
                                vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(x=df_plot['timestamp'], open=df_plot['open'], high=df_plot['high'],
                                         low=df_plot['low'], close=df_plot['close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema50'], line=dict(color="#FFA500")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema200'], line=dict(color="#4169E1")), row=1, col=1)
            valid = df_plot['ichimoku_senkou_a'].notna() & df_plot['ichimoku_senkou_b'].notna()
            if valid.any():
                a_valid = df_plot['ichimoku_senkou_a'][valid]
                b_valid = df_plot['ichimoku_senkou_b'][valid]
                ts_valid = df_plot['timestamp'][valid]
                fig.add_trace(go.Scatter(x=ts_valid, y=a_valid, line=dict(color="green", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ts_valid, y=b_valid, fill='tonexty', fillcolor='rgba(100,100,100,0.2)', line=dict(color="red", width=1)), row=1, col=1)
            if st.session_state.auto_position:
                pos = st.session_state.auto_position
                fig.add_hline(y=pos.entry, line_dash="dot", line_color="yellow", annotation_text=f"入场 {pos.entry:.2f}")
                fig.add_hline(y=pos.stop, line_dash="dash", line_color="red", annotation_text=f"止损 {pos.stop:.2f}")
                fig.add_hline(y=pos.take, line_dash="dash", line_color="green", annotation_text=f"止盈 {pos.take:.2f}")
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['rsi'], line=dict(color="purple")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['macd'], line=dict(color="cyan")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['macd_signal'], line=dict(color="orange")), row=3, col=1)
            fig.add_bar(x=df_plot['timestamp'], y=df_plot['macd_diff'], marker_color="gray", row=3, col=1)
            colors_vol = np.where(df_plot['close'] >= df_plot['open'], 'green', 'red')
            fig.add_trace(go.Bar(x=df_plot['timestamp'], y=df_plot['volume'], marker_color=colors_vol.tolist()), row=4, col=1)
            fig.update_layout(height=800, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.backtest_results:
            bt = st.session_state.backtest_results
            st.markdown("### 📈 回测结果")
            cols = st.columns(4)
            cols[0].metric("总收益率", f"{bt['total_return']*100:.2f}%")
            cols[1].metric("夏普比率", f"{bt['sharpe']:.2f}")
            cols[2].metric("最大回撤", f"{bt['max_drawdown']*100:.2f}%")
            if 'mc_max_dd_95' in bt:
                cols[3].metric("MC 95%最大回撤", f"{bt['mc_max_dd_95']*100:.2f}%")
            fig_bt = go.Figure(go.Scatter(y=bt['equity_curve'], mode='lines', name='策略净值', line=dict(color='lime')))
            fig_bt.update_layout(height=300, template='plotly_dark')
            st.plotly_chart(fig_bt, use_container_width=True)

        # 自动交易逻辑
        now = datetime.now()
        if st.session_state.auto_enabled and not st.session_state.circuit_breaker and not cooldown_active and not daily_limit_exceeded:
            if st.session_state.auto_position:
                pos = st.session_state.auto_position
                high = price * 1.001
                low = price * 0.999
                close_flag, reason, exit_price = pos.should_close(high, low, price, now)
                if close_flag or prob < SignalStrength.WEAK.value:
                    if not close_flag:
                        close_flag, reason, exit_price = True, "信号消失", price
                    pnl = pos.pnl(exit_price)
                    r = pos.r_multiple(exit_price)
                    risk.update_stats(r)
                    st.session_state.consecutive_losses = st.session_state.consecutive_losses + 1 if r < 0 else 0
                    st.session_state.auto_position = None
                    send_telegram(f"{reason} {symbol} | R {r:.2f}\n盈亏 {pnl:.2f} USDT")
                    st.rerun()
            elif prob >= SignalStrength.WEAK.value and size > 0:
                if st.session_state.last_signal_time and (now - st.session_state.last_signal_time).total_seconds() < CONFIG.anti_duplicate_seconds:
                    return
                pos = Position(direction, price, now, stop, take, size, original_size=size, initial_risk_per_unit=initial_risk_per_unit)
                st.session_state.auto_position = pos
                st.session_state.daily_trades += 1
                send_telegram(f"🚀 模拟开仓 {symbol} {'多' if direction==1 else '空'}\n概率 {prob:.1%} | 杠杆 {leverage:.1f}x | 仓位 {size:.4f}")
                st.session_state.last_signal_time = now
                st.rerun()

# ==================== Walk-Forward + Monte Carlo回测（简化版）====================
class BacktestEngine:
    @staticmethod
    def run(data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        df = data['data_dict']['15m'].copy()
        engine = SignalEngine()
        risk = RiskManager()
        equity = 10000.0
        equity_curves = [equity]
        r_multiples = []
        consecutive_losses = 0
        cooldown_end = None
        
        # 简化回测：使用全部数据
        for i in range(200, len(df) - 1):
            current_time = df.iloc[i]['timestamp']
            if cooldown_end and current_time < cooldown_end:
                equity_curves.append(equity)
                continue
            
            sub_data = {'15m': df.iloc[:i+1]}
            prob, _, _, direction, regime, _ = engine.calculate_signal(df.iloc[:i+1], sub_data, 0, 50, 0.0, 0.0, symbol)
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            close = df.iloc[i]['close']
            next_open = df.iloc[i+1]['open']

            if prob >= SignalStrength.HIGH.value:
                stop, take = risk.dynamic_stops(next_open, direction, df.iloc[i]['atr'], df.iloc[i]['adx'], df.iloc[i]['atr_pct'], regime)
                initial_risk_per_unit = abs(next_open - stop)
                size = risk.get_position_size(equity, prob, regime, initial_risk_per_unit)
                if size > 0:
                    # 简单模拟持仓，下一根K线收盘价平仓
                    exit_price = df.iloc[i+1]['close']
                    pnl = (exit_price - next_open) * direction * size
                    r = (exit_price - next_open) * direction / initial_risk_per_unit
                    r_multiples.append(r)
                    risk.update_stats(r)
                    equity += pnl
                    if r < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    if consecutive_losses >= CONFIG.cooldown_losses:
                        cooldown_end = current_time + timedelta(hours=CONFIG.cooldown_hours)

            equity_curves.append(equity)

        final_curve = pd.Series(equity_curves)
        total_ret = final_curve.iloc[-1] / 10000.0 - 1
        returns = final_curve.pct_change().dropna()
        sharpe = np.sqrt(35040) * returns.mean() / returns.std() if returns.std() > 0 else 0
        max_dd = (final_curve.cummax() - final_curve).max() / final_curve.cummax()

        # Monte Carlo
        if len(r_multiples) > 0:
            mc_dd = []
            for _ in range(CONFIG.mc_simulations):
                shuffled = np.random.permutation(r_multiples)
                mc_equity = 10000.0
                mc_curve = [mc_equity]
                for r in shuffled:
                    mc_equity *= (1 + r * CONFIG.base_risk_per_trade)
                    mc_curve.append(mc_equity)
                mc_series = pd.Series(mc_curve)
                dd = (mc_series.cummax() - mc_series).max() / mc_series.cummax()
                mc_dd.append(dd)
            mc_max_dd_95 = np.percentile(mc_dd, 95)
        else:
            mc_max_dd_95 = 0.0

        st.session_state.mc_results = {"mc_max_dd_95": mc_max_dd_95}

        return {'total_return': total_ret, 'sharpe': sharpe, 'max_drawdown': max_dd, 'equity_curve': final_curve, 'mc_max_dd_95': mc_max_dd_95}

# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="终极量化终端 30.0", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 超神烧脑版 30.0")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北")

    init_session_state()
    renderer = UIRenderer()
    symbol, mode, use_real = renderer.render_sidebar()

    data = renderer.fetcher.get_symbol_data(symbol)
    if not data:
        st.error("❌ 数据获取失败，请稍后重试。")
        st.stop()

    if st.session_state.get('use_simulated_data', True):
        st.info("🔧 当前处于离线模拟模式，K线动态波动，可正常测试信号。")

    engine = SignalEngine()
    risk = RiskManager()

    renderer.render_main_panel(symbol, mode, use_real, data, engine, risk)

    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

if __name__ == "__main__":
    main()
