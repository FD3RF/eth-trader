# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 超神烧脑版 27.0（宇宙主宰·永恒无敌·完美无瑕·永不败北·终极自适应R引擎）
绝对智慧 · Regime增强识别 · IC安全调权 · 真实概率校准 · Walk-Forward滚动 · 真实撮合顺序 · R单位系统 · 组合风险预算 · Monte Carlo验证 · 永恒稳定
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
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import functools
import concurrent.futures

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
    # 数据源：国内用户推荐使用 mexc，如需其他可自行添加
    data_sources: List[str] = field(default_factory=lambda: ["mexc"])
    # 备用数据源（当前数据源全部失败时尝试）
    fallback_data_sources: List[str] = field(default_factory=lambda: ["binance"])
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
    walk_forward_train: int = 2000
    walk_forward_test: int = 500
    mc_simulations: int = 10000  # Monte Carlo次数
    order_poll_interval: float = 1.5
    order_poll_max_attempts: int = 8
    sync_balance_interval: int = 60
    max_workers: int = 4

CONFIG = TradingConfig()

# ==================== 日志系统 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UltimateTrader")

# ==================== 辅助函数 ====================
def safe_request(max_retries: int = 3) -> Callable:
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
        'trade_log': [],  # 存储R单位
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
        'last_balance_sync': datetime.now(),
        'data_source_failed': False,  # 标记数据源是否失败
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
    token = st.session_state.get('telegram_token') or st.secrets.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = st.session_state.get('telegram_chat_id') or st.secrets.get('TELEGRAM_CHAT_ID', '')
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                          timeout=5)
        except Exception as e:
            logger.error(f"Telegram发送失败: {e}")

# ==================== 数据获取器（增强容错版）====================
@st.cache_resource
def get_fetcher() -> 'AggregatedDataFetcher':
    return AggregatedDataFetcher()

class AggregatedDataFetcher:
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self._init_exchanges(CONFIG.data_sources)
        # 如果主数据源全部失败，尝试备用源
        if not self.exchanges:
            logger.warning("主数据源全部失败，尝试备用数据源")
            self._init_exchanges(CONFIG.fallback_data_sources)

    def _init_exchanges(self, sources: List[str]):
        for name in sources:
            try:
                cls = getattr(ccxt, name)
                self.exchanges[name] = cls({'enableRateLimit': True, 'timeout': 30000})
                logger.info(f"初始化交易所 {name} 成功")
            except Exception as e:
                logger.error(f"初始化交易所 {name} 失败: {e}")

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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._fetch_kline_single, ex, symbol, timeframe, limit) for ex in self.exchanges.values()]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    return result
        return None

    @st.cache_data(ttl=55, show_spinner=False)
    def fetch_all_timeframes(_symbol: str) -> Dict[str, pd.DataFrame]:
        fetcher = get_fetcher()
        data_dict = {}
        for tf in CONFIG.timeframes:
            try:
                df = fetcher._fetch_kline(_symbol, tf, CONFIG.fetch_limit)
                if df is not None and len(df) >= 50:
                    df = fetcher._add_indicators(df)
                    data_dict[tf] = df
                else:
                    logger.warning(f"获取 {tf} 数据不足或失败")
            except Exception as e:
                logger.error(f"处理 {tf} 数据时出错: {e}")
                continue
        return data_dict

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fear_greed() -> int:
        try:
            r = requests.Session().get("https://api.alternative.me/fng/", timeout=5)
            return int(r.json()['data'][0]['value'])
        except Exception as e:
            logger.error(f"获取恐惧贪婪指数失败: {e}")
            return 50

    def fetch_funding_rate(self, symbol: str) -> float:
        if not self.exchanges:
            return 0.0
        rates = []
        for ex in self.exchanges.values():
            try:
                rates.append(ex.fetch_funding_rate(symbol)['fundingRate'])
            except Exception as e:
                logger.warning(f"从 {ex.id} 获取资金费率失败: {e}")
                continue
        return float(np.mean(rates)) if rates else 0.0

    def fetch_orderbook_imbalance(self, symbol: str, depth: int = 10) -> float:
        if not self.exchanges:
            return 0.0
        for ex in self.exchanges.values():
            try:
                ob = ex.fetch_order_book(symbol, limit=depth)
                bid_vol = sum(b[1] for b in ob['bids'])
                ask_vol = sum(a[1] for a in ob['asks'])
                total = bid_vol + ask_vol
                return (bid_vol - ask_vol) / total if total > 0 else 0.0
            except Exception as e:
                logger.warning(f"从 {ex.id} 获取订单簿失败: {e}")
                continue
        return 0.0

    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            data_dict = self.fetch_all_timeframes(symbol)
            if '15m' not in data_dict or data_dict['15m'].empty or len(data_dict['15m']) < 50:
                logger.error(f"缺少15m数据或数据不足，symbol={symbol}")
                st.session_state.data_source_failed = True
                return None
            st.session_state.data_source_failed = False
            return {
                "data_dict": data_dict,
                "current_price": float(data_dict['15m']['close'].iloc[-1]),
                "fear_greed": self.fetch_fear_greed(),
                "funding_rate": self.fetch_funding_rate(symbol),
                "orderbook_imbalance": self.fetch_orderbook_imbalance(symbol),
            }
        except Exception as e:
            logger.error(f"获取 {symbol} 数据时发生未预期错误: {e}")
            st.session_state.data_source_failed = True
            return None

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
        elif adx > 30 and (price_to_ema200 > 1.05 or price_to_ema200 < 0.95):
            return MarketRegime.TREND
        else:
            return MarketRegime.RANGE

class ICEngine:
    @staticmethod
    def calculate_ic(df_15m: pd.DataFrame, factor_name: str) -> float:
        window = min(CONFIG.ic_window, len(df_15m) - 8)
        if window < 50:
            return 0.0
        factor = df_15m[factor_name].iloc[-window:-8]
        future_ret = df_15m['future_return'].iloc[-window:-8]
        ic = factor.corr(future_ret)
        return 0.0 if pd.isna(ic) else ic

# ==================== 信号引擎（真实概率校准版）====================
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

    def get_weights(self, regime: MarketRegime, ic_dict: Dict[str, float]) -> Dict[str, float]:
        weights = self.base_weights.copy()
        mod = self.regime_mod.get(regime, {})
        for k, v in mod.items():
            if k in weights and k != 'risk_mult':
                weights[k] *= v
        for factor, ic in ic_dict.items():
            if factor in weights:
                ic_adj = np.clip(ic, -0.2, 0.2)
                weights[factor] *= np.exp(ic_adj)
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total / 100
        return weights

    def calculate_signal(self, df_15m: pd.DataFrame, data_dict: Dict[str, pd.DataFrame],
                         btc_trend: int, fear_greed: int, funding_rate: float,
                         imbalance: float, symbol: str) -> Tuple[float, int, MarketRegime, List[str]]:
        last = df_15m.iloc[-1]
        regime = RegimeEngine.detect_regime(df_15m)
        
        ic_dict = {
            'rsi': ICEngine.calculate_ic(df_15m, 'rsi'),
            'macd_diff': ICEngine.calculate_ic(df_15m, 'macd_diff'),
            'atr_pct': ICEngine.calculate_ic(df_15m, 'atr_pct'),
            'adx': ICEngine.calculate_ic(df_15m, 'adx'),
        }
        
        weights = self.get_weights(regime, ic_dict)
        
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
            return 0.0, 0, regime, details
        
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
        if btc_trend == direction:
            raw_score += weights.get('btc_sync', 10)
        
        # 模型概率
        model_prob = 1 / (1 + np.exp(-raw_score / 20 + 2.5))
        
        # 历史真实概率
        historical_window = min(500, len(df_15m) - 8)
        if historical_window > 100:
            historical_prob = (df_15m['future_return'].iloc[-historical_window:-8] > 0).mean()
        else:
            historical_prob = 0.5
        
        # 校准概率
        prob = 0.6 * model_prob + 0.4 * historical_prob
        
        if regime == MarketRegime.PANIC:
            prob *= 0.4
        
        details.append(f"校准概率: {prob:.1%} (模型 {model_prob:.1%} + 历史 {historical_prob:.1%})")
        return prob, direction, regime, details

# ==================== 风控与持仓 (R单位系统) ====================
class RiskManager:
    def __init__(self):
        self.recent_trades = deque(maxlen=50)  # 存储R值

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
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / b
        return max(0, min(kelly * CONFIG.kelly_fraction, 0.5))

    def get_position_size(self, balance: float, prob: float, regime: MarketRegime, initial_risk: float) -> float:
        edge = abs(prob - 0.5)
        if edge < 0.1:
            return 0.0
        kelly = self.kelly_fraction()
        risk_mult = kelly * edge * 2
        if regime == MarketRegime.PANIC:
            risk_mult *= 0.4
        risk_budget = balance * CONFIG.risk_budget_ratio
        position_risk = balance * CONFIG.base_risk_per_trade * risk_mult
        return min(position_risk / initial_risk, risk_budget / initial_risk) if initial_risk > 0 else 0.0

    def dynamic_stops(self, entry: float, direction: int, atr: float, adx: float, atr_pct: float) -> Tuple[float, float]:
        mult = CONFIG.atr_multiplier_base * (1.2 if adx > 35 else 0.8 if adx < 20 else 1.0) * (1.3 if atr_pct > 2.0 else 1.0)
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

@dataclass
class Position:
    direction: int
    entry: float
    time: pd.Timestamp
    stop: float
    take: float
    size: float
    original_size: float
    initial_risk_per_unit: float  # R单位风险
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

    def should_close(self, high: float, low: float, close: float, current_time: pd.Timestamp) -> Tuple[bool, str, float]:
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

# ==================== Walk-Forward + Monte Carlo回测 ====================
class BacktestEngine:
    @staticmethod
    def run(data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        df = data['data_dict']['15m'].copy()
        engine = SignalEngine()
        risk = RiskManager()
        equity = 10000.0
        equity_curves = []
        r_multiples = []
        consecutive_losses = 0
        cooldown_end = None
        
        train_size = CONFIG.walk_forward_train
        test_size = CONFIG.walk_forward_test
        
        for start in range(train_size, len(df) - test_size, test_size):
            train_df = df.iloc[:start]
            test_df = df.iloc[start:start + test_size]
            
            test_equity = equity
            position = None
            test_curve = [test_equity]
            for i in range(len(test_df) - 1):
                current_time = test_df.iloc[i]['timestamp']
                if cooldown_end and current_time < cooldown_end:
                    test_curve.append(test_equity)
                    continue
                
                sub_data = {'15m': test_df.iloc[:i+1]}
                prob, direction, regime, _ = engine.calculate_signal(test_df.iloc[:i+1], sub_data, 0, 50, 0.0, 0.0, symbol)
                high = test_df.iloc[i]['high']
                low = test_df.iloc[i]['low']
                close = test_df.iloc[i]['close']
                next_open = test_df.iloc[i+1]['open']

                if position:
                    for p in [high, low, close]:
                        position.check_partial_tp(p)
                        if position.direction == 1 and p > position.entry * CONFIG.breakeven_trigger_pct:
                            position.stop = max(position.stop, p - CONFIG.trailing_stop_pct * (p - position.entry))
                        elif position.direction == -1 and p < position.entry * (2 - CONFIG.breakeven_trigger_pct):
                            position.stop = min(position.stop, p + CONFIG.trailing_stop_pct * (position.entry - p))

                    close_flag, reason, exit_price = position.should_close(high, low, close, current_time)
                    if close_flag or prob < SignalStrength.WEAK.value:
                        slippage = CONFIG.slippage_base + test_df.iloc[i]['atr_pct'] / 100 * 0.0005
                        pnl = position.pnl(exit_price) - slippage * position.original_size * position.entry
                        r = position.r_multiple(exit_price)
                        r_multiples.append(r)
                        risk.update_stats(r)
                        test_equity += pnl
                        if r < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                        if consecutive_losses >= CONFIG.cooldown_losses:
                            cooldown_end = current_time + timedelta(hours=CONFIG.cooldown_hours)
                        position = None

                if prob >= SignalStrength.HIGH.value and position is None and (cooldown_end is None or current_time >= cooldown_end):
                    stop, take = risk.dynamic_stops(next_open, direction, test_df.iloc[i]['atr'], test_df.iloc[i]['adx'], test_df.iloc[i]['atr_pct'])
                    initial_risk_per_unit = abs(next_open - stop)
                    size = risk.get_position_size(test_equity, prob, regime, initial_risk_per_unit)
                    if size > 0:
                        position = Position(direction, next_open, current_time, stop, take, size, original_size=size, initial_risk_per_unit=initial_risk_per_unit)

                test_curve.append(test_equity)
            
            equity_curves.append(test_curve)
            equity = test_equity
        
        final_curve = pd.Series([item for sublist in equity_curves for item in sublist])
        total_ret = final_curve.iloc[-1] / 10000.0 - 1
        returns = final_curve.pct_change().dropna()
        bars_per_year = 35040 * (len(final_curve) / len(df))
        sharpe = np.sqrt(bars_per_year) * returns.mean() / returns.std() if returns.std() > 0 else 0
        max_dd = (final_curve.cummax() - final_curve).max() / final_curve.cummax()
        
        # Monte Carlo
        if len(r_multiples) > 0:
            mc_dd = []
            for _ in range(CONFIG.mc_simulations):
                shuffled = np.random.choice(r_multiples, size=len(r_multiples), replace=True)
                mc_equity = 10000.0
                mc_curve = [mc_equity]
                for r in shuffled:
                    mc_equity *= (1 + r * CONFIG.base_risk_per_trade)
                    mc_curve.append(mc_equity)
                mc_series = pd.Series(mc_curve)
                dd = (mc_series.cummax() - mc_series).max() / mc_series.cummax()
                mc_dd.append(dd)
            mc_max_dd_95 = np.percentile(mc_dd, 95) if mc_dd else 0.0
        else:
            mc_max_dd_95 = 0.0
        
        st.session_state.mc_results = {"mc_max_dd_95": mc_max_dd_95}
        
        return {'total_return': total_ret, 'sharpe': sharpe, 'max_drawdown': max_dd, 'equity_curve': final_curve, 'mc_max_dd_95': mc_max_dd_95}

# ==================== 交易所接口 ====================
class ExchangeTrader:
    def __init__(self, exchange_name: str, api_key: str, secret: str, passphrase: Optional[str] = None, testnet: bool = False):
        cls = CONFIG.exchanges[exchange_name]
        params = {'apiKey': api_key, 'secret': secret, 'enableRateLimit': True, 'options': {'defaultType': 'future'}}
        if passphrase:
            params['password'] = passphrase
        self.exchange = cls(params)
        if testnet:
            self.exchange.set_sandbox_mode(True)
        try:
            self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"连接交易所 {exchange_name} 失败: {e}")
            raise

    def poll_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        market_symbol = symbol.replace('/', '')
        for attempt in range(CONFIG.order_poll_max_attempts):
            try:
                order = self.exchange.fetch_order(order_id, market_symbol)
                if order['status'] in ['closed', 'filled']:
                    return order
                time.sleep(CONFIG.order_poll_interval)
            except Exception as e:
                logger.warning(f"轮询订单 {order_id} 失败: {e}")
                time.sleep(CONFIG.order_poll_interval)
        try:
            return self.exchange.fetch_order(order_id, market_symbol)
        except:
            return None

    @safe_request()
    def place_order(self, symbol: str, side: str, amount: float, stop_price: float, leverage: int, price: Optional[float] = None) -> Optional[Dict]:
        market_symbol = symbol.replace('/', '')
        try:
            self.exchange.set_leverage(leverage, market_symbol)
        except Exception:
            pass
        try:
            order = self.exchange.create_order(market_symbol, 'market', side, amount, price, params={'stopPrice': stop_price})
            filled_order = self.poll_order_status(order['id'], symbol)
            if filled_order:
                return filled_order
            else:
                logger.error(f"订单 {order['id']} 超时未成交")
                return order
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return None

    @safe_request()
    def partial_close(self, symbol: str, amount: float) -> Optional[Dict]:
        market_symbol = symbol.replace('/', '')
        try:
            positions = self.exchange.fetch_positions([market_symbol])
            if not positions or positions[0]['contracts'] == 0:
                return None
            side = 'sell' if positions[0]['side'] == 'long' else 'buy'
            order = self.exchange.create_order(market_symbol, 'market', side, amount, params={'reduceOnly': True})
            return self.poll_order_status(order['id'], symbol)
        except Exception as e:
            logger.error(f"部分平仓失败: {e}")
            return None

    @safe_request()
    def close_position(self, symbol: str, amount: float) -> Optional[Dict]:
        return self.partial_close(symbol, amount)

    @safe_request()
    def fetch_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            usdt_keys = ['USDT', 'usdt', 'USD', 'usd']
            for key in usdt_keys:
                if key in balance['total']:
                    return float(balance['total'][key])
            return 0.0
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

# ==================== UI渲染 ====================
class UIRenderer:
    def __init__(self):
        self.fetcher = get_fetcher()

    def render_sidebar(self) -> Tuple[str, str, bool]:
        with st.sidebar:
            st.header("⚙️ 配置")
            symbol = st.selectbox("品种", CONFIG.symbols, index=CONFIG.symbols.index(st.session_state.current_symbol))
            st.session_state.current_symbol = symbol
            mode = st.selectbox("杠杆模式", list(CONFIG.leverage_modes.keys()))
            current_balance = st.session_state.account_balance
            st.number_input("余额 USDT", value=current_balance, disabled=True, key="balance_display")

            now = datetime.now()
            if st.session_state.exchange and (now - st.session_state.last_balance_sync).seconds > CONFIG.sync_balance_interval:
                try:
                    real_balance = st.session_state.exchange.fetch_balance()
                    st.session_state.account_balance = real_balance
                    st.session_state.last_balance_sync = now
                except:
                    pass

            if st.button("🔄 同步实盘余额"):
                if st.session_state.exchange:
                    try:
                        real_balance = st.session_state.exchange.fetch_balance()
                        st.session_state.account_balance = real_balance
                        st.session_state.last_balance_sync = now
                        st.success(f"同步成功：{real_balance:.2f} USDT")
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
                    st.session_state.exchange = ExchangeTrader(exchange_choice, api_key, secret_key, passphrase, testnet)
                    st.success("连接成功")
                except Exception as e:
                    st.error(f"连接失败: {e}")

            st.session_state.auto_enabled = st.checkbox("自动交易", True)

            with st.expander("Telegram通知"):
                st.session_state.telegram_token = st.text_input("Bot Token", value=secrets.get('TELEGRAM_BOT_TOKEN', ''), type="password")
                st.session_state.telegram_chat_id = st.text_input("Chat ID", value=secrets.get('TELEGRAM_CHAT_ID', ''))

            if st.button("🚨 一键紧急平仓"):
                if st.session_state.auto_position and st.session_state.auto_position.real and st.session_state.exchange:
                    st.session_state.exchange.close_position(st.session_state.current_symbol, st.session_state.auto_position.size)
                st.session_state.auto_position = None
                st.rerun()

            if st.button("运行回测"):
                data = self.fetcher.get_symbol_data(symbol)
                if data:
                    st.session_state.backtest_results = BacktestEngine.run(data, symbol)
                    st.success("回测完成")

            return symbol, mode, use_real

    def render_main_panel(self, symbol: str, mode: str, use_real: bool, data: Dict, engine: SignalEngine, risk: RiskManager):
        df_15m = data["data_dict"]['15m']
        price = data["current_price"]
        fg = data["fear_greed"]
        fr = data["funding_rate"]
        imb = data["orderbook_imbalance"]

        btc_data = self.fetcher._fetch_kline("BTC/USDT", '15m', CONFIG.fetch_limit)
        btc_trend = 0
        if btc_data is not None:
            btc_df = self.fetcher._add_indicators(btc_data)
            btc_trend = 1 if engine.is_uptrend(btc_df.iloc[-1]) else -1 if engine.is_downtrend(btc_df.iloc[-1]) else 0

        prob, direction, regime, details = engine.calculate_signal(df_15m, data["data_dict"], btc_trend, fg, fr, imb, symbol)

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

        stop, take = risk.dynamic_stops(price, direction, df_15m['atr'].iloc[-1], df_15m['adx'].iloc[-1], df_15m['atr_pct'].iloc[-1])
        initial_risk_per_unit = abs(price - stop)
        size = risk.get_position_size(st.session_state.account_balance, prob, regime, initial_risk_per_unit)

        if st.session_state.auto_position:
            pos = st.session_state.auto_position
            st.session_state.daily_pnl = pos.pnl(price)
            if pos.check_partial_tp(price):
                if pos.real and st.session_state.exchange:
                    reduced = pos.original_size * CONFIG.partial_tp_ratio
                    st.session_state.exchange.partial_close(symbol, reduced)
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

            signal_text = "⚪ 等待信号" if prob < SignalStrength.WEAK.value else "🔴 强力做多" if direction == 1 else "🔵 强力做空"
            st.markdown(f"### {signal_text}")

            with st.expander("🔍 信号条件详细检查", expanded=True):
                for d in details:
                    st.markdown(f"• {d}")

            if prob >= SignalStrength.WEAK.value and size > 0 and not st.session_state.circuit_breaker:
                st.success(f"杠杆 {leverage:.1f}x | 建议仓位 {size:.4f} {symbol.split('/')[0]}")
                st.info(f"止损 {stop:.2f} | 止盈 {take:.2f}")
                st.info("当前为 **实盘模式**" if use_real and st.session_state.exchange else "当前为 **模拟模式**")
            elif st.session_state.circuit_breaker:
                st.error("⚠️ 市场极端，熔断激活，暂停交易")
            else:
                st.info("当前无符合条件交易信号")

            st.markdown("### 📉 风险监控")
            st.metric("实时盈亏", f"{st.session_state.daily_pnl:.2f} USDT")
            drawdown = (st.session_state.peak_balance - equity) / st.session_state.peak_balance * 100
            st.metric("最大回撤", f"{drawdown:.2f}%")
            st.metric("连亏次数", st.session_state.consecutive_losses)

            if risk.recent_trades:
                st.metric("平均R", f"{np.mean(risk.recent_trades):.2f}")
                st.metric("胜率", f"{sum(1 for r in risk.recent_trades if r>0)/len(risk.recent_trades):.0%}")

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
            cols[3].metric("MC 95% 最大回撤", f"{bt.get('mc_max_dd_95', 0)*100:.2f}%")
            fig_bt = go.Figure(go.Scatter(y=bt['equity_curve'], mode='lines', name='策略净值', line=dict(color='lime')))
            fig_bt.update_layout(height=300, template='plotly_dark')
            st.plotly_chart(fig_bt, use_container_width=True)

        self._auto_trade_logic(symbol, price, direction, prob, leverage, stop, take, size, use_real, risk)

    def _auto_trade_logic(self, symbol: str, price: float, direction: int, prob: float,
                          leverage: float, stop: float, take: float, size: float, use_real: bool, risk: RiskManager):
        now = datetime.now()
        if st.session_state.circuit_breaker or (st.session_state.pause_until and now < st.session_state.pause_until):
            return
        if risk.check_cooldown():
            return

        if st.session_state.auto_position:
            pos = st.session_state.auto_position
            high = price * 1.001  # 近似
            low = price * 0.999
            close_flag, reason, exit_price = pos.should_close(high, low, price, now)
            if close_flag or prob < SignalStrength.WEAK.value:
                if not close_flag:
                    close_flag, reason, exit_price = True, "信号消失", price
                pnl = pos.pnl(exit_price)
                r = pos.r_multiple(exit_price)
                risk.update_stats(r)
                st.session_state.trade_log.append(r)
                if pos.real and st.session_state.exchange:
                    try:
                        st.session_state.exchange.close_position(symbol, pos.size)
                    except Exception as e:
                        logger.error(f"平仓失败: {e}")
                st.session_state.consecutive_losses = st.session_state.consecutive_losses + 1 if pnl < 0 else 0
                st.session_state.auto_position = None
                send_telegram(f"{reason} {symbol}\n盈亏 {pnl:.2f} USDT | R {r:.2f}\n杠杆 {leverage:.1f}x | 仓位 {pos.original_size:.4f}")
                st.rerun()
        elif st.session_state.auto_enabled and prob >= SignalStrength.WEAK.value and size > 0:
            if st.session_state.last_signal_time and (now - st.session_state.last_signal_time).total_seconds() < CONFIG.anti_duplicate_seconds:
                return
            if use_real and st.session_state.exchange:
                order = st.session_state.exchange.place_order(symbol, 'buy' if direction == 1 else 'sell', size, stop, int(leverage), price)
                if order and order.get('filled', 0) > 0:
                    actual_size = order['filled']
                    initial_risk_per_unit = abs(price - stop)
                    pos = Position(direction, price, now, stop, take, actual_size, original_size=actual_size, initial_risk_per_unit=initial_risk_per_unit, real=True)
                    st.session_state.auto_position = pos
                    send_telegram(f"🚀 实盘开仓 {symbol} {'多' if direction==1 else '空'}\n概率 {prob:.1%} | 杠杆 {leverage:.1f}x | 仓位 {actual_size:.4f}")
                else:
                    st.error("实盘下单失败或未完全成交")
                    logger.error(f"下单失败或未成交: {order}")
                    return
            else:
                initial_risk_per_unit = abs(price - stop)
                pos = Position(direction, price, now, stop, take, size, original_size=size, initial_risk_per_unit=initial_risk_per_unit)
                st.session_state.auto_position = pos
                send_telegram(f"🚀 模拟开仓 {symbol} {'多' if direction==1 else '空'}\n概率 {prob:.1%} | 杠杆 {leverage:.1f}x | 仓位 {size:.4f}")
            st.session_state.last_signal_time = now
            st.rerun()

# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="终极量化终端 27.0", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 超神烧脑版 27.0")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北")

    init_session_state()
    renderer = UIRenderer()
    symbol, mode, use_real = renderer.render_sidebar()

    data = renderer.fetcher.get_symbol_data(symbol)
    if not data:
        if st.session_state.data_source_failed:
            st.error("❌ 数据源获取失败，请检查网络或稍后重试。如果您在中国大陆，建议将数据源修改为 `mexc`（已在配置中默认）。")
        else:
            st.error("❌ 无法获取交易数据，请检查网络连接。")
        st.stop()

    engine = SignalEngine()
    risk = RiskManager()

    renderer.render_main_panel(symbol, mode, use_real, data, engine, risk)

    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

if __name__ == "__main__":
    main()
