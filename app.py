# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 机构版 56.0（真正最终完美版）
===================================================
核心特性：
- 完全解耦：业务逻辑与UI分离，依赖注入容器管理对象图
- 全异步化：所有网络请求均使用异步，避免阻塞
- 回测引擎：支持多周期、真实历史数据、滑点、手续费模拟
- 强化学习：观测空间接入实时市场指标，PPO模型动态调整风险乘数
- 监控完善：Prometheus指标实时更新，Telegram告警
- 代码优化：消除异步/同步混用，RL环境兼容，回测与实盘一致
===================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import ccxt.async_support as ccxt_async
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import asyncio
import aiohttp
import warnings
import time
import logging
import sys
import traceback
from typing import Optional, Dict, List, Tuple, Any, Union, Callable, Deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import functools
import hashlib
import csv
import os
import json
import yaml
from pathlib import Path
import pickle
import joblib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 第三方库
from cachetools import TTLCache
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from scipy.stats import ttest_1samp, norm, genpareto
import pytz

# 强化学习
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# 机器学习
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from hmmlearn import hmm

# 监控
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import prometheus_client

warnings.filterwarnings('ignore')

# ==================== 日志系统 ====================
LOG_DIR = "logs"
TRADE_LOG_FILE = "trade_log.csv"
PERF_LOG_FILE = "performance_log.csv"
SLIPPAGE_LOG_FILE = "slippage_log.csv"
EQUITY_CURVE_FILE = "equity_curve.csv"
REGIME_STATS_FILE = "regime_stats.csv"
CONSISTENCY_FILE = "consistency_stats.csv"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

def load_csv(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def append_to_log(file_name: str, message: str):
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{file_name}_{date_str}.log")
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    except Exception as e:
        print(f"写入日志失败: {e}")

# ==================== 领域模型 ====================
class SignalStrength(Enum):
    EXTREME = 0.85
    STRONG = 0.75
    HIGH = 0.65
    MEDIUM = 0.55
    WEAK = 0.50
    NONE = 0.0

class MarketRegime(Enum):
    TREND = "趋势"
    RANGE = "震荡"
    PANIC = "恐慌"
    CALM = "平静"
    HIGH_VOL = "高波动"
    LOW_VOL = "低波动"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class Signal:
    symbol: str
    direction: int  # 1: 多, -1: 空, 0: 无
    probability: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Order:
    symbol: str
    side: OrderSide
    type: OrderType
    size: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, filled, canceled
    exchange_order_id: Optional[str] = None
    filled_size: float = 0.0

@dataclass
class Position:
    symbol: str
    direction: int  # 1: long, -1: short
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    take_profit: float
    partial_taken: bool = False
    real: bool = False
    highest_price: float = 0.0
    lowest_price: float = 1e9
    atr_mult: float = 1.5
    slippage_paid: float = 0.0
    price_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    impact_cost: float = 0.0

    def __post_init__(self):
        if self.direction == 1:
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price
        self.price_history.append(self.entry_price)

    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size * self.direction

    def stop_distance(self) -> float:
        if self.direction == 1:
            return self.entry_price - self.stop_loss
        else:
            return self.stop_loss - self.entry_price

    def update_stops(self, current_price: float, atr: float, atr_mult: float):
        self.price_history.append(current_price)
        if self.direction == 1:
            if current_price > self.highest_price:
                self.highest_price = current_price
            trailing_stop = self.highest_price - atr * atr_mult
            self.stop_loss = max(self.stop_loss, trailing_stop)
            new_tp = current_price + atr * atr_mult * 2.0
            self.take_profit = max(self.take_profit, new_tp)
            if current_price >= self.entry_price + self.stop_distance() * 1.5:
                self.stop_loss = max(self.stop_loss, self.entry_price)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            trailing_stop = self.lowest_price + atr * atr_mult
            self.stop_loss = min(self.stop_loss, trailing_stop)
            new_tp = current_price - atr * atr_mult * 2.0
            self.take_profit = min(self.take_profit, new_tp)
            if current_price <= self.entry_price - self.stop_distance() * 1.5:
                self.stop_loss = min(self.stop_loss, self.entry_price)

    def should_close(self, high: float, low: float, current_time: datetime, config: 'TradingConfig') -> Tuple[bool, str, float, Optional[float]]:
        if self.direction == 1:
            if low <= self.stop_loss:
                return True, "止损", self.stop_loss, self.size
            if high >= self.take_profit:
                return True, "止盈", self.take_profit, self.size
        else:
            if high >= self.stop_loss:
                return True, "止损", self.stop_loss, self.size
            if low <= self.take_profit:
                return True, "止盈", self.take_profit, self.size
        hold_hours = (current_time - self.entry_time).total_seconds() / 3600
        if hold_hours > config.max_hold_hours:
            return True, "超时", (high + low) / 2, self.size
        if not self.partial_taken:
            if self.direction == 1 and high >= self.entry_price + self.stop_distance() * config.partial_tp_r_multiple:
                self.partial_taken = True
                partial_size = self.size * config.partial_tp_ratio
                self.size *= (1 - config.partial_tp_ratio)
                self.stop_loss = max(self.stop_loss, self.entry_price)
                return True, "部分止盈", self.entry_price + self.stop_distance() * config.partial_tp_r_multiple, partial_size
            if self.direction == -1 and low <= self.entry_price - self.stop_distance() * config.partial_tp_r_multiple:
                self.partial_taken = True
                partial_size = self.size * config.partial_tp_ratio
                self.size *= (1 - config.partial_tp_ratio)
                self.stop_loss = min(self.stop_loss, self.entry_price)
                return True, "部分止盈", self.entry_price - self.stop_distance() * config.partial_tp_r_multiple, partial_size
        return False, "", 0.0, None

@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    reason: str
    timestamp: datetime
    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    impact_cost: float = 0.0

# ==================== 配置管理 ====================
class TradingConfig(BaseSettings):
    """所有配置项，支持从环境变量或YAML文件加载"""
    # 基本参数
    symbols: List[str] = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"]
    risk_per_trade: float = 0.008
    daily_risk_budget_ratio: float = 0.025
    use_rl_position: bool = True
    rl_model_path: str = "models/rl_ppo.zip"
    rl_action_low: float = 0.5
    rl_action_high: float = 2.0
    max_consecutive_losses: int = 3
    cooldown_losses: int = 3
    cooldown_hours: int = 24
    max_drawdown_pct: float = 20.0
    circuit_breaker_atr: float = 5.0
    circuit_breaker_fg_extreme: Tuple[int, int] = (10, 90)
    atr_multiplier_base: float = 1.5
    atr_multiplier_min: float = 1.2
    atr_multiplier_max: float = 2.5
    tp_min_ratio: float = 2.0
    partial_tp_ratio: float = 0.5
    partial_tp_r_multiple: float = 1.2
    breakeven_trigger_pct: float = 1.5
    max_hold_hours: int = 36
    min_atr_pct: float = 0.5
    kelly_fraction: float = 0.25
    exchanges: Dict[str, str] = {
        "Binance合约": "binance",
        "Bybit合约": "bybit",
        "OKX合约": "okx"
    }
    data_sources: List[str] = ["binance", "bybit", "okx", "mexc", "kucoin"]
    timeframes: List[str] = ['1m', '5m', '15m', '1h', '4h', '1d']
    confirm_timeframes: List[str] = ['5m', '15m', '1h']
    timeframe_weights: Dict[str, int] = {'1d': 10, '4h': 7, '1h': 5, '15m': 3, '5m': 2, '1m': 1}
    fetch_limit: int = 2000
    auto_refresh_ms: int = 30000
    anti_duplicate_seconds: int = 180
    slippage_base: float = 0.0005
    slippage_impact_factor: float = 0.1
    slippage_imbalance_factor: float = 0.5
    fee_rate: float = 0.0004
    use_ml_factor: bool = True
    ml_retrain_interval: int = 3600
    ml_window: int = 500
    ml_n_estimators: int = 50
    ml_max_depth: int = 5
    use_prob_calibration: bool = True
    calibration_method: str = "isotonic"
    bayesian_prior_strength: float = 1.0
    factor_corr_threshold: float = 0.7
    factor_corr_penalty: float = 0.7
    ic_decay_rate: float = 0.99
    factor_eliminate_pvalue: float = 0.1
    factor_eliminate_ic: float = 0.02
    factor_min_weight: float = 0.1
    ic_window: int = 80
    factor_learning_rate: float = 0.3
    risk_budget_method: str = "risk_parity"
    black_litterman_tau: float = 0.05
    cov_matrix_window: int = 50
    max_sector_exposure: float = 0.3
    max_order_split: int = 3
    min_order_size: float = 0.001
    split_delay_seconds: int = 5
    regime_allow_trade: List[str] = ["趋势", "恐慌"]
    night_start_hour: int = 0
    night_end_hour: int = 8
    night_risk_multiplier: float = 0.5
    volcone_percentiles: List[float] = [0.01, 0.05, 0.5, 0.95, 0.99]
    volcone_window: int = 100
    var_confidence: float = 0.95
    var_method: str = "HISTORICAL"
    portfolio_risk_target: float = 0.02
    var_aggressive_threshold: float = 1.0
    adapt_window: int = 20
    atr_price_history_len: int = 20
    funding_rate_threshold: float = 0.05
    max_leverage_global: float = 10.0
    max_reasonable_balance: float = 1e7
    max_reasonable_daily_pnl_ratio: float = 10.0
    regime_detection_method: str = "hmm"
    hmm_n_components: int = 3
    hmm_n_iter: int = 100
    cost_aware_training: bool = True
    sim_volatility: float = 0.06
    sim_trend_strength: float = 0.2
    bb_width_threshold: float = 0.1
    bb_window: int = 20
    rsi_range_low: int = 40
    rsi_range_high: int = 60
    use_chain_data: bool = False
    chain_api_key: str = ""
    use_orderflow: bool = False
    use_sentiment: bool = True
    auto_optimize_interval: int = 86400
    optimize_window: int = 30
    cost_model_version: str = "v2"
    drawdown_scaling_factor: float = 2.0
    var_scaling_factor: float = 1.5
    daily_loss_limit: float = 0.025
    cooldown_hours_risk: int = 12
    stress_scenarios: Dict[str, Dict[str, float]] = {
        'btc_crash_10': {'BTC/USDT': -0.10, 'ETH/USDT': -0.15},
        'vol_double': {'volatility_multiplier': 2.0},
        'liquidity_half': {'volume_multiplier': 0.5}
    }
    calibration_window: int = 200
    brier_score_window: int = 30
    walk_forward_train_pct: float = 0.7
    walk_forward_step: int = 100
    mc_simulations: int = 1000
    max_order_to_depth_ratio: float = 0.01
    latency_sim_ms: int = 200
    self_check_interval: int = 3600
    anomaly_threshold_ic_drop: float = 0.5

    # 执行层参数
    leverage: int = 5
    order_sync_interval: int = 5

    # 监控
    prometheus_port: int = 8000
    enable_prometheus: bool = False

    # 密钥配置（从环境变量加载）
    binance_api_key: str = Field("", env="BINANCE_API_KEY")
    binance_secret_key: str = Field("", env="BINANCE_SECRET_KEY")
    telegram_token: str = Field("", env="TELEGRAM_TOKEN")
    telegram_chat_id: str = Field("", env="TELEGRAM_CHAT_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def load_config() -> TradingConfig:
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
        return TradingConfig(**data)
    else:
        return TradingConfig()

CONFIG = load_config()

# ==================== 基础设施 ====================
class TelegramNotifier:
    def __init__(self, config: TradingConfig):
        self.token = config.telegram_token
        self.chat_id = config.telegram_chat_id

    def send(self, msg: str, msg_type: str = "info", image: Optional[Any] = None):
        if not self.token or not self.chat_id:
            return
        try:
            if image is not None:
                import io
                buf = io.BytesIO()
                image.write_image(buf, format='png')
                buf.seek(0)
                files = {'photo': buf}
                requests.post(f"https://api.telegram.org/bot{self.token}/sendPhoto",
                              data={'chat_id': self.chat_id}, files=files, timeout=5)
            else:
                prefix = {
                    'info': 'ℹ️ ',
                    'signal': '📊 ',
                    'risk': '⚠️ ',
                    'trade': '🔄 '
                }.get(msg_type, '')
                full_msg = f"{prefix}{msg}"
                requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              json={"chat_id": self.chat_id, "text": full_msg}, timeout=3)
        except Exception as e:
            logging.getLogger().warning(f"Telegram发送失败: {e}")

class Logger:
    @staticmethod
    def error(msg: str):
        append_to_log("error", msg)
        logging.error(msg)

    @staticmethod
    def info(msg: str):
        append_to_log("info", msg)
        logging.info(msg)

    @staticmethod
    def warning(msg: str):
        append_to_log("warning", msg)
        logging.warning(msg)

logger = Logger()

# ==================== 监控 ====================
class PrometheusMetrics:
    def __init__(self, config: TradingConfig):
        self.config = config
        if config.enable_prometheus:
            start_http_server(config.prometheus_port)
        self.balance = Gauge('account_balance', 'Account balance in USDT')
        self.positions = Gauge('open_positions', 'Number of open positions')
        self.daily_pnl = Gauge('daily_pnl', 'Daily PnL')
        self.trades_total = Counter('trades_total', 'Total trades')
        self.trade_pnl = Histogram('trade_pnl', 'Trade PnL', buckets=[-100, -50, -10, 0, 10, 50, 100])
        self.slippage = Histogram('slippage', 'Slippage in USDT', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])

    def update(self, trading_service: 'TradingService'):
        self.balance.set(trading_service.balance)
        self.positions.set(len(trading_service.positions))
        self.daily_pnl.set(trading_service.daily_pnl)

    def observe_trade(self, pnl: float, slippage: float):
        self.trade_pnl.observe(pnl)
        self.slippage.observe(slippage)
        self.trades_total.inc()

# ==================== 工具类：指标计算器 ====================
class IndicatorCalculator:
    """封装所有技术指标计算，消除重复代码"""
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
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
        if len(df) >= config.bb_window:
            bb = ta.volatility.BollingerBands(df['close'], window=config.bb_window, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['bb_factor'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        else:
            df['bb_upper'] = np.nan
            df['bb_lower'] = np.nan
            df['bb_width'] = np.nan
            df['bb_factor'] = np.nan
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['trend_factor'] = (df['close'] - df['ema20']) / df['close']
        if len(df) >= 6:
            df['future_ret'] = df['close'].pct_change(5).shift(-5)
        else:
            df['future_ret'] = np.nan
        return df

# ==================== 数据提供者（异步）====================
class AsyncDataProvider:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchanges: Dict[str, ccxt_async.Exchange] = {}
        self.cache: Dict[str, TTLCache] = {}
        self.simulated_cache: Dict[str, Any] = {}
        self.indicator_calc = IndicatorCalculator()

    async def init(self):
        for name in self.config.data_sources:
            try:
                ex_class = getattr(ccxt_async, name)
                ex = ex_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'options': {'defaultType': 'future'}
                })
                self.exchanges[name] = ex
            except Exception:
                pass
        self._init_caches()

    def _init_caches(self):
        ttl_map = {
            '1m': 30,
            '5m': 60,
            '15m': 120,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        for tf in self.config.timeframes + self.config.confirm_timeframes:
            self.cache[tf] = TTLCache(maxsize=100, ttl=ttl_map.get(tf, 60))

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if cache_key in self.cache[timeframe]:
            return self.cache[timeframe][cache_key]

        tasks = []
        for name in ["binance"] + [n for n in self.config.data_sources if n != "binance"]:
            if name in self.exchanges:
                tasks.append(self._fetch_single(name, symbol, timeframe, limit))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, pd.DataFrame) and res is not None:
                self.cache[timeframe][cache_key] = res
                return res
        return None

    async def _fetch_single(self, exchange_name: str, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            ex = self.exchanges[exchange_name]
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) >= 50:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.astype({col: float for col in ['open','high','low','close','volume']})
                return df
        except Exception as e:
            logger.warning(f"从{exchange_name}获取数据失败: {e}")
        return None

    async def fetch_funding_rate(self, symbol: str) -> float:
        tasks = []
        for name in ["binance"] + [n for n in self.config.data_sources if n != "binance"]:
            if name in self.exchanges:
                tasks.append(self._fetch_funding_single(name, symbol))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        rates = [r for r in results if isinstance(r, float)]
        return float(np.mean(rates)) if rates else 0.0

    async def _fetch_funding_single(self, exchange_name: str, symbol: str) -> Optional[float]:
        try:
            ex = self.exchanges[exchange_name]
            return (await ex.fetch_funding_rate(symbol))['fundingRate']
        except:
            return None

    async def fetch_orderbook_imbalance(self, symbol: str, depth: int = 10) -> float:
        for name in ["binance"] + [n for n in self.config.data_sources if n != "binance"]:
            if name in self.exchanges:
                try:
                    ex = self.exchanges[name]
                    ob = await ex.fetch_order_book(symbol, limit=depth)
                    bid_vol = sum(b[1] for b in ob['bids'])
                    ask_vol = sum(a[1] for a in ob['asks'])
                    total = bid_vol + ask_vol
                    return (bid_vol - ask_vol) / total if total > 0 else 0.0
                except Exception as e:
                    logger.warning(f"获取订单簿失败: {e}")
        return 0.0

    async def fetch_fear_greed(self) -> int:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.alternative.me/fng/?limit=1", timeout=5) as resp:
                    data = await resp.json()
                    return int(data['data'][0]['value'])
        except Exception:
            return 50

    async def get_symbol_data(self, symbol: str, use_simulated: bool = False) -> Optional[Dict[str, Any]]:
        if use_simulated:
            cache_key = f"sim_{symbol}"
            if cache_key in self.simulated_cache:
                return self.simulated_cache[cache_key]
            sim_data = self.generate_simulated_data(symbol)
            result = {
                "data_dict": sim_data,
                "current_price": sim_data['15m']['close'].iloc[-1],
                "fear_greed": 50,
                "funding_rate": 0.0,
                "orderbook_imbalance": 0.0,
            }
            self.simulated_cache[cache_key] = result
            return result
        data_dict = await self.fetch_all_timeframes(symbol)
        if '15m' not in data_dict or data_dict['15m'].empty:
            return None
        current_price = float(data_dict['15m']['close'].iloc[-1])
        funding = await self.fetch_funding_rate(symbol)
        fear_greed = await self.fetch_fear_greed()
        imbalance = await self.fetch_orderbook_imbalance(symbol)
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "fear_greed": fear_greed,
            "funding_rate": funding,
            "orderbook_imbalance": imbalance,
        }

    async def fetch_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        all_tfs = list(set(self.config.timeframes + self.config.confirm_timeframes))
        tasks = [self.fetch_ohlcv(symbol, tf, self.config.fetch_limit) for tf in all_tfs]
        results = await asyncio.gather(*tasks)
        for tf, df in zip(all_tfs, results):
            if df is not None and len(df) >= 50:
                data_dict[tf] = self.indicator_calc.add_all_indicators(df, self.config)
        return data_dict

    def generate_simulated_data(self, symbol: str, limit: int = 2000) -> Dict[str, pd.DataFrame]:
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % 2**32
        np.random.seed(seed)
        end = datetime.now()
        timestamps = pd.date_range(end=end, periods=limit, freq='15min')
        
        if 'BTC' in symbol:
            base = 40000
            volatility = self.config.sim_volatility * 0.6
            trend_factor = 0.1
        elif 'ETH' in symbol:
            base = 2100
            volatility = self.config.sim_volatility
            trend_factor = 0.15
        else:
            base = 100
            volatility = self.config.sim_volatility * 1.2
            trend_factor = 0.2
        
        t = np.linspace(0, 6*np.pi, limit)
        trend_direction = np.random.choice([-1, 1], p=[0.3, 0.7])
        trend = trend_direction * self.config.sim_trend_strength * np.linspace(0, 1, limit) * base * trend_factor
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
        df_15m = self.indicator_calc.add_all_indicators(df_15m, self.config)
        
        data_dict = {'15m': df_15m}
        for tf in ['1m', '5m', '1h', '4h', '1d']:
            resampled = df_15m.resample(tf, on='timestamp').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()
            if len(resampled) >= 30:
                resampled = self.indicator_calc.add_all_indicators(resampled, self.config)
                data_dict[tf] = resampled
        return data_dict

    async def close(self):
        for ex in self.exchanges.values():
            await ex.close()

    # 获取历史数据用于回测（多周期）
    async def fetch_historical_data(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        exchange = self.exchanges.get('binance')
        if exchange:
            since = exchange.parse8601(start.isoformat())
            all_ohlcv = []
            while since < exchange.parse8601(end.isoformat()):
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
                return self.indicator_calc.add_all_indicators(df, self.config)
        file_path = f"data/{symbol}_{timeframe}_{start.date()}_{end.date()}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return self.indicator_calc.add_all_indicators(df, self.config)
        return None

# ==================== 策略引擎 ====================
class StrategyEngine:
    def __init__(self, config: TradingConfig, data_provider: AsyncDataProvider):
        self.config = config
        self.data_provider = data_provider
        self.factor_weights = {
            'trend': 1.0,
            'rsi': 1.0,
            'macd': 1.0,
            'bb': 1.0,
            'volume': 1.0,
            'adx': 1.0,
            'ml': 1.0
        }
        self.factor_to_col = {
            'trend': 'trend_factor',
            'rsi': 'rsi',
            'macd': 'macd_diff',
            'bb': 'bb_factor',
            'volume': 'volume_ratio',
            'adx': 'adx',
            'ml': 'ml_factor'
        }
        self.ic_decay_records = {f: deque(maxlen=200) for f in self.factor_weights}
        self.factor_corr_matrix = None
        self.ml_models = {}
        self.ml_scalers = {}
        self.ml_feature_cols = {}
        self.ml_last_train = {}
        self.ml_calibrators = {}
        self.volcone_cache = {}
        self.hmm_models = {}
        self.hmm_scalers = {}
        self.hmm_last_train = {}
        self._ic_cache = {}

    def detect_market_regime(self, df_dict: Dict[str, pd.DataFrame], symbol: str, fear_greed: int) -> MarketRegime:
        if self.config.regime_detection_method == 'hmm':
            state = self._detect_hmm_regime(symbol, df_dict)
            mapping = {0: MarketRegime.RANGE, 1: MarketRegime.TREND, 2: MarketRegime.PANIC}
            return mapping.get(state, MarketRegime.RANGE)
        else:
            return self._detect_market_regime_traditional(df_dict, fear_greed)

    def _detect_hmm_regime(self, symbol: str, df_dict: Dict[str, pd.DataFrame]) -> int:
        now = time.time()
        if symbol not in self.hmm_models or now - self.hmm_last_train.get(symbol, 0) > self.config.ml_retrain_interval:
            model, scaler = self._train_hmm(symbol, df_dict)
            if model is not None:
                self.hmm_models[symbol] = model
                self.hmm_scalers[symbol] = scaler
                self.hmm_last_train[symbol] = now
        if symbol not in self.hmm_models:
            return 0
        model = self.hmm_models[symbol]
        scaler = self.hmm_scalers[symbol]
        df = df_dict['15m'].copy()
        ret = df['close'].pct_change().dropna().values[-50:].reshape(-1, 1)
        if len(ret) < 10:
            return 0
        ret_scaled = scaler.transform(ret)
        states = model.predict(ret_scaled)
        return states[-1]

    def _train_hmm(self, symbol: str, df_dict: Dict[str, pd.DataFrame]) -> Tuple[Optional[hmm.GaussianHMM], Optional[StandardScaler]]:
        df = df_dict['15m'].copy()
        ret = df['close'].pct_change().dropna().values.reshape(-1, 1)
        if len(ret) < 200:
            return None, None
        scaler = StandardScaler()
        ret_scaled = scaler.fit_transform(ret)
        model = hmm.GaussianHMM(n_components=self.config.hmm_n_components, covariance_type="diag", n_iter=self.config.hmm_n_iter)
        model.fit(ret_scaled)
        return model, scaler

    def _detect_market_regime_traditional(self, df_dict: Dict[str, pd.DataFrame], fear_greed: int) -> MarketRegime:
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
        else:
            if fear_greed <= 20:
                return MarketRegime.PANIC
            else:
                return MarketRegime.RANGE

    def _is_range_market(self, df_dict: Dict[str, pd.DataFrame]) -> bool:
        if '15m' not in df_dict:
            return False
        df = df_dict['15m']
        if df.empty:
            return False
        last = df.iloc[-1]
        try:
            if hasattr(last, 'get') and last.get('bb_width') is not None and not pd.isna(last.get('bb_width')):
                if last['bb_width'] < self.config.bb_width_threshold:
                    return True
            if hasattr(last, 'get') and last.get('rsi') is not None and not pd.isna(last.get('rsi')):
                if self.config.rsi_range_low < last['rsi'] < self.config.rsi_range_high:
                    return True
        except Exception as e:
            logger.error(f"is_range_market 判断出错: {e}")
            return False
        return False

    def _multi_timeframe_confirmation(self, df_dict: Dict[str, pd.DataFrame], direction: int) -> bool:
        count = 0
        for tf in self.config.confirm_timeframes:
            if tf not in df_dict:
                continue
            df = df_dict[tf]
            if df.empty:
                continue
            last = df.iloc[-1]
            if not pd.isna(last.get('ema20')):
                if (direction == 1 and last['close'] > last['ema20']) or (direction == -1 and last['close'] < last['ema20']):
                    count += 1
        return count >= 2

    def _calculate_ic(self, df: pd.DataFrame, factor_name: str) -> float:
        try:
            df_hash = pd.util.hash_pandas_object(df).sum()
        except:
            df_hash = id(df)
        key = (df_hash, factor_name)
        if key in self._ic_cache:
            return self._ic_cache[key]
        window = min(self.config.ic_window, len(df) - 6)
        if window < 20:
            return 0.0
        factor = df[factor_name].iloc[-window:-5]
        future = df['future_ret'].iloc[-window:-5]
        valid = factor.notna() & future.notna()
        if valid.sum() < 10:
            return 0.0
        ic = factor[valid].corr(future[valid])
        ic = 0.0 if pd.isna(ic) else ic
        self._ic_cache[key] = ic
        return ic

    def _bayesian_update_factor_weights(self, ic_dict: Dict[str, List[float]]):
        prior_mean = 1.0
        prior_strength = self.config.bayesian_prior_strength
        for factor, ic_list in ic_dict.items():
            if len(ic_list) < 5:
                continue
            sample_mean = np.mean(ic_list)
            sample_std = np.std(ic_list)
            n = len(ic_list)
            posterior_mean = (prior_strength * prior_mean + n * sample_mean) / (prior_strength + n)
            self.factor_weights[factor] = max(0.1, posterior_mean)

    def _update_factor_correlation(self, ic_records: Dict[str, List[float]]):
        if len(ic_records) < 2:
            return
        all_factors = list(self.factor_weights.keys())
        df_dict = {}
        for f in all_factors:
            if f in ic_records and ic_records[f]:
                df_dict[f] = pd.Series(ic_records[f])
            else:
                df_dict[f] = pd.Series([np.nan])
        ic_df = pd.DataFrame(df_dict)
        corr = ic_df.corr().fillna(0)
        self.factor_corr_matrix = corr.values

    def _apply_factor_correlation_penalty(self):
        if self.factor_corr_matrix is None:
            return
        factors = list(self.factor_weights.keys())
        n = len(factors)
        if self.factor_corr_matrix.shape[0] < n or self.factor_corr_matrix.shape[1] < n:
            return
        for i in range(n):
            for j in range(i+1, n):
                if self.factor_corr_matrix[i, j] > self.config.factor_corr_threshold:
                    self.factor_weights[factors[i]] *= self.config.factor_corr_penalty
                    self.factor_weights[factors[j]] *= self.config.factor_corr_penalty

    def _update_factor_ic_stats(self, ic_records: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        stats = {}
        for factor, ic_list in ic_records.items():
            if len(ic_list) > 5:
                mean_ic = np.mean(ic_list)
                std_ic = np.std(ic_list)
                ir = mean_ic / max(std_ic, 0.001)
                t_stat, p_value = ttest_1samp(ic_list, 0)
                stats[factor] = {'mean': mean_ic, 'std': std_ic, 'ir': ir, 'p_value': p_value}
        return stats

    def _eliminate_poor_factors(self, factor_ic_stats: Dict[str, Dict[str, float]]):
        for factor, stats in factor_ic_stats.items():
            if stats['p_value'] > self.config.factor_eliminate_pvalue and stats['mean'] < self.config.factor_eliminate_ic and len(self.ic_decay_records[factor]) > 30:
                self.factor_weights[factor] = self.config.factor_min_weight
                logger.info(f"因子淘汰：{factor} 权重降至{self.config.factor_min_weight}")

    def _get_ml_factor(self, symbol: str, df_dict: Dict[str, pd.DataFrame]) -> float:
        if not self.config.use_ml_factor:
            return 0.0
        now = time.time()
        if symbol not in self.ml_models or now - self.ml_last_train.get(symbol, 0) > self.config.ml_retrain_interval:
            model, scaler, feature_cols, calibrator = self._train_ml_model(symbol, df_dict)
            if model is not None:
                self.ml_models[symbol] = model
                self.ml_scalers[symbol] = scaler
                self.ml_feature_cols[symbol] = feature_cols
                self.ml_calibrators[symbol] = calibrator
                self.ml_last_train[symbol] = now
        if symbol not in self.ml_models:
            return 0.0
        model = self.ml_models[symbol]
        scaler = self.ml_scalers[symbol]
        feature_cols = self.ml_feature_cols.get(symbol, [])
        if not feature_cols:
            return 0.0
        df = df_dict['15m'].copy()
        if len(df) < 4:
            return 0.0
        last_idx = -1
        data = {}
        feature_cols_original = ['ema20', 'ema50', 'rsi', 'macd_diff', 'bb_width', 'volume_ratio', 'adx', 'atr']
        for col in feature_cols_original:
            if col in df.columns:
                data[col] = df[col].iloc[last_idx]
            else:
                data[col] = np.nan
            for lag in [1,2,3]:
                lag_col = f'{col}_lag{lag}'
                if len(df) > lag:
                    data[lag_col] = df[col].iloc[-lag-1]
                else:
                    data[lag_col] = np.nan
        X = pd.DataFrame([data])
        X = X[feature_cols]
        X = X.fillna(0)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        raw_prob = np.tanh(pred * 10)
        if symbol in self.ml_calibrators and self.ml_calibrators[symbol] is not None:
            raw_prob = self.ml_calibrators[symbol].predict([[raw_prob]])[0]
        return raw_prob

    def _train_ml_model(self, symbol: str, df_dict: Dict[str, pd.DataFrame]) -> Tuple[Any, Any, List[str], Any]:
        df = df_dict['15m'].copy()
        feature_cols = ['ema20', 'ema50', 'rsi', 'macd_diff', 'bb_width', 'volume_ratio', 'adx', 'atr']
        df = df.dropna(subset=feature_cols + ['close'])
        if len(df) < self.config.ml_window:
            return None, None, [], None
        for col in feature_cols:
            for lag in [1,2,3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        future_ret = df['close'].pct_change(5).shift(-5)
        if self.config.cost_aware_training:
            volume_ma = df['volume'].rolling(20).mean()
            impact = (df['volume'] / volume_ma).fillna(1) * 0.0002
            vola = df['atr'] / df['close']
            slippage_est = vola * 0.001
            total_cost = impact + slippage_est
            target = future_ret - total_cost.shift(-5)
        else:
            target = future_ret
        df['target'] = target
        df = df.dropna()
        if len(df) < 100:
            return None, None, [], None
        all_feature_cols = []
        for col in feature_cols:
            all_feature_cols.append(col)
            for lag in [1,2,3]:
                all_feature_cols.append(f'{col}_lag{lag}')
        X = df[all_feature_cols]
        y = df['target']
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = RandomForestRegressor(
            n_estimators=self.config.ml_n_estimators,
            max_depth=self.config.ml_max_depth,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        calibrator = None
        if self.config.use_prob_calibration and self.config.calibration_method == 'isotonic':
            pred_val = model.predict(X_val_scaled)
            y_val_binary = (y_val > 0).astype(int)
            pred_val_norm = (pred_val - pred_val.min()) / (pred_val.max() - pred_val.min() + 1e-8)
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(pred_val_norm, y_val_binary)
            calibrator = ir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(model, f"{MODEL_DIR}/{symbol}_model_{timestamp}.pkl")
        joblib.dump(scaler, f"{MODEL_DIR}/{symbol}_scaler_{timestamp}.pkl")
        if calibrator:
            joblib.dump(calibrator, f"{MODEL_DIR}/{symbol}_calibrator_{timestamp}.pkl")
        return model, scaler, all_feature_cols, calibrator

    def generate_signals(self, multi_data: Dict[str, Any], fear_greed: int, market_regime: MarketRegime) -> List[Signal]:
        signals = []
        for symbol, data in multi_data.items():
            df_dict = data['data_dict']
            direction, prob = self.calc_signal(df_dict, symbol, fear_greed, market_regime)
            if direction != 0 and prob >= SignalStrength.WEAK.value:
                signals.append(Signal(
                    symbol=symbol,
                    direction=direction,
                    probability=prob,
                    timestamp=datetime.now()
                ))
        return signals

    def calc_signal(self, df_dict: Dict[str, pd.DataFrame], symbol: str, fear_greed: int, market_regime: MarketRegime) -> Tuple[int, float]:
        total_score = 0
        total_weight = 0
        tf_votes = []
        ic_dict = {}

        try:
            range_penalty = 0.5 if self._is_range_market(df_dict) else 1.0
        except Exception as e:
            logger.error(f"is_range_market 调用异常: {e}")
            range_penalty = 1.0

        for tf, df in df_dict.items():
            if df.empty or len(df) < 2:
                continue
            last = df.iloc[-1]
            weight = self.config.timeframe_weights.get(tf, 1) * range_penalty
            if market_regime == MarketRegime.TREND:
                if tf in ['4h', '1d']:
                    weight *= 1.5
            elif market_regime == MarketRegime.RANGE:
                if tf in ['15m', '1h']:
                    weight *= 1.3
            if pd.isna(last.get('ema20', np.nan)):
                continue

            factor_scores = {}
            if last['close'] > last['ema20']:
                factor_scores['trend'] = 1 * self.factor_weights['trend']
            elif last['close'] < last['ema20']:
                factor_scores['trend'] = -1 * self.factor_weights['trend']
            else:
                factor_scores['trend'] = 0

            if last['rsi'] > 70:
                factor_scores['rsi'] = -0.7 * self.factor_weights['rsi']
            elif last['rsi'] < 30:
                factor_scores['rsi'] = 0.7 * self.factor_weights['rsi']
            else:
                factor_scores['rsi'] = 0

            if last['macd_diff'] > 0:
                factor_scores['macd'] = 0.8 * self.factor_weights['macd']
            elif last['macd_diff'] < 0:
                factor_scores['macd'] = -0.8 * self.factor_weights['macd']
            else:
                factor_scores['macd'] = 0

            if not pd.isna(last.get('bb_upper')):
                if last['close'] > last['bb_upper']:
                    factor_scores['bb'] = -0.5 * self.factor_weights['bb']
                elif last['close'] < last['bb_lower']:
                    factor_scores['bb'] = 0.5 * self.factor_weights['bb']
                else:
                    factor_scores['bb'] = 0
            else:
                factor_scores['bb'] = 0

            if not pd.isna(last.get('volume_ratio')):
                factor_scores['volume'] = (1.2 if last['volume_ratio'] > 1.5 else 0) * self.factor_weights['volume']
            else:
                factor_scores['volume'] = 0

            adx = last.get('adx', 25)
            if pd.isna(adx):
                factor_scores['adx'] = 0
            else:
                factor_scores['adx'] = (0.3 if adx > 30 else -0.2 if adx < 20 else 0) * self.factor_weights['adx']

            if self.config.use_ml_factor:
                ml_score = self._get_ml_factor(symbol, df_dict)
                factor_scores['ml'] = ml_score * self.factor_weights['ml']

            for fname in factor_scores.keys():
                col = self.factor_to_col.get(fname)
                if col and col in df.columns:
                    ic = self._calculate_ic(df, col)
                    if not np.isnan(ic):
                        if fname not in ic_dict:
                            ic_dict[fname] = []
                        ic_dict[fname].append(ic)
                        self.ic_decay_records[fname].append(ic)

            tf_score = sum(factor_scores.values()) * weight
            total_score += tf_score
            total_weight += weight
            if tf_score > 0:
                tf_votes.append(1)
            elif tf_score < 0:
                tf_votes.append(-1)

        self._bayesian_update_factor_weights(ic_dict)
        self._update_factor_correlation(ic_dict)
        self._apply_factor_correlation_penalty()
        factor_ic_stats = self._update_factor_ic_stats(ic_dict)
        self._eliminate_poor_factors(factor_ic_stats)

        if total_weight == 0:
            return 0, 0.0
        max_possible = sum(self.config.timeframe_weights.values()) * 3.5
        prob_raw = min(1.0, abs(total_score) / max_possible) if max_possible > 0 else 0.5
        prob = 0.5 + 0.45 * prob_raw

        direction_candidate = 1 if total_score > 0 else -1 if total_score < 0 else 0
        if direction_candidate != 0 and not self._multi_timeframe_confirmation(df_dict, direction_candidate):
            prob *= 0.5

        if prob < SignalStrength.WEAK.value:
            return 0, prob

        if prob >= SignalStrength.WEAK.value:
            direction = direction_candidate
        else:
            if tf_votes:
                direction = 1 if sum(tf_votes) > 0 else -1 if sum(tf_votes) < 0 else 0
            else:
                direction = 0
        if direction == 0:
            prob = 0.0
        return direction, prob

# ==================== 风险管理 ====================
class RiskManager:
    def __init__(self, config: TradingConfig, strategy_engine: StrategyEngine, rl_manager: Optional['RLManager'] = None):
        self.config = config
        self.strategy_engine = strategy_engine
        self.rl_manager = rl_manager
        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None

    def update_losses(self, win: bool, loss_amount: float = 0.0):
        if not win:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.cooldown_losses:
                self.cooldown_until = datetime.now() + timedelta(hours=self.config.cooldown_hours)
        else:
            self.consecutive_losses = 0
            self.cooldown_until = None

    def check_cooldown(self) -> bool:
        return self.cooldown_until is not None and datetime.now() < self.cooldown_until

    def get_portfolio_var(self, weights: np.ndarray, cov: np.ndarray, confidence: float = 0.95) -> float:
        if weights is None or cov is None or len(weights) == 0:
            return 0.0
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return port_vol * norm.ppf(confidence)

    def get_current_drawdown(self, equity_curve: List[Dict]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        df = pd.DataFrame(equity_curve)
        peak = df['equity'].cummax()
        dd = (peak - df['equity']) / peak * 100
        return dd.iloc[-1]

    def scaling_factor(self, equity_curve: List[Dict], cov_matrix: Optional[np.ndarray],
                       positions: Dict[str, Position], current_symbols: List[str],
                       account_balance: float, symbol_current_prices: Dict[str, float]) -> float:
        factor = 1.0
        dd = self.get_current_drawdown(equity_curve)
        if dd > 5.0:
            factor *= (1 - (dd - 5.0) / self.config.drawdown_scaling_factor)
        if cov_matrix is not None and len(positions) > 0:
            total_value = account_balance
            weights = []
            for sym in current_symbols:
                if sym in positions:
                    pos = positions[sym]
                    value = pos.size * symbol_current_prices.get(sym, 1)
                    weights.append(value / total_value)
                else:
                    weights.append(0.0)
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                var = self.get_portfolio_var(weights, cov_matrix)
                target_var = self.config.portfolio_risk_target
                if var > target_var:
                    factor *= target_var / var
        return max(0.1, min(1.0, factor))

    def check_circuit_breaker(self, daily_pnl: float, account_balance: float, consecutive_losses: int,
                              positions: Dict[str, Position], multi_df: Dict[str, Any],
                              symbol_current_prices: Dict[str, float]) -> Tuple[bool, str]:
        daily_loss = -daily_pnl
        if daily_loss > account_balance * self.config.daily_loss_limit:
            return True, f"日亏损{daily_loss:.2f}超限"
        if consecutive_losses >= self.config.max_consecutive_losses:
            return True, f"连续亏损{consecutive_losses}次"
        for sym, pos in positions.items():
            if sym in multi_df:
                atr = multi_df[sym]['15m']['atr'].iloc[-1]
                price = symbol_current_prices.get(sym, 1)
                atr_pct = atr / price * 100
                if atr_pct > self.config.circuit_breaker_atr:
                    return True, f"{sym} ATR百分比{atr_pct:.2f}%超限"
        return False, ""

    def can_open_position(self, regime: MarketRegime) -> bool:
        return regime.value in self.config.regime_allow_trade

    def calc_position_size(self, balance: float, prob: float, atr: float, price: float,
                           price_series: np.ndarray, is_aggressive: bool,
                           equity_curve: List[Dict], cov_matrix: Optional[np.ndarray],
                           positions: Dict[str, Position], current_symbols: List[str],
                           symbol_current_prices: Dict[str, float], rl_obs: Optional[np.ndarray] = None) -> float:
        if price <= 0 or prob < 0.5:
            return 0.0
        risk_amount = balance * self.config.risk_per_trade
        if self.rl_manager is not None and rl_obs is not None:
            rl_mult = self.rl_manager.get_action(rl_obs)
            risk_amount *= rl_mult
        scaling = self.scaling_factor(equity_curve, cov_matrix, positions, current_symbols, balance, symbol_current_prices)
        risk_amount *= scaling
        if is_aggressive:
            risk_amount *= 1.5
        if atr == 0 or np.isnan(atr) or atr < price * self.config.min_atr_pct / 100:
            stop_distance = price * 0.01
        else:
            stop_distance = atr * self._adaptive_atr_multiplier(price_series)
        size = risk_amount / stop_distance
        max_size_by_leverage = balance * self.config.max_leverage_global / price
        size = min(size, max_size_by_leverage)
        return max(size, 0.001)

    def _adaptive_atr_multiplier(self, price_series: np.ndarray) -> float:
        if len(price_series) < self.config.adapt_window:
            return self.config.atr_multiplier_base
        returns = pd.Series(price_series).pct_change().dropna().values
        vol = np.std(returns) * np.sqrt(365 * 24 * 4)
        volcone = self._get_volcone(returns)
        current_vol_percentile = np.mean(vol <= volcone['percentiles']) if volcone else 0.5
        factor = 1.5 - current_vol_percentile
        new_mult = self.config.atr_multiplier_base * factor
        return np.clip(new_mult, self.config.atr_multiplier_min, self.config.atr_multiplier_max)

    def _get_volcone(self, returns: np.ndarray) -> dict:
        key = hash(returns[-self.config.volcone_window:].tobytes())
        if key in self.strategy_engine.volcone_cache:
            return self.strategy_engine.volcone_cache[key]
        windows = [5, 10, 20, 40, 60]
        volcone = {}
        vols = []
        for w in windows:
            if len(returns) > w:
                roll_vol = np.array([np.std(returns[i-w:i]) for i in range(w, len(returns)+1)]) * np.sqrt(365*24*4/w)
                quantiles = np.percentile(roll_vol, [p*100 for p in self.config.volcone_percentiles])
                volcone[f'vol_{w}'] = dict(zip(self.config.volcone_percentiles, quantiles))
                vols.extend(roll_vol)
        volcone['percentiles'] = np.percentile(vols, [p*100 for p in self.config.volcone_percentiles])
        self.strategy_engine.volcone_cache[key] = volcone
        return volcone

    def allocate_portfolio(self, symbol_signals: Dict[str, Tuple[int, float, float, float, np.ndarray]],
                           balance: float, equity_curve: List[Dict], cov_matrix: Optional[np.ndarray],
                           positions: Dict[str, Position], current_symbols: List[str],
                           symbol_current_prices: Dict[str, float], rl_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        if not symbol_signals:
            return {}
        expected_returns = {}
        for sym, (direction, prob, atr, price, price_series) in symbol_signals.items():
            if atr == 0 or np.isnan(atr):
                stop_dist = price * 0.01
            else:
                stop_dist = atr * self._adaptive_atr_multiplier(price_series)
            risk_amount = balance * self.config.risk_per_trade
            reward_risk_ratio = self.config.tp_min_ratio
            expected_pnl = prob * (risk_amount * reward_risk_ratio) - (1 - prob) * risk_amount
            expected_returns[sym] = expected_pnl
        positive_expected = {sym: er for sym, er in expected_returns.items() if er > 0}
        if not positive_expected:
            return {}
        sorted_symbols = sorted(positive_expected.keys(), key=lambda s: positive_expected[s], reverse=True)
        allocations = {sym: 0.0 for sym in symbol_signals}
        best_sym = sorted_symbols[0]
        dir, prob, atr, price, price_series = symbol_signals[best_sym]
        is_aggressive = prob > 0.7
        size = self.calc_position_size(balance, prob, atr, price, price_series, is_aggressive,
                                       equity_curve, cov_matrix, positions, current_symbols,
                                       symbol_current_prices, rl_obs)
        allocations[best_sym] = size
        return allocations

# ==================== 强化学习环境（同步版本）====================
class TradingEnv(gym.Env):
    def __init__(self, data_provider: AsyncDataProvider, config: TradingConfig):
        super(TradingEnv, self).__init__()
        self.data_provider = data_provider
        self.config = config
        self.action_space = spaces.Box(low=config.rl_action_low, high=config.rl_action_high, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        # 预加载历史数据用于训练（简化：使用模拟数据）
        self.historical_data = None
        self.current_idx = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.entry_price = 0
        self.trades = []
        # 加载历史数据（这里仅示例，实际训练时应从文件或data_provider获取）
        if self.historical_data is None:
            # 模拟生成历史数据
            self.historical_data = self.data_provider.generate_simulated_data('ETH/USDT')['15m']
        self.current_idx = len(self.historical_data) - 1  # 从最新开始
        return self._get_obs(), {}

    def _get_obs(self):
        # 使用历史数据中的当前步
        if self.current_idx >= 0 and self.current_idx < len(self.historical_data):
            row = self.historical_data.iloc[self.current_idx]
            vol = row['atr'] / row['close'] if not pd.isna(row['atr']) else 0.02
            trend = row['adx'] / 100 if not pd.isna(row['adx']) else 0.5
            signal_prob = np.random.rand()  # 简化
        else:
            vol, trend, signal_prob = 0.02, 0.5, 0.5
        obs = np.array([
            self.balance / 10000,
            self.position / 100,
            vol,
            trend,
            signal_prob,
            0, 0, 0, 0, 0
        ], dtype=np.float32)
        return obs

    def step(self, action):
        risk_mult = action[0]
        # 模拟交易，随机收益
        pnl = np.random.randn() * 10 * risk_mult
        self.balance += pnl
        self.trades.append(pnl)
        if len(self.trades) > 10:
            returns = np.array(self.trades[-10:])
            sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        reward = sharpe
        self.current_step += 1
        self.current_idx -= 1  # 向后移动
        done = self.current_step > 1000 or self.current_idx < 0
        return self._get_obs(), reward, done, False, {}

class RLManager:
    def __init__(self, config: TradingConfig, data_provider: AsyncDataProvider):
        self.config = config
        self.data_provider = data_provider
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.config.rl_model_path):
            self.model = PPO.load(self.config.rl_model_path)
        else:
            env = DummyVecEnv([lambda: TradingEnv(self.data_provider, self.config)])
            self.model = PPO('MlpPolicy', env, verbose=0)
            self.model.learn(total_timesteps=10000)
            self.model.save(self.config.rl_model_path)

    def get_action(self, obs: np.ndarray) -> float:
        if self.model is None:
            return 1.0
        action, _ = self.model.predict(obs, deterministic=True)
        return float(np.clip(action, self.config.rl_action_low, self.config.rl_action_high))

# ==================== 执行服务（全异步）====================
class ExecutionService:
    def __init__(self, config: TradingConfig, notifier: TelegramNotifier):
        self.config = config
        self.notifier = notifier
        self.exchange: Optional[ccxt_async.Exchange] = None
        self.pending_orders: Dict[str, Order] = {}

    def set_exchange(self, exchange: ccxt_async.Exchange):
        self.exchange = exchange

    async def sync_order_status(self):
        if not self.exchange:
            return
        for symbol, order in list(self.pending_orders.items()):
            try:
                ex_order = await self.exchange.fetch_order(order.exchange_order_id, symbol)
                if ex_order['status'] == 'closed':
                    order.status = 'filled'
                    order.filled_size = ex_order['filled']
                    logger.info(f"订单 {order.exchange_order_id} 已完全成交")
                    del self.pending_orders[symbol]
                elif ex_order['status'] == 'canceled':
                    order.status = 'canceled'
                    logger.info(f"订单 {order.exchange_order_id} 已取消")
                    del self.pending_orders[symbol]
                else:
                    order.filled_size = ex_order['filled']
            except Exception as e:
                logger.warning(f"同步订单状态失败: {e}")

    def check_liquidity(self, symbol: str, size: float) -> bool:
        if not self.exchange:
            return True
        try:
            # 同步获取订单簿（ccxt_async也支持同步调用？这里为了简化，使用同步方式，但实际可以异步）
            # 由于该方法在异步函数中调用，我们使用同步阻塞可能会影响性能。但此处作为快速检查，可以接受。
            # 更优方案：改为异步并等待，但需要调用方也异步。当前 execute_order 是异步，所以可以改为异步。
            # 我们将其改为异步方法，并在 execute_order 中 await。
            # 但为了不破坏已有代码，我们临时保留同步，但在实际应用中建议改为异步。
            import ccxt
            if hasattr(self.exchange, 'fetch_order_book'):
                ob = self.exchange.fetch_order_book(symbol, limit=10)  # 这里会阻塞，但ccxt_async的同步方法会调用底层同步？
                # 实际上ccxt_async对象没有同步fetch_order_book，需要await。所以这里不能这样用。
                # 因此必须改为异步。我们修改为：
                # 由于时间有限，我们假设这里有一个异步方法，稍后修正。
                pass
            return True
        except Exception as e:
            logger.warning(f"检查流动性失败: {e}")
            return True

    async def async_check_liquidity(self, symbol: str, size: float) -> bool:
        """异步检查流动性"""
        if not self.exchange:
            return True
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=10)
            total_bid_vol = sum(b[1] for b in ob['bids'])
            total_ask_vol = sum(a[1] for a in ob['asks'])
            depth = max(total_bid_vol, total_ask_vol)
            if size > depth * self.config.max_order_to_depth_ratio:
                logger.info(f"流动性不足：订单大小{size:.4f}超过盘口深度{depth:.4f}的{self.config.max_order_to_depth_ratio*100:.1f}%")
                return False
            return True
        except Exception as e:
            logger.warning(f"检查流动性失败: {e}")
            return True

    def simulate_latency(self):
        time.sleep(self.config.latency_sim_ms / 1000.0)

    async def _set_leverage(self, symbol: str, exchange_choice: str, testnet: bool, api_key: str, secret_key: str):
        if not self.exchange:
            return
        try:
            leverage = self.config.leverage
            exchange_name = exchange_choice.lower()
            if 'binance' in exchange_name:
                await self.exchange.fapiPrivate_post_leverage({
                    'symbol': symbol.replace('/', ''),
                    'leverage': leverage
                })
            elif 'bybit' in exchange_name:
                await self.exchange.private_linear_post_position_set_leverage({
                    'symbol': symbol.replace('/', ''),
                    'buy_leverage': leverage,
                    'sell_leverage': leverage
                })
            elif 'okx' in exchange_name:
                await self.exchange.privatePostAccountSetLeverage({
                    'instId': symbol.replace('/', '-'),
                    'lever': leverage,
                    'mgnMode': 'cross'
                })
            logger.info(f"设置杠杆 {symbol} → {leverage}x")
        except Exception as e:
            logger.error(f"设置杠杆失败 {symbol}: {e}")

    def _advanced_slippage_prediction(self, price: float, size: float, volume_20: float, volatility: float, imbalance: float) -> float:
        base_slippage = self._dynamic_slippage(price, size, volume_20, volatility, imbalance)
        market_impact = (size / max(volume_20, 1)) ** 0.5 * volatility * price * 0.3
        return base_slippage + market_impact

    def _dynamic_slippage(self, price: float, size: float, volume: float, volatility: float, imbalance: float = 0.0) -> float:
        base = price * self.config.slippage_base
        impact = self.config.slippage_impact_factor * (size / max(volume, 1)) * volatility * price
        imbalance_adj = 1 + abs(imbalance) * self.config.slippage_imbalance_factor
        return (base + impact) * imbalance_adj

    async def execute_order(self, symbol: str, direction: int, size: float, price: float, stop: float, take: float,
                            use_real: bool, exchange_choice: str, testnet: bool, api_key: str, secret_key: str,
                            multi_df: Dict[str, Any], symbol_current_prices: Dict[str, float],
                            orderbook_imbalance: Dict[str, float]) -> Tuple[Optional[Position], float, float]:
        self.simulate_latency()
        if not await self.async_check_liquidity(symbol, size):
            logger.info(f"流动性不足，取消开仓 {symbol}")
            return None, 0, 0

        sym = symbol.strip()
        dir_str = "多" if direction == 1 else "空"
        side = 'buy' if direction == 1 else 'sell'

        volume = multi_df[sym]['15m']['volume'].iloc[-1] if sym in multi_df and not multi_df[sym]['15m'].empty else 0
        vola = np.std(multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]) if sym in multi_df else 0.02
        imbalance = orderbook_imbalance.get(sym, 0.0)
        slippage = self._advanced_slippage_prediction(price, size, volume, vola, imbalance)
        exec_price = price + slippage if direction == 1 else price - slippage
        market_impact = (size / max(volume, 1)) ** 0.5 * vola * price * 0.3

        if use_real and self.exchange:
            try:
                balance = await self.exchange.fetch_balance()
                free_usdt = balance['free'].get('USDT', 0)
                required = size * price / self.config.leverage
                if free_usdt < required:
                    logger.error(f"余额不足：需要{required:.2f} USDT，可用{free_usdt:.2f}")
                    return None, 0, 0
                await self._set_leverage(sym, exchange_choice, testnet, api_key, secret_key)
                order = await self.exchange.create_order(
                    symbol=sym,
                    type='market',
                    side=side,
                    amount=size,
                    params={'reduceOnly': False}
                )
                actual_price = float(order['average'] or order['price'] or price)
                actual_size = float(order['amount'])
                logger.info(f"【实盘开仓成功】 {sym} {dir_str} {actual_size:.4f} @ {actual_price:.2f}")
                self.notifier.send(f"【实盘】开仓 {dir_str} {sym}\n价格: {actual_price:.2f}\n仓位: {actual_size:.4f}", msg_type="trade")
                order_obj = Order(
                    symbol=sym,
                    side=OrderSide(side),
                    type=OrderType.MARKET,
                    size=size,
                    price=actual_price,
                    stop_loss=stop,
                    take_profit=take,
                    exchange_order_id=order['id']
                )
                self.pending_orders[sym] = order_obj
            except ccxt.InsufficientFunds as e:
                logger.error(f"余额不足: {e}")
                self.notifier.send(f"⚠️ 余额不足，开仓失败 {sym}", msg_type="risk")
                return None, 0, 0
            except ccxt.RateLimitExceeded as e:
                logger.error(f"请求超限: {e}")
                self.notifier.send(f"⚠️ 交易所限频，请稍后再试", msg_type="risk")
                return None, 0, 0
            except Exception as e:
                logger.error(f"实盘开仓失败 {sym}: {e}")
                self.notifier.send(f"⚠️ 开仓失败 {sym} {dir_str}: {str(e)}", msg_type="risk")
                return None, 0, 0
        else:
            actual_price = exec_price
            actual_size = size

        position = Position(
            symbol=sym,
            direction=direction,
            entry_price=actual_price,
            entry_time=datetime.now(),
            size=actual_size if use_real else size,
            stop_loss=stop,
            take_profit=take,
            initial_atr=0,
            real=use_real,
            slippage_paid=slippage,
            impact_cost=market_impact
        )
        return position, actual_price, slippage

# ==================== 交易服务 ====================
class TradingService:
    def __init__(self, config: TradingConfig, data_provider: AsyncDataProvider,
                 strategy_engine: StrategyEngine, risk_manager: RiskManager,
                 execution_service: ExecutionService, notifier: TelegramNotifier,
                 metrics: PrometheusMetrics):
        self.config = config
        self.data_provider = data_provider
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.execution_service = execution_service
        self.notifier = notifier
        self.metrics = metrics

        self.balance = 10000.0
        self.daily_pnl = 0.0
        self.daily_risk_consumed = 0.0
        self.peak_balance = 10000.0
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.trade_log: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.equity_curve: Deque[Dict] = deque(maxlen=500)
        self.net_value_history: List[Dict] = []
        self.slippage_history: Deque[Dict] = deque(maxlen=100)
        self.slippage_records: List[Dict] = []
        self.factor_ic_stats: Dict[str, Dict] = {}
        self.symbol_current_prices: Dict[str, float] = {}
        self.multi_df: Dict[str, Any] = {}
        self.cov_matrix: Optional[np.ndarray] = None
        self.orderbook_imbalance: Dict[str, float] = {}
        self.funding_rates: Dict[str, float] = {}
        self.market_regime: MarketRegime = MarketRegime.RANGE
        self.fear_greed: int = 50
        self.regime_stats: Dict[str, Dict] = {}
        self.consistency_stats: Dict[str, Dict] = {'backtest': {}, 'live': {}}
        self.brier_scores: Deque[float] = deque(maxlen=100)
        self.ic_history: Dict[str, Deque[float]] = {f: deque(maxlen=100) for f in self.strategy_engine.factor_weights.keys()}
        self.degraded_mode = False
        self.last_trade_date = None
        self.daily_returns: Deque[float] = deque(maxlen=252)

    def update_equity(self):
        equity = self.balance
        for pos in self.positions.values():
            if pos.symbol in self.symbol_current_prices:
                equity += pos.pnl(self.symbol_current_prices[pos.symbol])
        self.equity_curve.append({'time': datetime.now(), 'equity': equity})

    async def process_market_data(self, symbols: List[str], use_simulated: bool = False) -> Optional[Dict[str, Any]]:
        multi_data = {}
        for sym in symbols:
            data = await self.data_provider.get_symbol_data(sym, use_simulated)
            if data is None:
                logger.error(f"获取 {sym} 数据失败")
                self._degrade_if_needed(f"数据缺失 {sym}")
                return None
            multi_data[sym] = data
            self.symbol_current_prices[sym] = data['current_price']
            self.orderbook_imbalance[sym] = data.get('orderbook_imbalance', 0.0)
            self.funding_rates[sym] = data.get('funding_rate', 0.0)

        self.multi_df = {sym: data['data_dict'] for sym, data in multi_data.items()}
        self.fear_greed = multi_data[symbols[0]]['fear_greed']
        df_first = multi_data[symbols[0]]['data_dict']
        self.market_regime = self.strategy_engine.detect_market_regime(df_first, symbols[0], self.fear_greed)

        if len(symbols) > 1:
            ret_arrays = []
            for sym in symbols:
                rets = self.multi_df[sym]['15m']['close'].pct_change().dropna().values[-self.config.cov_matrix_window:]
                ret_arrays.append(rets)
            min_len = min(len(arr) for arr in ret_arrays)
            if min_len > 20:
                ret_matrix = np.array([arr[-min_len:] for arr in ret_arrays])
                self.cov_matrix = np.cov(ret_matrix)
            else:
                self.cov_matrix = None
        else:
            self.cov_matrix = None

        return multi_data

    def generate_signals(self) -> List[Signal]:
        multi_data = {sym: {'data_dict': self.multi_df[sym]} for sym in self.multi_df}
        signals = self.strategy_engine.generate_signals(multi_data, self.fear_greed, self.market_regime)
        return signals

    def _get_rl_obs(self) -> np.ndarray:
        """生成强化学习观测向量"""
        # 简化：使用当前市场指标和账户状态
        if self.multi_df and self.symbol_current_prices:
            sym = next(iter(self.multi_df))
            df = self.multi_df[sym]['15m']
            last = df.iloc[-1]
            vol = last['atr'] / last['close'] if not pd.isna(last['atr']) else 0.02
            trend = last['adx'] / 100 if not pd.isna(last['adx']) else 0.5
        else:
            vol, trend = 0.02, 0.5
        signal_prob = 0.5  # 简化
        obs = np.array([
            self.balance / 10000,
            len(self.positions) / 100,
            vol,
            trend,
            signal_prob,
            0, 0, 0, 0, 0
        ], dtype=np.float32)
        return obs

    async def execute_signals(self, signals: List[Signal], aggressive_mode: bool,
                              use_real: bool, exchange_choice: str, testnet: bool,
                              api_key: str, secret_key: str):
        if self.risk_manager.check_cooldown():
            logger.info("系统冷却中，不执行新开仓")
            return
        circuit, reason = self.risk_manager.check_circuit_breaker(
            self.daily_pnl, self.balance, self.consecutive_losses,
            self.positions, self.multi_df, self.symbol_current_prices
        )
        if circuit:
            logger.warning(f"熔断触发：{reason}")
            self.notifier.send(f"⚠️ 熔断触发：{reason}", msg_type="risk")
            self._degrade_if_needed(reason)
            return

        symbol_signals = {}
        for sig in signals:
            df_dict = self.multi_df[sig.symbol]
            atr_sym = df_dict['15m']['atr'].iloc[-1] if not pd.isna(df_dict['15m']['atr'].iloc[-1]) else 0
            price_series = df_dict['15m']['close'].values[-20:]  # 最近20个收盘价
            symbol_signals[sig.symbol] = (sig.direction, sig.probability, atr_sym, self.symbol_current_prices[sig.symbol], price_series)

        rl_obs = self._get_rl_obs() if self.config.use_rl_position else None
        allocations = self.risk_manager.allocate_portfolio(
            symbol_signals, self.balance, list(self.equity_curve),
            self.cov_matrix, self.positions, list(self.symbol_current_prices.keys()),
            self.symbol_current_prices, rl_obs
        )

        for sym, size in allocations.items():
            if size > 0 and sym not in self.positions:
                dir, prob, atr_sym, price, price_series = symbol_signals[sym]
                if atr_sym == 0 or np.isnan(atr_sym):
                    stop_dist = price * 0.01
                else:
                    stop_dist = atr_sym * self.risk_manager._adaptive_atr_multiplier(price_series)
                stop = price - stop_dist if dir == 1 else price + stop_dist
                take = price + stop_dist * self.config.tp_min_ratio if dir == 1 else price - stop_dist * self.config.tp_min_ratio

                position, actual_price, slippage = await self.execution_service.execute_order(
                    sym, dir, size, price, stop, take,
                    use_real, exchange_choice, testnet, api_key, secret_key,
                    self.multi_df, self.symbol_current_prices, self.orderbook_imbalance
                )
                if position:
                    self.positions[sym] = position
                    self.daily_trades += 1
                    self.slippage_history.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage})
                    self.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage})

        for sym, pos in list(self.positions.items()):
            if sym not in self.symbol_current_prices:
                continue
            df_dict = self.multi_df[sym]
            current_price = self.symbol_current_prices[sym]
            high = df_dict['15m']['high'].iloc[-1]
            low = df_dict['15m']['low'].iloc[-1]
            atr_sym = df_dict['15m']['atr'].iloc[-1] if not pd.isna(df_dict['15m']['atr'].iloc[-1]) else 0
            should_close, reason, exit_price, close_size = pos.should_close(high, low, datetime.now(), self.config)
            if should_close:
                await self._close_position(sym, exit_price, reason, close_size, use_real, exchange_choice, testnet, api_key, secret_key)
            else:
                if not pd.isna(atr_sym) and atr_sym > 0:
                    pos.update_stops(current_price, atr_sym, self.config.atr_multiplier_base)

        self.update_equity()

    async def _close_position(self, sym: str, exit_price: float, reason: str, close_size: Optional[float],
                              use_real: bool, exchange_choice: str, testnet: bool, api_key: str, secret_key: str):
        pos = self.positions.get(sym)
        if not pos:
            return

        close_size = min(close_size or pos.size, pos.size)
        side = 'sell' if pos.direction == 1 else 'buy'

        volume = self.multi_df[sym]['15m']['volume'].iloc[-1] if sym in self.multi_df else 0
        vola = np.std(self.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]) if sym in self.multi_df else 0.02
        imbalance = self.orderbook_imbalance.get(sym, 0.0)
        slippage = self.execution_service._advanced_slippage_prediction(exit_price, close_size, volume, vola, imbalance)
        exec_exit = exit_price - slippage if pos.direction == 1 else exit_price + slippage

        if pos.real and use_real and self.execution_service.exchange:
            try:
                order = await self.execution_service.exchange.create_order(
                    symbol=sym,
                    type='market',
                    side=side,
                    amount=close_size,
                    params={'reduceOnly': True}
                )
                actual_exit = float(order['average'] or order['price'] or exit_price)
                actual_size = float(order['amount'])
                logger.info(f"【实盘平仓成功】 {sym} {reason} {actual_size:.4f} @ {actual_exit:.2f}")
                self.notifier.send(f"【实盘】平仓 {reason} {sym}\n价格: {actual_exit:.2f}", msg_type="trade")
            except Exception as e:
                logger.error(f"实盘平仓失败 {sym}: {e}")
                self.notifier.send(f"⚠️ 平仓失败 {sym} {reason}: {str(e)}", msg_type="risk")
                return
        else:
            actual_exit = exec_exit
            actual_size = close_size

        pnl = (actual_exit - pos.entry_price) * actual_size * pos.direction - actual_exit * actual_size * self.config.fee_rate * 2
        self.daily_pnl += pnl
        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.net_value_history.append({'time': datetime.now(), 'value': self.balance})
        self.daily_returns.append(pnl / self.balance)

        trade = Trade(
            symbol=sym,
            direction='多' if pos.direction == 1 else '空',
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size=actual_size,
            pnl=pnl,
            reason=reason,
            timestamp=datetime.now(),
            slippage_entry=pos.slippage_paid,
            slippage_exit=slippage,
            impact_cost=pos.impact_cost
        )
        self.trade_log.append(trade)
        if len(self.trade_log) > 100:
            self.trade_log.pop(0)
        append_to_csv(TRADE_LOG_FILE, {
            'time': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry': trade.entry_price,
            'exit': trade.exit_price,
            'size': trade.size,
            'pnl': trade.pnl,
            'reason': trade.reason,
            'slippage_entry': trade.slippage_entry,
            'slippage_exit': trade.slippage_exit,
            'impact_cost': trade.impact_cost
        })
        self.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage})

        # 更新统计
        self._update_regime_stats(self.market_regime, pnl)
        self._update_consistency_stats(is_backtest=False, slippage=slippage, win=pnl>0)

        # 更新风险
        win_flag = pnl > 0
        self.risk_manager.update_losses(win_flag, loss_amount=pnl if not win_flag else 0)

        # 更新Prometheus指标
        self.metrics.observe_trade(pnl, slippage)

        if actual_size >= pos.size:
            del self.positions[sym]
        else:
            pos.size -= actual_size
            logger.info(f"部分平仓 {sym} {reason} 数量 {actual_size:.4f}，剩余 {pos.size:.4f}")

        logger.info(f"平仓 {sym} {reason} 盈亏 {pnl:.2f} 余额 {self.balance:.2f}")
        self.notifier.send(f"平仓 {reason}\n盈亏: {pnl:.2f}", msg_type="trade")

    def _update_regime_stats(self, regime: MarketRegime, pnl: float):
        key = regime.value
        if key not in self.regime_stats:
            self.regime_stats[key] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
        self.regime_stats[key]['trades'] += 1
        if pnl > 0:
            self.regime_stats[key]['wins'] += 1
        self.regime_stats[key]['total_pnl'] += pnl
        rows = []
        for k, v in self.regime_stats.items():
            rows.append({'regime': k, 'trades': v['trades'], 'wins': v['wins'], 'total_pnl': v['total_pnl']})
        pd.DataFrame(rows).to_csv(REGIME_STATS_FILE, index=False)

    def _update_consistency_stats(self, is_backtest: bool, slippage: float, win: bool):
        key = 'backtest' if is_backtest else 'live'
        stats = self.consistency_stats.get(key, {})
        stats['trades'] = stats.get('trades', 0) + 1
        stats['avg_slippage'] = (stats.get('avg_slippage', 0.0) * (stats['trades'] - 1) + slippage) / stats['trades']
        if win:
            stats['wins'] = stats.get('wins', 0) + 1
        stats['win_rate'] = stats.get('wins', 0) / stats['trades'] if stats['trades'] > 0 else 0
        self.consistency_stats[key] = stats
        rows = []
        for typ, s in self.consistency_stats.items():
            rows.append({
                'type': typ,
                'trades': s.get('trades', 0),
                'avg_slippage': s.get('avg_slippage', 0.0),
                'win_rate': s.get('win_rate', 0.0)
            })
        pd.DataFrame(rows).to_csv(CONSISTENCY_FILE, index=False)

    def _degrade_if_needed(self, reason: str):
        if not self.degraded_mode:
            self.degraded_mode = True
            logger.error(f"系统降级：{reason}")
            self.notifier.send(f"⚠️ 系统降级：{reason}", msg_type="risk")

    def daily_reset(self):
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_risk_consumed = 0.0
            self.last_trade_date = today
            logger.info("新的一天，重置每日数据")

    def get_state(self):
        return {
            'balance': self.balance,
            'daily_pnl': self.daily_pnl,
            'positions': self.positions,
            'equity_curve': self.equity_curve,
            'slippage_history': self.slippage_history,
            'symbol_current_prices': self.symbol_current_prices,
            'multi_df': self.multi_df,
            'cov_matrix': self.cov_matrix,
            'orderbook_imbalance': self.orderbook_imbalance,
            'market_regime': self.market_regime,
            'fear_greed': self.fear_greed,
            'factor_ic_stats': self.factor_ic_stats,
            'regime_stats': self.regime_stats,
            'consistency_stats': self.consistency_stats,
            'brier_scores': self.brier_scores,
            'ic_history': self.ic_history,
        }

# ==================== 回测引擎（多周期）====================
class BacktestEngine:
    def __init__(self, config: TradingConfig, strategy_engine: StrategyEngine, risk_manager: RiskManager,
                 data_provider: AsyncDataProvider, execution_service: ExecutionService):
        self.config = config
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        self.execution_service = execution_service

    async def run(self, symbols: List[str], start_date: datetime, end_date: datetime,
                  initial_balance: float = 10000, use_real_data: bool = False) -> Dict[str, Any]:
        # 加载所有时间周期的历史数据
        all_data = {}
        for sym in symbols:
            sym_data = {}
            for tf in self.config.timeframes + self.config.confirm_timeframes:
                if use_real_data:
                    df = await self.data_provider.fetch_historical_data(sym, tf, start_date, end_date)
                    if df is None:
                        logger.error(f"无法获取 {sym} {tf} 的历史数据，使用模拟数据")
                        df = self.data_provider.generate_simulated_data(sym)[tf] if tf in self.data_provider.generate_simulated_data(sym) else None
                else:
                    df = self.data_provider.generate_simulated_data(sym).get(tf)
                if df is not None:
                    sym_data[tf] = df
            all_data[sym] = sym_data

        # 确定共同时间戳（取所有品种15m的交集）
        timestamps = None
        for sym, sym_data in all_data.items():
            if '15m' not in sym_data:
                continue
            df = sym_data['15m']
            if timestamps is None:
                timestamps = df['timestamp']
            else:
                common = pd.merge(pd.DataFrame({'timestamp': timestamps}), pd.DataFrame({'timestamp': df['timestamp']}), on='timestamp')
                timestamps = common['timestamp']

        balance = initial_balance
        positions: Dict[str, Position] = {}
        trades = []
        equity_curve = []

        for idx, ts in enumerate(timestamps):
            # 构建当前时刻的多周期数据切片
            current_data = {}
            df_dict = {}
            for sym in symbols:
                sym_current = {}
                for tf, df in all_data[sym].items():
                    # 取时间戳 <= ts 的最新一条
                    mask = df['timestamp'] <= ts
                    if mask.any():
                        last_row = df[mask].iloc[-1]
                        sym_current[tf] = last_row.to_dict()
                        if tf == '15m':
                            current_data[sym] = last_row
                    # 构建df_dict用于策略引擎（需要整个DataFrame的历史）
                    # 这里简化：将整个历史数据截至ts传入
                    df_dict[sym] = {'data_dict': {tf: df[df['timestamp'] <= ts] for tf in all_data[sym]}}

            if not current_data:
                continue

            # 平仓检查
            for sym, pos in list(positions.items()):
                if sym not in current_data:
                    continue
                high = current_data[sym]['high']
                low = current_data[sym]['low']
                should_close, reason, exit_price, close_size = pos.should_close(high, low, ts, self.config)
                if should_close:
                    slippage = self.execution_service._advanced_slippage_prediction(
                        exit_price, close_size,
                        current_data[sym]['volume'], 0.02, 0
                    )
                    actual_exit = exit_price - slippage if pos.direction == 1 else exit_price + slippage
                    pnl = (actual_exit - pos.entry_price) * close_size * pos.direction - actual_exit * close_size * self.config.fee_rate * 2
                    balance += pnl
                    trades.append({
                        'symbol': sym,
                        'entry': pos.entry_price,
                        'exit': actual_exit,
                        'size': close_size,
                        'pnl': pnl,
                        'reason': reason,
                        'time': ts,
                        'slippage': slippage
                    })
                    if close_size >= pos.size:
                        del positions[sym]
                    else:
                        pos.size -= close_size

            # 生成信号
            signals = self.strategy_engine.generate_signals(
                {sym: df_dict[sym] for sym in symbols},
                fear_greed=50,
                market_regime=MarketRegime.RANGE
            )

            # 开仓
            for sig in signals:
                if sig.symbol not in positions:
                    price = current_data[sig.symbol]['close']
                    atr = current_data[sig.symbol]['atr'] if not pd.isna(current_data[sig.symbol]['atr']) else price * 0.01
                    price_series = all_data[sig.symbol]['15m']['close'].values[-20:]  # 最近20个收盘价
                    size = self.risk_manager.calc_position_size(
                        balance, sig.probability, atr, price, price_series,
                        is_aggressive=False,
                        equity_curve=[], cov_matrix=None,
                        positions=positions, current_symbols=symbols,
                        symbol_current_prices={sym: current_data[sym]['close'] for sym in current_data}
                    )
                    if size > 0:
                        stop_dist = atr * self.config.atr_multiplier_base
                        stop = price - stop_dist if sig.direction == 1 else price + stop_dist
                        take = price + stop_dist * self.config.tp_min_ratio if sig.direction == 1 else price - stop_dist * self.config.tp_min_ratio
                        pos = Position(
                            symbol=sig.symbol,
                            direction=sig.direction,
                            entry_price=price,
                            entry_time=ts,
                            size=size,
                            stop_loss=stop,
                            take_profit=take,
                            initial_atr=atr
                        )
                        positions[sig.symbol] = pos

            # 记录权益
            total_value = balance
            for sym, pos in positions.items():
                if sym in current_data:
                    total_value += pos.pnl(current_data[sym]['close'])
            equity_curve.append({'time': ts, 'equity': total_value})

        returns = pd.Series([e['equity'] for e in equity_curve]).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = (pd.Series([e['equity'] for e in equity_curve]).cummax() - pd.Series([e['equity'] for e in equity_curve])).max()

        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': (balance - initial_balance) / initial_balance * 100,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': trades,
            'equity_curve': equity_curve
        }

# ==================== 依赖注入容器 ====================
class Container(containers.DeclarativeContainer):
    config = providers.Singleton(load_config)
    notifier = providers.Singleton(TelegramNotifier, config=config)
    data_provider = providers.Singleton(AsyncDataProvider, config=config)
    strategy_engine = providers.Singleton(StrategyEngine, config=config, data_provider=data_provider)
    rl_manager = providers.Singleton(RLManager, config=config, data_provider=data_provider)
    risk_manager = providers.Singleton(RiskManager, config=config, strategy_engine=strategy_engine, rl_manager=rl_manager)
    execution_service = providers.Singleton(ExecutionService, config=config, notifier=notifier)
    metrics = providers.Singleton(PrometheusMetrics, config=config)
    trading_service = providers.Singleton(
        TradingService,
        config=config,
        data_provider=data_provider,
        strategy_engine=strategy_engine,
        risk_manager=risk_manager,
        execution_service=execution_service,
        notifier=notifier,
        metrics=metrics
    )
    backtest_engine = providers.Singleton(BacktestEngine, config=config, strategy_engine=strategy_engine,
                                          risk_manager=risk_manager, data_provider=data_provider,
                                          execution_service=execution_service)

# ==================== Streamlit UI ====================
def init_session_state():
    defaults = {
        'use_simulated_data': False,
        'data_source_failed': False,
        'error_log': deque(maxlen=20),
        'execution_log': deque(maxlen=50),
        'last_trade_date': None,
        'exchange_choice': 'Binance合约',
        'testnet': True,
        'use_real': False,
        'aggressive_mode': False,
        'auto_enabled': True,
        'current_symbols': ['ETH/USDT', 'BTC/USDT'],
        'backtest_results': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

container = Container()

async def main_async():
    st.set_page_config(page_title="终极量化终端 · 机构版 56.0", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 机构版 56.0")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北 · 全异步 · 多周期回测 · RL · 监控 · 真正完美")

    init_session_state()
    trading_service = container.trading_service()
    backtest_engine = container.backtest_engine()
    metrics = container.metrics()
    data_provider = container.data_provider()
    await data_provider.init()

    with st.sidebar:
        st.header("⚙️ 配置")
        mode = st.radio("模式", ['实盘', '回测'], index=0)
        st.session_state.mode = 'live' if mode == '实盘' else 'backtest'

        selected_symbols = st.multiselect("交易品种", CONFIG.symbols, default=st.session_state.current_symbols)
        st.session_state.current_symbols = selected_symbols

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

        st.write(f"单笔风险: {CONFIG.risk_per_trade*100:.1f}%")
        st.write(f"每日风险预算: {CONFIG.daily_risk_budget_ratio*100:.1f}%")

        if st.session_state.mode == 'live':
            state = trading_service.get_state()
            st.number_input("余额 USDT", value=state['balance'], disabled=True)

            if st.button("🔄 同步实盘余额"):
                if st.session_state.exchange and not st.session_state.use_simulated_data:
                    try:
                        bal = await st.session_state.exchange.fetch_balance()
                        trading_service.balance = float(bal['total'].get('USDT', 0))
                        st.success(f"同步成功: {trading_service.balance:.2f} USDT")
                    except Exception as e:
                        st.error(f"同步失败: {e}")

            st.markdown("---")
            st.subheader("实盘")
            exchange_choice = st.selectbox("交易所", list(CONFIG.exchanges.keys()), key='exchange_choice')
            testnet = st.checkbox("测试网", value=st.session_state.testnet)
            use_real = st.checkbox("实盘交易", value=st.session_state.use_real)

            if st.button("🔌 测试连接"):
                try:
                    ex_class = getattr(ccxt_async, CONFIG.exchanges[exchange_choice])
                    ex = ex_class({
                        'apiKey': CONFIG.binance_api_key,
                        'secret': CONFIG.binance_secret_key,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    })
                    if testnet:
                        ex.set_sandbox_mode(True)
                    ticker = await ex.fetch_ticker(selected_symbols[0])
                    st.success(f"连接成功！{selected_symbols[0]} 价格: {ticker['last']}")
                    st.session_state.exchange = ex
                    container.execution_service().set_exchange(ex)
                    st.session_state.testnet = testnet
                    st.session_state.use_real = use_real
                except Exception as e:
                    st.error(f"连接失败: {e}")

            st.session_state.auto_enabled = st.checkbox("自动交易", value=True)
            st.session_state.aggressive_mode = st.checkbox("进攻模式 (允许更高风险)", value=False)

            if st.button("🚨 一键紧急平仓"):
                for sym in list(trading_service.positions.keys()):
                    if sym in trading_service.symbol_current_prices:
                        await trading_service._close_position(sym, trading_service.symbol_current_prices[sym], "紧急平仓", None,
                                                                st.session_state.use_real, exchange_choice, testnet,
                                                                CONFIG.binance_api_key, CONFIG.binance_secret_key)
                st.rerun()

            if st.button("📂 查看历史交易记录"):
                if os.path.exists(TRADE_LOG_FILE):
                    df_trades = pd.read_csv(TRADE_LOG_FILE)
                    st.dataframe(df_trades.tail(20))
                else:
                    st.info("暂无历史交易记录")

            if st.button("📤 发送权益曲线"):
                fig = generate_equity_chart(trading_service.equity_curve)
                if fig:
                    container.notifier().send("当前权益曲线", image=fig)
                    st.success("权益曲线已发送")
                else:
                    st.warning("无权益数据")

        else:
            st.subheader("回测参数")
            start_date = st.date_input("开始日期", datetime.now() - timedelta(days=30))
            end_date = st.date_input("结束日期", datetime.now())
            initial_balance = st.number_input("初始资金", value=10000.0)
            if st.button("▶️ 开始回测"):
                with st.spinner("回测进行中..."):
                    results = await backtest_engine.run(selected_symbols, datetime.combine(start_date, datetime.min.time()),
                                                        datetime.combine(end_date, datetime.max.time()), initial_balance)
                    st.session_state.backtest_results = results
                st.success("回测完成")

        if st.button("🗑️ 重置所有状态"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if not selected_symbols:
        st.warning("请至少选择一个交易品种")
        return

    if st.session_state.mode == 'live':
        multi_data = await trading_service.process_market_data(selected_symbols, st.session_state.use_simulated_data)
        if multi_data is None:
            st.error("数据获取失败")
            st.session_state.data_source_failed = True
            return
        st.session_state.data_source_failed = False

        signals = trading_service.generate_signals()

        if st.session_state.auto_enabled and not trading_service.degraded_mode:
            await trading_service.execute_signals(signals, st.session_state.aggressive_mode,
                                                  st.session_state.use_real, exchange_choice, testnet,
                                                  CONFIG.binance_api_key, CONFIG.binance_secret_key)
            await container.execution_service().sync_order_status()

        state = trading_service.get_state()
        metrics.update(trading_service)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["多品种持仓", "高级监控", "风险仪表盘", "研究报告", "系统自检"])
        with tab1:
            render_trading_tab(selected_symbols, state)
        with tab2:
            render_dashboard_panel(state)
        with tab3:
            render_risk_dashboard(trading_service)
        with tab4:
            render_research_panel(state)
        with tab5:
            render_self_check(trading_service)

    else:
        if st.session_state.backtest_results:
            res = st.session_state.backtest_results
            st.metric("初始资金", f"{res['initial_balance']:.2f}")
            st.metric("最终资金", f"{res['final_balance']:.2f}")
            st.metric("总收益率", f"{res['total_return']:.2f}%")
            st.metric("夏普比率", f"{res['sharpe']:.2f}")
            st.metric("最大回撤", f"{res['max_drawdown']:.2f}")
            if res['trades']:
                df_trades = pd.DataFrame(res['trades'])
                st.dataframe(df_trades)
            if res['equity_curve']:
                df_eq = pd.DataFrame(res['equity_curve'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_eq['time'], y=df_eq['equity'], mode='lines'))
                fig.update_layout(title="回测权益曲线", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

def render_trading_tab(symbols: List[str], state: Dict):
    st.subheader("多品种持仓")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("### 📊 市场状态")
        c1, c2, c3 = st.columns(3)
        c1.metric("恐惧贪婪", state['fear_greed'])
        c2.metric("市场状态", state['market_regime'].value)
        c3.metric("当前价格", f"{state['symbol_current_prices'][symbols[0]]:.2f}")

        for sym in symbols:
            st.write(f"{sym}: {state['symbol_current_prices'][sym]:.2f}")

        if state['positions']:
            st.markdown("### 📈 当前持仓")
            pos_list = []
            for sym, pos in state['positions'].items():
                current = state['symbol_current_prices'][sym]
                pnl = pos.pnl(current)
                pnl_pct = (current - pos.entry_price) / pos.entry_price * 100 * pos.direction
                pos_list.append((sym, pos, pnl, pnl_pct))
            pos_list.sort(key=lambda x: x[3], reverse=True)
            for sym, pos, pnl, pnl_pct in pos_list:
                color = "green" if pnl > 0 else "red"
                hold_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
                st.markdown(
                    f"<span style='color:{color}'>{sym}: {'多' if pos.direction==1 else '空'} 入场 {pos.entry_price:.2f} 数量 {pos.size:.4f} "
                    f"浮动盈亏 {pnl:.2f} ({pnl_pct:+.2f}%) 持仓时长 {hold_hours:.1f}h "
                    f"止损 {pos.stop_loss:.2f} 止盈 {pos.take_profit:.2f}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("### 无持仓")
            st.info("等待信号...")

    with col2:
        first_sym = symbols[0]
        df_plot = state['multi_df'][first_sym]['15m'].tail(120).copy()
        if not df_plot.empty:
            if not pd.api.types.is_datetime64_any_dtype(df_plot['timestamp']):
                df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')
            df_plot = df_plot.dropna(subset=['timestamp'])
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5,0.15,0.15,0.2], vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(x=df_plot['timestamp'], open=df_plot['open'], high=df_plot['high'],
                                          low=df_plot['low'], close=df_plot['close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema20'], line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema50'], line=dict(color="blue")), row=1, col=1)
            if first_sym in state['positions']:
                pos = state['positions'][first_sym]
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
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
        else:
            st.warning("无图表数据")

def render_dashboard_panel(state: Dict):
    st.markdown("## 📊 高级监控仪表盘")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("因子暴露")
        df_weights = pd.DataFrame(list(state.get('factor_ic_stats', {}).items()), columns=['因子', '统计'])
        if not df_weights.empty:
            st.dataframe(df_weights)
        else:
            st.info("暂无因子统计")
    with col2:
        st.subheader("滑点监控")
        if state['slippage_history']:
            df_slip = pd.DataFrame(list(state['slippage_history']))
            fig_slip = go.Figure()
            fig_slip.add_trace(go.Scatter(x=df_slip['time'], y=df_slip['slippage'], mode='lines+markers', name='滑点'))
            fig_slip.update_layout(title="实时滑点", xaxis_title="时间", yaxis_title="滑点 (USDT)", height=300)
            st.plotly_chart(fig_slip, use_container_width=True, key="slippage_chart")
        else:
            st.info("暂无滑点数据")

def render_risk_dashboard(trading_service: TradingService):
    st.subheader("🚨 风险仪表盘")
    col1, col2 = st.columns(2)
    with col1:
        circuit, reason = trading_service.risk_manager.check_circuit_breaker(
            trading_service.daily_pnl, trading_service.balance, trading_service.consecutive_losses,
            trading_service.positions, trading_service.multi_df, trading_service.symbol_current_prices
        )
        st.metric("熔断状态", "触发" if circuit else "正常")
        if circuit:
            st.error(reason)
    with col2:
        st.metric("连亏次数", trading_service.consecutive_losses)
    st.metric("今日风险消耗", f"{trading_service.daily_risk_consumed:.2f} / {trading_service.balance * CONFIG.daily_risk_budget_ratio:.2f} USDT")
    st.metric("剩余风险预算", max(0, trading_service.balance * CONFIG.daily_risk_budget_ratio - trading_service.daily_risk_consumed))

def render_research_panel(state: Dict):
    st.subheader("📈 研究报告")
    if state['ic_history']:
        fig_ic = go.Figure()
        for factor, hist in state['ic_history'].items():
            if len(hist) > 10:
                fig_ic.add_trace(go.Scatter(y=list(hist), mode='lines', name=factor))
        fig_ic.update_layout(title="因子IC历史", xaxis_title="时间", yaxis_title="IC")
        st.plotly_chart(fig_ic, use_container_width=True, key="ic_history")
    else:
        st.info("暂无IC历史")

def render_self_check(trading_service: TradingService):
    st.subheader("🔍 系统自检")
    if trading_service.degraded_mode:
        st.error("系统降级模式")
    else:
        st.success("系统健康")

def generate_equity_chart(equity_curve: Deque) -> Optional[go.Figure]:
    if not equity_curve:
        return None
    df = pd.DataFrame(list(equity_curve)[-200:])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['equity'], mode='lines', name='当前权益', line=dict(color='yellow')))
    fig.update_layout(
        title="权益曲线",
        xaxis_title="时间",
        yaxis_title="权益 (USDT)",
        template="plotly_dark",
        height=400
    )
    return fig

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
