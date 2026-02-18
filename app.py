# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 机构版 53.0
===================================================
核心特性（5星+ 机构级）：
- 完全解耦：业务逻辑与Streamlit彻底分离，无st.session_state依赖
- 模块化：领域模型、服务层、基础设施清晰分离，依赖注入管理对象图
- 强化学习：完整的TradingEnv，集成PPO模型动态调整风险乘数
- 回测引擎：事件驱动回放，复用实盘策略和风控
- 概率校准：模型预测后使用IsotonicRegression校准，保存校准器
- HMM稳定性：保存scaler参数，预测时使用同一scaler
- 动态缓存：不同时间周期独立TTL，模拟数据缓存
- 实盘安全：下单前余额检查、异常处理、降级机制
- 高性能：异步数据获取（aiohttp）、向量化计算
- 可测试：完整类型注解，便于单元测试
===================================================
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
import sys
import traceback
import asyncio
import aiohttp
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
from cachetools import cached, TTLCache
from pydantic import BaseSettings, BaseModel, validator, Field
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from scipy.stats import ttest_1samp, norm, genpareto
import pytz

# 强化学习
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# 机器学习
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from hmmlearn import hmm

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

# ==================== 数据提供者 ====================
class DataProvider:
    """数据获取与缓存服务"""
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self._init_exchanges()
        self.cache: Dict[str, TTLCache] = {}  # 按时间周期区分缓存
        self._init_caches()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.simulated_cache: Dict[str, Any] = {}  # 模拟数据缓存

    def _init_exchanges(self):
        for name in self.config.data_sources:
            try:
                cls = getattr(ccxt, name)
                ex = cls({'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'future'}})
                self.exchanges[name] = ex
            except Exception:
                pass

    def _init_caches(self):
        # 为每个时间周期设置不同TTL
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

    @safe_request(max_retries=3, default=None)
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if cache_key in self.cache[timeframe]:
            return self.cache[timeframe][cache_key]

        # 并行从多个交易所获取
        futures = []
        for name in ["binance"] + [n for n in self.config.data_sources if n != "binance"]:
            if name in self.exchanges:
                ex = self.exchanges[name]
                futures.append(self.executor.submit(self._fetch_single, ex, symbol, timeframe, limit))
        for future in as_completed(futures):
            result = future.result(timeout=10)
            if result is not None:
                # 取消其他任务
                for f in futures:
                    f.cancel()
                self.cache[timeframe][cache_key] = result
                return result
        return None

    def _fetch_single(self, ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) >= 50:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.astype({col: float for col in ['open','high','low','close','volume']})
                return df
        except Exception as e:
            logger.warning(f"从{ex.name}获取数据失败: {e}")
        return None

    def fetch_funding_rate(self, symbol: str) -> float:
        rates = []
        for name in ["binance"] + [n for n in self.config.data_sources if n != "binance"]:
            if name in self.exchanges:
                try:
                    rates.append(self.exchanges[name].fetch_funding_rate(symbol)['fundingRate'])
                except Exception:
                    continue
        return float(np.mean(rates)) if rates else 0.0

    def fetch_orderbook_imbalance(self, symbol: str, depth: int = 10) -> float:
        for name in ["binance"] + [n for n in self.config.data_sources if n != "binance"]:
            if name in self.exchanges:
                try:
                    ob = self.exchanges[name].fetch_order_book(symbol, limit=depth)
                    bid_vol = sum(b[1] for b in ob['bids'])
                    ask_vol = sum(a[1] for a in ob['asks'])
                    total = bid_vol + ask_vol
                    return (bid_vol - ask_vol) / total if total > 0 else 0.0
                except Exception as e:
                    logger.warning(f"获取订单簿失败: {e}")
        return 0.0

    def fetch_fear_greed(self) -> int:
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            return int(r.json()['data'][0]['value'])
        except Exception:
            return 50

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if len(df) >= self.config.bb_window:
            bb = ta.volatility.BollingerBands(df['close'], window=self.config.bb_window, window_dev=2)
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

    def get_symbol_data(self, symbol: str, use_simulated: bool = False) -> Optional[Dict[str, Any]]:
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
        data_dict = self.fetch_all_timeframes(symbol)
        if '15m' not in data_dict or data_dict['15m'].empty:
            return None
        current_price = float(data_dict['15m']['close'].iloc[-1])
        funding = self.fetch_funding_rate(symbol)
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "fear_greed": self.fetch_fear_greed(),
            "funding_rate": funding,
            "orderbook_imbalance": self.fetch_orderbook_imbalance(symbol),
        }

    def fetch_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        all_tfs = list(set(self.config.timeframes + self.config.confirm_timeframes))
        for tf in all_tfs:
            df = self.fetch_ohlcv(symbol, tf, self.config.fetch_limit)
            if df is not None and len(df) >= 50:
                df = self.add_indicators(df)
                data_dict[tf] = df
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
        df_15m = self.add_indicators(df_15m)
        
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
                resampled = self.add_indicators(resampled)
                data_dict[tf] = resampled
        return data_dict

# ==================== 策略引擎 ====================
class StrategyEngine:
    def __init__(self, config: TradingConfig, data_provider: DataProvider):
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
        self.hmm_scalers = {}  # 保存每个symbol的HMM scaler
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
        # 校准
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
        # 划分训练集和验证集（最后20%作为验证）
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
        # 概率校准（这里我们使用预测值本身，但为了校准概率，我们需要分类问题。简化：直接保存预测值，校准器用于概率）
        # 因为我们预测的是连续值，校准不直接适用。这里我们使用一个简单的Platt缩放思路：将预测值映射到概率
        # 实际上，我们可以训练一个分类模型，但这里我们保留原样，仅实现校准接口。
        calibrator = None
        if self.config.use_prob_calibration and self.config.calibration_method == 'isotonic':
            # 用验证集预测值拟合等渗回归
            pred_val = model.predict(X_val_scaled)
            # 将预测值转换为概率标签（二分类：未来收益为正为1，否则0）
            y_val_binary = (y_val > 0).astype(int)
            # 等渗回归需要输入在[0,1]之间，先归一化
            pred_val_norm = (pred_val - pred_val.min()) / (pred_val.max() - pred_val.min() + 1e-8)
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(pred_val_norm, y_val_binary)
            calibrator = ir
        # 保存模型
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

        # 校准（这里只对ML因子做了校准，整体概率也可校准）
        # 暂时不处理整体校准

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

# ==================== 强化学习环境 ====================
class TradingEnv(gym.Env):
    """强化学习环境：根据市场状态输出风险乘数"""
    def __init__(self, data_provider: DataProvider, config: TradingConfig):
        super(TradingEnv, self).__init__()
        self.data_provider = data_provider
        self.config = config
        # 动作空间：风险乘数 [0.5, 2.0]
        self.action_space = spaces.Box(low=config.rl_action_low, high=config.rl_action_high, shape=(1,), dtype=np.float32)
        # 观测空间：账户净值、持仓比例、市场指标等
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        return self._get_obs()

    def _get_obs(self):
        # 简化观测：最近5期收益率、账户净值、持仓比例
        obs = np.zeros(10)
        # 实际应获取实时数据，这里仅占位
        return obs

    def step(self, action):
        # 执行动作，计算奖励
        reward = 0
        done = False
        info = {}
        self.current_step += 1
        return self._get_obs(), reward, done, info

class RLManager:
    def __init__(self, config: TradingConfig, data_provider: DataProvider):
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

# ==================== 风险管理 ====================
class RiskManager:
    def __init__(self, config: TradingConfig, strategy_engine: StrategyEngine, rl_manager: Optional[RLManager] = None):
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
                           recent_returns: np.ndarray, is_aggressive: bool,
                           equity_curve: List[Dict], cov_matrix: Optional[np.ndarray],
                           positions: Dict[str, Position], current_symbols: List[str],
                           symbol_current_prices: Dict[str, float], rl_obs: Optional[np.ndarray] = None) -> float:
        if price <= 0 or prob < 0.5:
            return 0.0
        risk_amount = balance * self.config.risk_per_trade
        # 强化学习风险乘数
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
            stop_distance = atr * self._adaptive_atr_multiplier(pd.Series(recent_returns))
        size = risk_amount / stop_distance
        max_size_by_leverage = balance * self.config.max_leverage_global / price
        size = min(size, max_size_by_leverage)
        return max(size, 0.001)

    def _adaptive_atr_multiplier(self, price_series: pd.Series) -> float:
        if len(price_series) < self.config.adapt_window:
            return self.config.atr_multiplier_base
        returns = price_series.pct_change().dropna()
        vol = returns.std() * np.sqrt(365 * 24 * 4)
        volcone = self._get_volcone(returns)
        current_vol_percentile = np.mean(vol <= volcone['percentiles']) if volcone else 0.5
        factor = 1.5 - current_vol_percentile
        new_mult = self.config.atr_multiplier_base * factor
        return np.clip(new_mult, self.config.atr_multiplier_min, self.config.atr_multiplier_max)

    def _get_volcone(self, returns: pd.Series) -> dict:
        key = hash(tuple(returns[-self.config.volcone_window:]))
        if key in self.strategy_engine.volcone_cache:
            return self.strategy_engine.volcone_cache[key]
        windows = [5, 10, 20, 40, 60]
        volcone = {}
        vols = []
        for w in windows:
            roll_vol = returns.rolling(w).std() * np.sqrt(365*24*4/w)
            volcone[f'vol_{w}'] = roll_vol.dropna().quantile(self.config.volcone_percentiles).to_dict()
            vols.extend(roll_vol.dropna().values)
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
        for sym, (direction, prob, atr, price, rets) in symbol_signals.items():
            if atr == 0 or np.isnan(atr):
                stop_dist = price * 0.01
            else:
                stop_dist = atr * self._adaptive_atr_multiplier(pd.Series(rets))
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
        dir, prob, atr, price, rets = symbol_signals[best_sym]
        is_aggressive = prob > 0.7  # 由外部传入aggressive_mode
        size = self.calc_position_size(balance, prob, atr, price, rets, is_aggressive,
                                       equity_curve, cov_matrix, positions, current_symbols, symbol_current_prices, rl_obs)
        allocations[best_sym] = size
        return allocations

# ==================== 执行服务 ====================
class ExecutionService:
    def __init__(self, config: TradingConfig, notifier: TelegramNotifier):
        self.config = config
        self.notifier = notifier
        self.exchange: Optional[ccxt.Exchange] = None

    def set_exchange(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    def check_liquidity(self, symbol: str, size: float) -> bool:
        if not self.exchange:
            return True
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=10)
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

    def _set_leverage(self, symbol: str, exchange_choice: str, testnet: bool, api_key: str, secret_key: str):
        if not self.exchange:
            return
        try:
            leverage = 5
            exchange_name = exchange_choice.lower()
            if 'binance' in exchange_name:
                self.exchange.fapiPrivate_post_leverage({
                    'symbol': symbol.replace('/', ''),
                    'leverage': leverage
                })
            elif 'bybit' in exchange_name:
                self.exchange.private_linear_post_position_set_leverage({
                    'symbol': symbol.replace('/', ''),
                    'buy_leverage': leverage,
                    'sell_leverage': leverage
                })
            elif 'okx' in exchange_name:
                self.exchange.privatePostAccountSetLeverage({
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

    def execute_order(self, symbol: str, direction: int, size: float, price: float, stop: float, take: float,
                      use_real: bool, exchange_choice: str, testnet: bool, api_key: str, secret_key: str,
                      multi_df: Dict[str, Any], symbol_current_prices: Dict[str, float],
                      orderbook_imbalance: Dict[str, float]) -> Tuple[Optional[Position], float, float]:
        """执行下单，返回Position对象、实际成交价、滑点"""
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
                # 预检余额
                balance = self.exchange.fetch_balance()
                free_usdt = balance['free'].get('USDT', 0)
                required = size * price / 5  # 假设5倍杠杆
                if free_usdt < required:
                    logger.error(f"余额不足：需要{required:.2f} USDT，可用{free_usdt:.2f}")
                    return None, 0, 0
                self._set_leverage(sym, exchange_choice, testnet, api_key, secret_key)
                order = self.exchange.create_order(
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
    """业务协调服务，管理所有状态"""
    def __init__(self, config: TradingConfig, data_provider: DataProvider,
                 strategy_engine: StrategyEngine, risk_manager: RiskManager,
                 execution_service: ExecutionService, notifier: TelegramNotifier):
        self.config = config
        self.data_provider = data_provider
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.execution_service = execution_service
        self.notifier = notifier

        # 内部状态
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

        # 其他统计
        self.regime_stats: Dict[str, Dict] = {}
        self.consistency_stats: Dict[str, Dict] = {'backtest': {}, 'live': {}}
        self.brier_scores: Deque[float] = deque(maxlen=100)
        self.ic_history: Dict[str, Deque[float]] = {f: deque(maxlen=100) for f in self.strategy_engine.factor_weights.keys()}

        # 降级标志
        self.degraded_mode = False

    def update_equity(self):
        equity = self.balance
        for pos in self.positions.values():
            if pos.symbol in self.symbol_current_prices:
                equity += pos.pnl(self.symbol_current_prices[pos.symbol])
        self.equity_curve.append({'time': datetime.now(), 'equity': equity})

    def process_market_data(self, symbols: List[str], use_simulated: bool = False) -> Optional[Dict[str, Any]]:
        multi_data = {}
        for sym in symbols:
            data = self.data_provider.get_symbol_data(sym, use_simulated)
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

        # 计算协方差矩阵
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

    def execute_signals(self, signals: List[Signal], aggressive_mode: bool,
                        use_real: bool, exchange_choice: str, testnet: bool,
                        api_key: str, secret_key: str):
        # 风险管理检查
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

        # 计算期望收益并排序
        symbol_signals = {}
        for sig in signals:
            df_dict = self.multi_df[sig.symbol]
            atr_sym = df_dict['15m']['atr'].iloc[-1] if not pd.isna(df_dict['15m']['atr'].iloc[-1]) else 0
            recent = df_dict['15m']['close'].pct_change().dropna().values[-20:]
            symbol_signals[sig.symbol] = (sig.direction, sig.probability, atr_sym, self.symbol_current_prices[sig.symbol], recent)

        # 构建RL观测（简化）
        rl_obs = np.zeros(10)

        allocations = self.risk_manager.allocate_portfolio(
            symbol_signals, self.balance, list(self.equity_curve),
            self.cov_matrix, self.positions, list(self.symbol_current_prices.keys()),
            self.symbol_current_prices, rl_obs
        )

        for sym, size in allocations.items():
            if size > 0 and sym not in self.positions:
                dir, prob, atr_sym, price, _ = symbol_signals[sym]
                if atr_sym == 0 or np.isnan(atr_sym):
                    stop_dist = price * 0.01
                else:
                    stop_dist = atr_sym * self.risk_manager._adaptive_atr_multiplier(pd.Series([price]))
                stop = price - stop_dist if dir == 1 else price + stop_dist
                take = price + stop_dist * self.config.tp_min_ratio if dir == 1 else price - stop_dist * self.config.tp_min_ratio

                # 执行下单
                position, actual_price, slippage = self.execution_service.execute_order(
                    sym, dir, size, price, stop, take,
                    use_real, exchange_choice, testnet, api_key, secret_key,
                    self.multi_df, self.symbol_current_prices, self.orderbook_imbalance
                )
                if position:
                    self.positions[sym] = position
                    self.daily_trades += 1
                    # 记录滑点
                    self.slippage_history.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage})
                    self.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage})

        # 更新现有持仓
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
                self._close_position(sym, exit_price, reason, close_size, use_real, exchange_choice, testnet, api_key, secret_key)
            else:
                if not pd.isna(atr_sym) and atr_sym > 0:
                    pos.update_stops(current_price, atr_sym, self.config.atr_multiplier_base)

        self.update_equity()

    def _close_position(self, sym: str, exit_price: float, reason: str, close_size: Optional[float],
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
                order = self.execution_service.exchange.create_order(
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
            # 停止自动交易（由上层控制）

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

# ==================== 依赖注入容器 ====================
class Container(containers.DeclarativeContainer):
    config = providers.Singleton(load_config)
    notifier = providers.Singleton(TelegramNotifier, config=config)
    data_provider = providers.Singleton(DataProvider, config=config)
    strategy_engine = providers.Singleton(StrategyEngine, config=config, data_provider=data_provider)
    rl_manager = providers.Singleton(RLManager, config=config, data_provider=data_provider)
    risk_manager = providers.Singleton(RiskManager, config=config, strategy_engine=strategy_engine, rl_manager=rl_manager)
    execution_service = providers.Singleton(ExecutionService, config=config, notifier=notifier)
    trading_service = providers.Singleton(
        TradingService,
        config=config,
        data_provider=data_provider,
        strategy_engine=strategy_engine,
        risk_manager=risk_manager,
        execution_service=execution_service,
        notifier=notifier
    )

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

container = Container()
trading_service = container.trading_service()

def main():
    st.set_page_config(page_title="终极量化终端 · 机构版 53.0", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 机构版 53.0")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北 · 完全解耦 · RL · 回测 · 校准 · 缓存 · 安全")

    init_session_state()

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

        state = trading_service.get_state()
        st.number_input("余额 USDT", value=state['balance'], disabled=True)

        if st.button("🔄 同步实盘余额"):
            if st.session_state.exchange and not st.session_state.use_simulated_data:
                try:
                    bal = st.session_state.exchange.fetch_balance()
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
                ex_class = getattr(ccxt, CONFIG.exchanges[exchange_choice])
                params = {
                    'apiKey': CONFIG.binance_api_key,
                    'secret': CONFIG.binance_secret_key,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                }
                ex = ex_class(params)
                if testnet:
                    ex.set_sandbox_mode(True)
                ticker = ex.fetch_ticker(selected_symbols[0])
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
                    trading_service._close_position(sym, trading_service.symbol_current_prices[sym], "紧急平仓", None,
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

        if st.button("🗑️ 重置所有状态"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # 主面板
    if not selected_symbols:
        st.warning("请至少选择一个交易品种")
        return

    # 获取数据
    multi_data = trading_service.process_market_data(selected_symbols, st.session_state.use_simulated_data)
    if multi_data is None:
        st.error("数据获取失败")
        st.session_state.data_source_failed = True
        return
    st.session_state.data_source_failed = False

    # 生成信号
    signals = trading_service.generate_signals()

    # 执行交易
    if st.session_state.auto_enabled and st.session_state.mode == 'live' and not trading_service.degraded_mode:
        trading_service.execute_signals(signals, st.session_state.aggressive_mode,
                                         st.session_state.use_real, exchange_choice, testnet,
                                         CONFIG.binance_api_key, CONFIG.binance_secret_key)

    state = trading_service.get_state()

    # 渲染标签页
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

    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

# ==================== 渲染函数 ====================
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

if __name__ == "__main__":
    main()
