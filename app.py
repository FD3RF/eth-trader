# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 职业版 48.3 (终极融合版)
===================================================
[版本说明]
- 融合 48.1(最终完美版) 与 48.2(生产完善版) 所有优势
- 资金费精确对齐 UTC 0/8/16 结算点
- 协方差矩阵奇异/NaN 时回退到等风险贡献
- WebSocket 订单监听异常降级到 REST 轮询
- 修复机器学习训练中的数据泄露
- 数据库连接池 + WAL 模式
- 环境变量管理 API 密钥
- 因子权重定时更新
- 回测进度条显示
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
from datetime import datetime, timedelta, timezone
from streamlit_autorefresh import st_autorefresh
import warnings
import time
import logging
import sys
import traceback
from typing import Optional, Dict, List, Tuple, Any, Union, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import functools
import hashlib
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import ttest_1samp, norm, genpareto
import pytz
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import pickle
import sqlite3
import threading
import asyncio
import queue
from skopt import gp_minimize
from skopt.space import Real, Integer

# ==================== 环境变量与安全配置 ====================
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")
BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY", "")
BYBIT_SECRET_KEY = os.environ.get("BYBIT_SECRET_KEY", "")
OKX_API_KEY = os.environ.get("OKX_API_KEY", "")
OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY", "")
OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE", "")

# ==================== 依赖检查 ====================
def check_dependencies() -> None:
    required_packages = {
        'streamlit': st.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'ta': ta.__version__,
        'ccxt': ccxt.__version__,
        'requests': requests.__version__,
        'plotly': go.__version__,
        'scipy': 'installed',
        'pytz': pytz.__version__,
        'sklearn': 'installed',
        'joblib': joblib.__version__,
        'skopt': 'installed',
    }
    missing = []
    for pkg, ver in required_packages.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        st.error(f"缺少依赖库: {', '.join(missing)}。请运行: pip install " + ' '.join(missing))
        st.stop()

check_dependencies()

# 尝试导入 hmmlearn，如果失败则降级
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("警告: hmmlearn 未安装，将使用传统方法检测市场状态。")

# 尝试导入 ccxt.pro (WebSocket支持)
try:
    import ccxt.pro as ccxtpro
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("警告: ccxt.pro 未安装，WebSocket实时性增强不可用，将回退到REST。")

warnings.filterwarnings('ignore')

# ==================== 全局异常捕获 ====================
def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    with open('crash.log', 'a') as f:
        f.write(f"\n--- {datetime.now()} ---\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

sys.excepthook = global_exception_handler

# ==================== 日志统一配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UltimateTrader")

def log_error(msg: str) -> None:
    logger.error(msg)
    if 'error_log' in st.session_state:
        st.session_state.error_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")

def log_execution(msg: str) -> None:
    logger.info(msg)
    if 'execution_log' in st.session_state:
        st.session_state.execution_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")

# ==================== 数据库连接池 ====================
DB_PATH = "trading_data.db"
DB_CONN = None
DB_LOCK = threading.Lock()

def get_db_conn() -> sqlite3.Connection:
    """获取全局数据库连接（线程安全，启用WAL模式）"""
    global DB_CONN
    with DB_LOCK:
        if DB_CONN is None:
            DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
            DB_CONN.execute("PRAGMA journal_mode=WAL")
            _init_db_tables(DB_CONN)
    return DB_CONN

def _init_db_tables(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (time TEXT, symbol TEXT, direction TEXT, entry REAL, exit REAL,
                  size REAL, pnl REAL, reason TEXT, slippage_entry REAL,
                  slippage_exit REAL, impact_cost REAL, raw_prob REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS equity_curve
                 (time TEXT, equity REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS regime_stats
                 (regime TEXT PRIMARY KEY, trades INTEGER, wins INTEGER, total_pnl REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS consistency_stats
                 (type TEXT PRIMARY KEY, trades INTEGER, avg_slippage REAL, win_rate REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS slippage_log
                 (time TEXT, symbol TEXT, slippage REAL, impact REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS funding_rates
                 (time TEXT, symbol TEXT, rate REAL)''')
    conn.commit()

def append_to_db(table: str, row: dict) -> None:
    conn = get_db_conn()
    with DB_LOCK:
        cursor = conn.cursor()
        columns = ', '.join(row.keys())
        placeholders = ', '.join(['?' for _ in row])
        cursor.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", list(row.values()))
        conn.commit()

def load_from_db(table: str, limit: int = None, condition: str = "", params: tuple = ()) -> pd.DataFrame:
    conn = get_db_conn()
    query = f"SELECT * FROM {table} {condition} ORDER BY time DESC"
    if limit:
        query += f" LIMIT {limit}"
    return pd.read_sql_query(query, conn, params=params)

# 日志目录和文件常量
LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 资金费精确结算常量 ====================
FUNDING_TIMES = [0, 8, 16]  # UTC 小时

def get_next_funding_time(entry_time: datetime) -> datetime:
    """计算下一次资金费结算时间（UTC）"""
    utc = entry_time.astimezone(timezone.utc)
    current_hour = utc.hour
    next_hour = min((h for h in FUNDING_TIMES if h > current_hour), default=FUNDING_TIMES[0])
    if next_hour <= current_hour:
        next_day = (utc + timedelta(days=1)).date()
    else:
        next_day = utc.date()
    return datetime.combine(next_day, datetime.min.time()).replace(hour=next_hour, tzinfo=timezone.utc)

# ==================== 协方差稳定性处理 ====================
def safe_cov_matrix(cov: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """如果协方差矩阵奇异或包含NaN，返回单位矩阵（等风险贡献）"""
    if cov is None or np.any(np.isnan(cov)) or np.linalg.det(cov) < 1e-10:
        return np.eye(cov.shape[0]) if cov is not None else None
    return cov

# ==================== 配置与常量 ====================
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

class VaRMethod(Enum):
    NORMAL = "正态法"
    HISTORICAL = "历史模拟法"
    EXTREME = "极值法"
    VOLCONE = "波动率锥法"

@dataclass
class TradingConfig:
    """所有可调参数集中管理（最终完美版）"""
    symbols: List[str] = field(default_factory=lambda: ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"])
    # ========== 风险预算模型 ==========
    risk_per_trade: float = 0.008          # 单笔风险比例（账户余额的0.8%）——将被动态凯利覆盖
    daily_risk_budget_ratio: float = 0.025 # 每日风险预算比例（2.5%）
    # 冷却与熔断
    max_consecutive_losses: int = 3
    cooldown_losses: int = 3
    cooldown_hours: int = 24
    max_drawdown_pct: float = 20.0
    circuit_breaker_atr: float = 5.0
    circuit_breaker_fg_extreme: Tuple[int, int] = (10, 90)
    # 波动率相关
    atr_multiplier_base: float = 1.5
    atr_multiplier_min: float = 1.2
    atr_multiplier_max: float = 2.5
    tp_min_ratio: float = 2.0               # 止盈/止损最小比例
    partial_tp_ratio: float = 0.5
    partial_tp_r_multiple: float = 1.2
    breakeven_trigger_pct: float = 1.5
    max_hold_hours: int = 36
    min_atr_pct: float = 0.5                # 最小ATR百分比（用于计算止损距离下限）
    # 凯利相关（动态凯利将覆盖）
    kelly_fraction: float = 0.25
    # 交易所与数据
    exchanges: Dict[str, Any] = field(default_factory=lambda: {
        "Binance合约": ccxt.binance,
        "Bybit合约": ccxt.bybit,
        "OKX合约": ccxt.okx
    })
    data_sources: List[str] = field(default_factory=lambda: ["binance", "bybit", "okx", "mexc", "kucoin"])
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h', '4h', '1d'])
    confirm_timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])
    timeframe_weights: Dict[str, int] = field(default_factory=lambda: {'1d': 10, '4h': 7, '1h': 5, '15m': 3, '5m': 2, '1m': 1})
    fetch_limit: int = 2000
    auto_refresh_ms: int = 30000
    anti_duplicate_seconds: int = 180
    # 滑点与手续费
    slippage_base: float = 0.0005
    slippage_impact_factor: float = 0.1
    slippage_imbalance_factor: float = 0.5
    fee_rate: float = 0.0004
    # 因子与机器学习
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
    factor_learning_rate: float = 0.3        # 保留用于传统权重更新
    # 协方差风险预算
    risk_budget_method: str = "risk_parity"
    black_litterman_tau: float = 0.05
    cov_matrix_window: int = 50
    max_sector_exposure: float = 0.3
    # 订单拆分
    max_order_split: int = 3
    min_order_size: float = 0.001
    split_delay_seconds: int = 5
    # 市场状态过滤
    regime_allow_trade: List[MarketRegime] = field(default_factory=lambda: [MarketRegime.TREND, MarketRegime.PANIC])
    # 夜间模式
    night_start_hour: int = 0
    night_end_hour: int = 8
    night_risk_multiplier: float = 0.5
    # 波动率锥
    volcone_percentiles: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.5, 0.95, 0.99])
    volcone_window: int = 100
    # 其他
    var_confidence: float = 0.95
    var_method: VaRMethod = VaRMethod.HISTORICAL
    portfolio_risk_target: float = 0.02
    var_aggressive_threshold: float = 1.0
    adapt_window: int = 20
    atr_price_history_len: int = 20
    funding_rate_threshold: float = 0.0005   # 修正：0.05% (之前为5%过高)
    max_leverage_global: float = 5.0        # 修改为默认5倍（与set_leverage一致）
    # 异常检测阈值
    max_reasonable_balance: float = 1e7
    max_reasonable_daily_pnl_ratio: float = 10.0
    # HMM
    regime_detection_method: str = "hmm" if HMM_AVAILABLE else "traditional"
    hmm_n_components: int = 3
    hmm_n_iter: int = 100
    # 成本感知训练
    cost_aware_training: bool = True
    # 模拟数据参数
    sim_volatility: float = 0.06
    sim_trend_strength: float = 0.2
    # 布林带宽度阈值（用于震荡判断）
    bb_width_threshold: float = 0.1
    bb_window: int = 20
    rsi_range_low: int = 40
    rsi_range_high: int = 60
    # ========== 新增优化参数 ==========
    max_queue_size: int = 1000               # 队列积压阈值
    training_schedule_interval: int = 3600   # 模型训练统一间隔（秒）
    signal_cache_ttl: int = 60                # 信号缓存有效期（秒）
    funding_fee_hours: int = 8                # 资金费结算周期（小时）

CONFIG = TradingConfig()

# ==================== 因子名称常量（增加新因子）====================
FACTOR_NAMES = ['trend', 'rsi', 'macd', 'bb', 'volume', 'adx', 'ml', 'imbalance', 'funding_change']
FACTOR_TO_COL = {
    'trend': 'trend_factor',
    'rsi': 'rsi',
    'macd': 'macd_diff',
    'bb': 'bb_factor',
    'volume': 'volume_ratio',
    'adx': 'adx',
    'ml': 'ml_factor',
    'imbalance': 'orderbook_imbalance',
    'funding_change': 'funding_rate_change'
}

# ==================== 工具装饰器 ====================
def safe_request(max_retries: int = 3, default=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return default
                    time.sleep(2 ** attempt)
            return default
        return decorator
    return decorator

# ==================== WebSocket数据获取器（增强重连与降级）====================
class WebSocketFetcher:
    def __init__(self, symbols: List[str], timeframes: List[str]):
        self.symbols = symbols
        self.timeframes = timeframes
        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.latest_data: Dict[Tuple[str, str], Deque] = {}
        self.lock = threading.Lock()
        self.max_queue_size = CONFIG.max_queue_size
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60

    def start(self) -> None:
        if not WS_AVAILABLE:
            log_error("ccxt.pro不可用，无法启动WebSocket")
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def _run_async_loop(self) -> None:
        """为线程创建独立事件循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_loop())
        loop.close()

    async def _ws_loop(self) -> None:
        exchange = None
        while self.running:
            try:
                if exchange:
                    await exchange.close()
                exchange = ccxtpro.binance()
                await exchange.load_markets()
                self.reconnect_delay = 1
                tasks = []
                for symbol in self.symbols:
                    for tf in self.timeframes:
                        tasks.append(self._watch_ohlcv(exchange, symbol, tf))
                if st.session_state.use_real and st.session_state.exchange:
                    tasks.append(self._watch_orders_safe(exchange))
                await asyncio.gather(*tasks)
            except Exception as e:
                log_error(f"WebSocket连接异常: {e}，将在{self.reconnect_delay}秒后重连")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                continue

    async def _watch_ohlcv(self, exchange, symbol: str, timeframe: str) -> None:
        while self.running:
            try:
                ohlcv = await exchange.watch_ohlcv(symbol, timeframe)
                if ohlcv:
                    last = ohlcv[-1]
                    ts = last[0]
                    dt = pd.to_datetime(ts, unit='ms')
                    row = {
                        'timestamp': dt,
                        'open': float(last[1]),
                        'high': float(last[2]),
                        'low': float(last[3]),
                        'close': float(last[4]),
                        'volume': float(last[5])
                    }
                    key = (symbol, timeframe)
                    with self.lock:
                        if key not in self.latest_data:
                            self.latest_data[key] = deque(maxlen=500)
                        self.latest_data[key].append(row)
                    if self.data_queue.qsize() > self.max_queue_size:
                        log_error(f"WebSocket数据队列超过阈值 {self.max_queue_size}，可能处理不及时")
                    self.data_queue.put({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'data': row
                    })
            except Exception as e:
                log_error(f"WebSocket watch_ohlcv error for {symbol} {timeframe}: {e}")
                raise

    async def _watch_orders_safe(self, exchange) -> None:
        """实盘订单状态实时更新，异常时抛出以便上层重连（降级到REST轮询）"""
        while self.running:
            try:
                orders = await exchange.watch_orders()
                for order in orders:
                    oid = order['id']
                    if oid in st.session_state.pending_orders:
                        stored = st.session_state.pending_orders[oid]
                        stored.status = order['status']
                        stored.filled = order.get('filled', 0.0)
                        stored.last_check = datetime.now()
                        if order['status'] == 'closed' and stored.symbol not in st.session_state.positions:
                            pos = Position(
                                symbol=stored.symbol,
                                direction=1 if stored.side == 'buy' else -1,
                                entry_price=order['average'] or stored.price,
                                entry_time=datetime.now(),
                                size=stored.filled,
                                stop_loss=stored.stop_loss,
                                take_profit=stored.take_profit,
                                initial_atr=0,
                                real=True,
                                prob=stored.prob
                            )
                            st.session_state.positions[stored.symbol] = pos
                            del st.session_state.pending_orders[oid]
                            log_execution(f"WebSocket订单成交，创建持仓: {stored.symbol}")
            except Exception as e:
                log_error(f"WebSocket watch_orders error: {e}")
                raise  # 让上层重连，降级到REST轮询由 check_pending_orders 完成

    def get_latest_klines(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        key = (symbol, timeframe)
        with self.lock:
            if key not in self.latest_data:
                return pd.DataFrame()
            data = list(self.latest_data[key])
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('timestamp')
        return df

# ==================== 聚合数据获取器 ====================
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

    @safe_request(max_retries=3, default=None)
    def _fetch_kline_single(self, ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if ohlcv and len(ohlcv) >= 50:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.astype({col: float for col in ['open','high','low','close','volume']})
            return df
        return None

    def _fetch_kline_parallel(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        for name in ["binance"] + [n for n in CONFIG.data_sources if n != "binance"]:
            if name in self.exchanges:
                ex = self.exchanges[name]
                result = self._fetch_kline_single(ex, symbol, timeframe, limit)
                if result is not None:
                    return result
        return None

    def fetch_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        all_tfs = list(set(CONFIG.timeframes + CONFIG.confirm_timeframes))
        for tf in all_tfs:
            if st.session_state.ws_fetcher:
                ws_df = st.session_state.ws_fetcher.get_latest_klines(symbol, tf, limit=CONFIG.fetch_limit)
                if not ws_df.empty and len(ws_df) >= 50:
                    data_dict[tf] = add_indicators(ws_df)
                    continue
            df = self._fetch_kline_parallel(symbol, tf, CONFIG.fetch_limit)
            if df is not None and len(df) >= 50:
                df = add_indicators(df)
                data_dict[tf] = df
        return data_dict

    @safe_request(max_retries=3, default=0.0)
    def fetch_funding_rate(self, symbol: str) -> float:
        rates = []
        for name in ["binance"] + [n for n in CONFIG.data_sources if n != "binance"]:
            if name in self.exchanges:
                try:
                    rates.append(self.exchanges[name].fetch_funding_rate(symbol)['fundingRate'])
                except Exception:
                    continue
        return float(np.mean(rates)) if rates else 0.0

    @safe_request(max_retries=3, default=0.0)
    def fetch_orderbook_imbalance(self, symbol: str, depth: int = 10) -> float:
        for name in ["binance"] + [n for n in CONFIG.data_sources if n != "binance"]:
            if name in self.exchanges:
                try:
                    ob = self.exchanges[name].fetch_order_book(symbol, limit=depth)
                    bid_vol = sum(b[1] for b in ob['bids'])
                    ask_vol = sum(a[1] for a in ob['asks'])
                    total = bid_vol + ask_vol
                    return (bid_vol - ask_vol) / total if total > 0 else 0.0
                except:
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
        try:
            data_dict = self.fetch_all_timeframes(symbol)
            if '15m' not in data_dict or data_dict['15m'].empty:
                log_error(f"获取 {symbol} 数据失败，自动切换模拟模式")
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
            funding = self.fetch_funding_rate(symbol)
            st.session_state.funding_rates[symbol] = funding
            st.session_state.funding_rates_history[symbol].append(funding)
            if len(st.session_state.funding_rates_history[symbol]) > 100:
                st.session_state.funding_rates_history[symbol].pop(0)
            funding_change = 0.0
            if len(st.session_state.funding_rates_history[symbol]) >= 2:
                funding_change = st.session_state.funding_rates_history[symbol][-1] - st.session_state.funding_rates_history[symbol][-2]
            imbalance = self.fetch_orderbook_imbalance(symbol)
            st.session_state.orderbook_imbalance[symbol] = imbalance
            for tf in data_dict:
                data_dict[tf]['orderbook_imbalance'] = imbalance
                data_dict[tf]['funding_rate_change'] = funding_change
            return {
                "data_dict": data_dict,
                "current_price": current_price,
                "fear_greed": fetch_fear_greed(),
                "funding_rate": funding,
                "orderbook_imbalance": imbalance,
            }
        except Exception as e:
            log_error(f"get_symbol_data 异常: {e}")
            st.session_state.use_simulated_data = True
            sim_data = generate_simulated_data(symbol)
            return {
                "data_dict": sim_data,
                "current_price": sim_data['15m']['close'].iloc[-1],
                "fear_greed": 50,
                "funding_rate": 0.0,
                "orderbook_imbalance": 0.0,
            }

# ==================== HMM相关函数（修复训练数据泄露）====================
def train_hmm(symbol: str, df_dict: Dict[str, pd.DataFrame], train_end_idx: int = -1) -> Optional[Tuple[Any, Any]]:
    if not HMM_AVAILABLE:
        return None
    df = df_dict['15m'].iloc[:train_end_idx].copy() if train_end_idx != -1 else df_dict['15m'].copy()
    ret = df['close'].pct_change().dropna().values.reshape(-1, 1)
    if len(ret) < 200:
        return None
    scaler = StandardScaler()
    ret_scaled = scaler.fit_transform(ret)
    model = hmm.GaussianHMM(n_components=CONFIG.hmm_n_components, covariance_type="diag", n_iter=CONFIG.hmm_n_iter)
    model.fit(ret_scaled)
    return model, scaler

def detect_hmm_regime(symbol: str, df_dict: Dict[str, pd.DataFrame]) -> int:
    if not HMM_AVAILABLE:
        return 0
    glob = st.session_state.globals
    if symbol not in glob['hmm_models']:
        return 0
    model = glob['hmm_models'][symbol]
    scaler = glob['hmm_scalers'][symbol]
    df = df_dict['15m'].copy()
    ret = df['close'].pct_change().dropna().values[-50:].reshape(-1, 1)
    if len(ret) < 10:
        return 0
    ret_scaled = scaler.transform(ret)
    states = model.predict(ret_scaled)
    return states[-1]

def start_ws_fetcher() -> None:
    """启动WebSocket数据获取线程"""
    symbols = st.session_state.current_symbols
    timeframes = list(set(CONFIG.timeframes + CONFIG.confirm_timeframes))
    fetcher = WebSocketFetcher(symbols, timeframes)
    fetcher.start()
    st.session_state.ws_fetcher = fetcher
    st.session_state.ws_data_queue = fetcher.data_queue
    log_execution("WebSocket数据获取器已启动")

# ==================== 辅助函数 ====================
def get_kelly_fraction(lookback: int = 50) -> float:
    trades = st.session_state.trade_log[-lookback:]
    if len(trades) < 10:
        return CONFIG.risk_per_trade
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    if len(wins) == 0 or len(losses) == 0:
        return CONFIG.risk_per_trade
    p = len(wins) / len(df)
    avg_win = wins['pnl'].mean()
    avg_loss = abs(losses['pnl'].mean())
    b = avg_win / avg_loss if avg_loss != 0 else 1
    kelly = (p * b - (1 - p)) / b
    kelly = max(0, min(kelly, 0.25))
    return kelly * 0.3

def dynamic_kelly_fraction() -> float:
    return get_kelly_fraction()

# ==================== 会话状态初始化 ====================
def init_session_state() -> None:
    """初始化Streamlit会话状态，从数据库恢复持久数据，并加载模型"""
    equity_df = load_from_db('equity_curve', limit=500)
    equity_curve = deque(maxlen=500)
    if not equity_df.empty:
        for _, row in equity_df.iterrows():
            try:
                t = pd.to_datetime(row['time'])
                equity_curve.append({'time': t, 'equity': float(row['equity'])})
            except:
                pass

    regime_stats = {}
    regime_df = load_from_db('regime_stats')
    if not regime_df.empty:
        for _, row in regime_df.iterrows():
            regime_stats[row['regime']] = {
                'trades': int(row['trades']),
                'wins': int(row['wins']),
                'total_pnl': float(row['total_pnl'])
            }

    consistency_stats = {'backtest': {}, 'live': {}}
    cons_df = load_from_db('consistency_stats')
    if not cons_df.empty:
        for _, row in cons_df.iterrows():
            typ = row['type']
            consistency_stats[typ] = {
                'trades': int(row['trades']),
                'avg_slippage': float(row['avg_slippage']),
                'win_rate': float(row['win_rate'])
            }

    defaults = {
        'account_balance': 10000.0,
        'daily_pnl': 0.0,
        'daily_risk_consumed': 0.0,
        'peak_balance': 10000.0,
        'consecutive_losses': 0,
        'daily_trades': 0,
        'trade_log': [],
        'positions': {},
        'auto_enabled': True,
        'pause_until': None,
        'exchange': None,
        'net_value_history': [],
        'equity_curve': equity_curve,
        'last_signal_time': {},
        'current_symbols': ['ETH/USDT', 'BTC/USDT'],
        'telegram_token': None,
        'telegram_chat_id': None,
        'backtest_results': None,
        'wf_results': None,
        'param_sensitivity': None,
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
        'binance_api_key': BINANCE_API_KEY,
        'binance_secret_key': BINANCE_SECRET_KEY,
        'fear_greed': 50,
        'market_regime': MarketRegime.RANGE,
        'multi_df': {},
        'performance_metrics': {},
        'mode': 'live',
        'factor_ic_stats': {},
        'symbol_current_prices': {},
        'daily_returns': deque(maxlen=252),
        'cov_matrix': None,
        'cov_matrix_cache': {'key': None, 'matrix': None},
        'slippage_records': [],
        'regime_stats': regime_stats,
        'consistency_stats': consistency_stats,
        'aggressive_mode': False,
        'dynamic_max_daily_trades': 9999,
        'var_method': CONFIG.var_method.value,
        'funding_rates': {},
        'funding_rates_history': defaultdict(list),
        'ml_factor_scores': {},
        'volcone': None,
        'adaptive_params': {},
        'sector_exposure': {},
        'hmm_regime': None,
        'calibration_model': None,
        'walk_forward_index': 0,
        'advanced_metrics': {},
        'backtest_results_old': None,
        'backtest_results_new': None,
        'pending_orders': {},
        'ws_fetcher': None,
        'ws_data_queue': None,
        'best_params': None,
        'last_training_time': time.time(),
        'globals': {
            'factor_weights': {f: 1.0 for f in FACTOR_NAMES},
            'ic_decay_records': {f: deque(maxlen=200) for f in FACTOR_NAMES},
            'factor_corr_matrix': None,
            'ml_models': {},
            'ml_scalers': {},
            'ml_feature_cols': {},
            'ml_last_train': {},
            'ml_calibrators': {},
            'ml_calibrators_count': {},
            'volcone_cache': {},
            'hmm_models': {},
            'hmm_scalers': {},
            'hmm_last_train': {},
            'ic_cache': {},
        }
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    load_models_from_disk()

    if WS_AVAILABLE and not st.session_state.use_simulated_data and st.session_state.ws_fetcher is None:
        start_ws_fetcher()

def load_models_from_disk() -> None:
    """加载所有持久化的机器学习模型"""
    glob = st.session_state.globals
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith('ml_model_') and fname.endswith('.pkl'):
            symbol = fname[9:-4]
            path = os.path.join(MODEL_DIR, fname)
            try:
                data = joblib.load(path)
                glob['ml_models'][symbol] = data['model']
                glob['ml_scalers'][symbol] = data['scaler']
                glob['ml_feature_cols'][symbol] = data['feature_cols']
                glob['ml_last_train'][symbol] = data.get('timestamp', 0)
            except Exception as e:
                logger.error(f"加载ML模型 {symbol} 失败: {e}")
        elif fname.startswith('calib_') and fname.endswith('.pkl'):
            symbol = fname[6:-4]
            path = os.path.join(MODEL_DIR, fname)
            try:
                data = joblib.load(path)
                glob['ml_calibrators'][symbol] = data['calibrator']
                glob['ml_calibrators_count'][symbol] = data.get('count', 0)
            except Exception as e:
                logger.error(f"加载校准模型 {symbol} 失败: {e}")

def save_ml_model(symbol: str) -> None:
    glob = st.session_state.globals
    if symbol in glob['ml_models']:
        data = {
            'model': glob['ml_models'][symbol],
            'scaler': glob['ml_scalers'][symbol],
            'feature_cols': glob['ml_feature_cols'][symbol],
            'timestamp': time.time()
        }
        path = os.path.join(MODEL_DIR, f"ml_model_{symbol}.pkl")
        joblib.dump(data, path)

def save_calibration_model(symbol: str) -> None:
    glob = st.session_state.globals
    if symbol in glob['ml_calibrators']:
        data = {
            'calibrator': glob['ml_calibrators'][symbol],
            'count': glob['ml_calibrators_count'].get(symbol, 0)
        }
        path = os.path.join(MODEL_DIR, f"calib_{symbol}.pkl")
        joblib.dump(data, path)

def check_and_fix_anomalies() -> None:
    fixed = False
    if st.session_state.account_balance > CONFIG.max_reasonable_balance or st.session_state.account_balance < 0:
        log_error(f"检测到异常余额 {st.session_state.account_balance:.2f}，自动重置为10000")
        st.session_state.account_balance = 10000.0
        st.session_state.peak_balance = 10000.0
        fixed = True
    if abs(st.session_state.daily_pnl) > st.session_state.account_balance * CONFIG.max_reasonable_daily_pnl_ratio:
        log_error(f"检测到异常每日盈亏 {st.session_state.daily_pnl:.2f}，自动重置为0")
        st.session_state.daily_pnl = 0.0
        fixed = True
    if abs(st.session_state.daily_risk_consumed) > st.session_state.account_balance * 0.5:
        log_error(f"检测到异常风险消耗 {st.session_state.daily_risk_consumed:.2f}，自动重置为0")
        st.session_state.daily_risk_consumed = 0.0
        fixed = True
    if fixed:
        conn = get_db_conn()
        with DB_LOCK:
            c = conn.cursor()
            c.execute("DELETE FROM trades")
            c.execute("DELETE FROM equity_curve")
            c.execute("DELETE FROM regime_stats")
            c.execute("DELETE FROM consistency_stats")
            c.execute("DELETE FROM slippage_log")
            conn.commit()
        st.session_state.trade_log = []
        st.session_state.equity_curve.clear()
        st.rerun()

def send_telegram(msg: str, msg_type: str = "info", image: Optional[Any] = None) -> None:
    token = st.session_state.get('telegram_token')
    chat_id = st.session_state.get('telegram_chat_id')
    if not token or not chat_id:
        return
    try:
        if image is not None:
            import io
            buf = io.BytesIO()
            image.write_image(buf, format='png')
            buf.seek(0)
            files = {'photo': buf}
            requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                          data={'chat_id': chat_id}, files=files, timeout=5)
        else:
            prefix = {
                'info': 'ℹ️ ',
                'signal': '📊 ',
                'risk': '⚠️ ',
                'trade': '🔄 '
            }.get(msg_type, '')
            full_msg = f"{prefix}{msg}"
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": full_msg}, timeout=3)
    except Exception as e:
        logger.warning(f"Telegram发送失败: {e}")

def update_performance_metrics() -> None:
    trades = st.session_state.trade_log[-100:]
    if len(trades) < 10:
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

def calculate_advanced_metrics() -> Dict[str, float]:
    trades_df = pd.DataFrame(st.session_state.trade_log)
    equity_df = pd.DataFrame(list(st.session_state.equity_curve))
    if len(trades_df) < 5 or len(equity_df) < 5:
        return {}
    equity_df['time'] = pd.to_datetime(equity_df['time'])
    equity_df = equity_df.set_index('time').sort_index()
    if len(equity_df) >= 2:
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days < 1:
            days = 1
    else:
        days = 1
    returns = equity_df['equity'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-6
    sortino = returns.mean() / downside_std * np.sqrt(252)
    total_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]
    max_dd = (equity_df['equity'].cummax() - equity_df['equity']).max() / equity_df['equity'].cummax().max()
    calmar = total_return / max_dd if max_dd != 0 else 0
    win_rate = (trades_df['pnl'] > 0).mean()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if any(trades_df['pnl'] > 0) else 0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if any(trades_df['pnl'] < 0) else 1
    profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'total_return': total_return * 100,
        'max_drawdown': max_dd * 100
    }

def current_equity() -> float:
    balance = st.session_state.account_balance
    floating = 0.0
    for sym, pos in st.session_state.positions.items():
        if sym in st.session_state.symbol_current_prices:
            floating += pos.pnl(st.session_state.symbol_current_prices[sym])
    return balance + floating

def calculate_drawdown() -> Tuple[float, float]:
    if len(st.session_state.equity_curve) < 2:
        return 0.0, 0.0
    df = pd.DataFrame(list(st.session_state.equity_curve))
    peak = df['equity'].cummax()
    dd = (peak - df['equity']) / peak * 100
    current_dd = dd.iloc[-1]
    max_dd = dd.max()
    return current_dd, max_dd

def record_equity_point() -> None:
    equity = current_equity()
    now = datetime.now()
    st.session_state.equity_curve.append({'time': now, 'equity': equity})
    append_to_db('equity_curve', {'time': now.isoformat(), 'equity': equity})

def update_regime_stats(regime: MarketRegime, pnl: float) -> None:
    key = regime.value
    if key not in st.session_state.regime_stats:
        st.session_state.regime_stats[key] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
    st.session_state.regime_stats[key]['trades'] += 1
    if pnl > 0:
        st.session_state.regime_stats[key]['wins'] += 1
    st.session_state.regime_stats[key]['total_pnl'] += pnl
    rows = []
    for k, v in st.session_state.regime_stats.items():
        rows.append({'regime': k, 'trades': v['trades'], 'wins': v['wins'], 'total_pnl': v['total_pnl']})
    conn = get_db_conn()
    with DB_LOCK:
        c = conn.cursor()
        for row in rows:
            c.execute("REPLACE INTO regime_stats (regime, trades, wins, total_pnl) VALUES (?,?,?,?)",
                      (row['regime'], row['trades'], row['wins'], row['total_pnl']))
        conn.commit()

def update_consistency_stats(is_backtest: bool, slippage: float, win: bool) -> None:
    key = 'backtest' if is_backtest else 'live'
    stats = st.session_state.consistency_stats.get(key, {})
    stats['trades'] = stats.get('trades', 0) + 1
    stats['avg_slippage'] = (stats.get('avg_slippage', 0.0) * (stats['trades'] - 1) + slippage) / stats['trades']
    if win:
        stats['wins'] = stats.get('wins', 0) + 1
    stats['win_rate'] = stats.get('wins', 0) / stats['trades'] if stats['trades'] > 0 else 0
    st.session_state.consistency_stats[key] = stats
    rows = []
    for typ, s in st.session_state.consistency_stats.items():
        rows.append({
            'type': typ,
            'trades': s.get('trades', 0),
            'avg_slippage': s.get('avg_slippage', 0.0),
            'win_rate': s.get('win_rate', 0.0)
        })
    conn = get_db_conn()
    with DB_LOCK:
        c = conn.cursor()
        for row in rows:
            c.execute("REPLACE INTO consistency_stats (type, trades, avg_slippage, win_rate) VALUES (?,?,?,?)",
                      (row['type'], row['trades'], row['avg_slippage'], row['win_rate']))
        conn.commit()

# ==================== 风险预算检查 ====================
def check_and_reset_daily() -> None:
    today = datetime.now().date()
    if st.session_state.get('last_trade_date') != today:
        st.session_state.daily_trades = 0
        st.session_state.daily_pnl = 0.0
        st.session_state.daily_risk_consumed = 0.0
        st.session_state.last_trade_date = today
        log_execution("新的一天，重置每日数据")

def check_risk_budget() -> bool:
    budget = st.session_state.account_balance * CONFIG.daily_risk_budget_ratio
    if st.session_state.daily_risk_consumed >= budget:
        return False
    return True

# ==================== 自适应ATR倍数 ====================
def adaptive_atr_multiplier(price_series: pd.Series, timeframe_minutes: int = 15) -> float:
    if len(price_series) < CONFIG.adapt_window:
        return CONFIG.atr_multiplier_base
    returns = price_series.pct_change().dropna()
    periods_per_year = 365 * 24 * 60 / timeframe_minutes
    vol = returns.std() * np.sqrt(periods_per_year)
    volcone = get_volcone(returns, timeframe_minutes)
    current_vol_percentile = np.mean(vol <= volcone['percentiles']) if volcone else 0.5
    factor = 1.5 - current_vol_percentile
    new_mult = CONFIG.atr_multiplier_base * factor
    return np.clip(new_mult, CONFIG.atr_multiplier_min, CONFIG.atr_multiplier_max)

def get_volcone(returns: pd.Series, timeframe_minutes: int) -> dict:
    glob = st.session_state.globals
    window = min(CONFIG.volcone_window, len(returns))
    if window < 10:
        return {'percentiles': np.array([0.5] * len(CONFIG.volcone_percentiles))}
    key_data = returns.iloc[-window:].values.tobytes()
    key = hashlib.md5(key_data).hexdigest() + str(timeframe_minutes)
    if key in glob['volcone_cache']:
        return glob['volcone_cache'][key]
    windows = [5, 10, 20, 40, 60]
    volcone = {}
    vols = []
    periods_per_year = 365 * 24 * 60 / timeframe_minutes
    for w in windows:
        if len(returns) < w:
            continue
        roll_vol = returns.rolling(w).std() * np.sqrt(periods_per_year / w)
        volcone[f'vol_{w}'] = roll_vol.dropna().quantile(CONFIG.volcone_percentiles).to_dict()
        vols.extend(roll_vol.dropna().values)
    if not vols:
        volcone['percentiles'] = np.array([0.5] * len(CONFIG.volcone_percentiles))
    else:
        volcone['percentiles'] = np.percentile(vols, [p*100 for p in CONFIG.volcone_percentiles])
    glob['volcone_cache'][key] = volcone
    return volcone

# ==================== Regime检测 ====================
def detect_market_regime_advanced(df_dict: Dict[str, pd.DataFrame], symbol: str) -> MarketRegime:
    if CONFIG.regime_detection_method == 'hmm' and HMM_AVAILABLE:
        state = detect_hmm_regime(symbol, df_dict)
        mapping = {0: MarketRegime.RANGE, 1: MarketRegime.TREND, 2: MarketRegime.PANIC}
        return mapping.get(state, MarketRegime.RANGE)
    else:
        return detect_market_regime_traditional(df_dict)

def detect_market_regime_traditional(df_dict: Dict[str, pd.DataFrame]) -> MarketRegime:
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

# ==================== 机器学习因子（修复数据泄露，使用历史窗口训练）====================
def train_ml_model_cost_aware(symbol: str, df_dict: Dict[str, pd.DataFrame], train_end_idx: int = -1) -> Tuple[Any, Any, List[str]]:
    """
    使用截至 train_end_idx 的数据训练ML模型。
    若 train_end_idx 为 -1，则使用全部数据（仅用于初始训练，实时训练应传入当前索引）。
    注意：标签使用未来5期收益，因此训练数据需丢弃最后5行以避免泄露。
    """
    df = df_dict['15m'].iloc[:train_end_idx].copy() if train_end_idx != -1 else df_dict['15m'].copy()
    # 构建特征
    feature_cols = ['ema20', 'ema50', 'rsi', 'macd_diff', 'bb_width', 'volume_ratio', 'adx', 'atr']
    df = df.dropna(subset=feature_cols + ['close'])
    if len(df) < CONFIG.ml_window:
        return None, None, []
    # 添加滞后特征
    for col in feature_cols:
        for lag in [1,2,3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    # 构建标签：未来5期收益率
    future_ret = df['close'].shift(-5) / df['close'] - 1
    if CONFIG.cost_aware_training:
        vol = df['atr'].rolling(20).mean() / df['close']
        cost_estimate = vol * 0.001
        target = future_ret - cost_estimate.shift(-5)
    else:
        target = future_ret
    df['target'] = target
    # 丢弃包含未来信息的行（最后5行）
    df = df.iloc[:-5].copy()
    df = df.dropna()
    if len(df) < 100:
        return None, None, []
    all_feature_cols = []
    for col in feature_cols:
        all_feature_cols.append(col)
        for lag in [1,2,3]:
            all_feature_cols.append(f'{col}_lag{lag}')
    X = df[all_feature_cols]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(
        n_estimators=CONFIG.ml_n_estimators,
        max_depth=CONFIG.ml_max_depth,
        random_state=42
    )
    model.fit(X_scaled, y)
    return model, scaler, all_feature_cols

def get_ml_factor(symbol: str, df_dict: Dict[str, pd.DataFrame]) -> float:
    if not CONFIG.use_ml_factor:
        return 0.0
    glob = st.session_state.globals
    if symbol not in glob['ml_models']:
        return 0.0
    model = glob['ml_models'][symbol]
    scaler = glob['ml_scalers'][symbol]
    feature_cols = glob['ml_feature_cols'].get(symbol, [])
    if not feature_cols:
        return 0.0
    df = df_dict['15m'].copy()
    required_len = max([int(col.split('_lag')[-1]) if '_lag' in col else 0 for col in feature_cols]) + 1
    if len(df) < required_len:
        return 0.0
    last_idx = -1
    data = {}
    feature_cols_original = ['ema20', 'ema50', 'rsi', 'macd_diff', 'bb_width', 'volume_ratio', 'adx', 'atr']
    for col in feature_cols_original:
        data[col] = df[col].iloc[last_idx] if col in df.columns else np.nan
        for lag in [1,2,3]:
            lag_col = f'{col}_lag{lag}'
            if len(df) > lag:
                data[lag_col] = df[col].iloc[-lag-1]
            else:
                data[lag_col] = np.nan
    X = pd.DataFrame([data])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return np.tanh(pred * 10)

# ==================== 统一模型训练调度器（修复数据泄露）====================
def schedule_model_training() -> None:
    """每小时执行一次所有币种的HMM和ML模型训练，使用历史窗口避免未来信息"""
    now = time.time()
    last = st.session_state.get('last_training_time', 0)
    if now - last < CONFIG.training_schedule_interval:
        return
    st.session_state.last_training_time = now
    log_execution("开始统一模型训练")
    symbols = st.session_state.current_symbols
    for sym in symbols:
        if sym not in st.session_state.multi_df:
            continue
        df_dict = st.session_state.multi_df[sym]
        # 使用当前时刻之前的数据训练（即全部历史，但注意模型训练时内部会丢弃最后5行）
        if CONFIG.regime_detection_method == 'hmm' and HMM_AVAILABLE:
            result = train_hmm(sym, df_dict)  # HMM 内部也会使用全部数据，但HMM不涉及未来，可以接受
            if result is not None:
                st.session_state.globals['hmm_models'][sym], st.session_state.globals['hmm_scalers'][sym] = result
                st.session_state.globals['hmm_last_train'][sym] = now
        if CONFIG.use_ml_factor:
            model, scaler, features = train_ml_model_cost_aware(sym, df_dict)
            if model is not None:
                st.session_state.globals['ml_models'][sym] = model
                st.session_state.globals['ml_scalers'][sym] = scaler
                st.session_state.globals['ml_feature_cols'][sym] = features
                st.session_state.globals['ml_last_train'][sym] = now
                save_ml_model(sym)
    log_execution("统一模型训练完成")

# ==================== 概率校准 ====================
def train_calibration_model(symbol: str) -> None:
    if not CONFIG.use_prob_calibration:
        return
    glob = st.session_state.globals
    df = load_from_db('trades', condition=f"WHERE symbol=?", params=(symbol,))
    if len(df) < 20 or 'raw_prob' not in df.columns:
        return
    raw_probs = df['raw_prob'].values
    true_labels = (df['pnl'] > 0).astype(int).values
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(raw_probs, true_labels)
    glob['ml_calibrators'][symbol] = ir
    glob['ml_calibrators_count'][symbol] = len(df)
    save_calibration_model(symbol)
    log_execution(f"{symbol} 概率校准模型已更新（基于{len(df)}笔交易）")

def apply_calibration(symbol: str, raw_prob: float) -> float:
    if not CONFIG.use_prob_calibration:
        return raw_prob
    glob = st.session_state.globals
    if symbol not in glob['ml_calibrators']:
        return raw_prob
    calibrator = glob['ml_calibrators'][symbol]
    try:
        return float(calibrator.predict([raw_prob])[0])
    except:
        return raw_prob

# ==================== 因子权重定时更新（避免样本量不足）====================
def update_factor_weights_scheduled() -> None:
    """从数据库中读取历史IC记录，定时更新因子权重"""
    glob = st.session_state.globals
    ic_dict = {}
    for factor, deq in glob['ic_decay_records'].items():
        if len(deq) >= 10:
            ic_dict[factor] = list(deq)
    if not ic_dict:
        return
    prior_mean = 1.0
    prior_strength = CONFIG.bayesian_prior_strength
    for factor, ic_list in ic_dict.items():
        sample_mean = np.mean(ic_list)
        n = len(ic_list)
        posterior_mean = (prior_strength * prior_mean + n * sample_mean) / (prior_strength + n)
        glob['factor_weights'][factor] = max(0.1, posterior_mean)
    log_execution("因子权重已定时更新")

def update_factor_ic_stats(ic_records: Dict[str, List[float]]) -> None:
    stats = {}
    for factor, ic_list in ic_records.items():
        if len(ic_list) > 5:
            mean_ic = np.mean(ic_list)
            std_ic = np.std(ic_list)
            ir = mean_ic / max(std_ic, 0.001)
            t_stat, p_value = ttest_1samp(ic_list, 0)
            stats[factor] = {'mean': mean_ic, 'std': std_ic, 'ir': ir, 'p_value': p_value}
    st.session_state.factor_ic_stats = stats

def calculate_ic(df: pd.DataFrame, factor_name: str) -> float:
    glob = st.session_state.globals
    try:
        df_hash = pd.util.hash_pandas_object(df).sum()
    except:
        df_hash = id(df)
    key = (df_hash, factor_name)
    if len(glob['ic_cache']) > 500:
        keys = list(glob['ic_cache'].keys())
        for k in keys[:250]:
            del glob['ic_cache'][k]
    if key in glob['ic_cache']:
        return glob['ic_cache'][key]
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
    glob['ic_cache'][key] = ic
    return ic

def calculate_cov_matrix(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]], window: int = 50) -> Optional[np.ndarray]:
    if len(symbols) < 2:
        return None
    hash_input = str(sorted(symbols)) + str(window)
    for sym in symbols:
        df = data_dicts[sym]['15m']['close'].iloc[-window:]
        hash_input += str(pd.util.hash_pandas_object(df).sum())
    cache_key = hashlib.md5(hash_input.encode()).hexdigest()
    if st.session_state.cov_matrix_cache.get('key') == cache_key:
        return st.session_state.cov_matrix_cache['matrix']
    returns_list = []
    min_len = None
    for sym in symbols:
        df = data_dicts[sym]['15m']['close'].iloc[-window:]
        ret = df.pct_change().dropna().values
        if len(ret) < window // 2:
            return None
        returns_list.append(ret)
        if min_len is None or len(ret) < min_len:
            min_len = len(ret)
    if min_len < 10:
        return None
    aligned_returns = [ret[-min_len:] for ret in returns_list]
    returns_array = np.array(aligned_returns)
    if returns_array.shape[0] != len(symbols):
        return None
    cov = np.cov(returns_array)
    st.session_state.cov_matrix_cache = {'key': cache_key, 'matrix': cov}
    return cov

def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / vols
    w0 = inv_vol / np.sum(inv_vol)
    return w0

def black_litterman_weights(cov: np.ndarray, market_cap_weights: Optional[np.ndarray] = None, views: Optional[Dict] = None) -> np.ndarray:
    if market_cap_weights is None:
        market_cap_weights = np.ones(cov.shape[0]) / cov.shape[0]
    pi = CONFIG.black_litterman_tau * cov @ market_cap_weights
    return market_cap_weights

def allocate_with_risk_budget(symbols: List[str], cov: np.ndarray, balance: float, signals: Dict[str, Tuple]) -> Dict[str, float]:
    if CONFIG.risk_budget_method == 'risk_parity':
        weights = risk_parity_weights(cov)
    elif CONFIG.risk_budget_method == 'black_litterman':
        weights = black_litterman_weights(cov)
    else:
        weights = np.ones(len(symbols)) / len(symbols)
    allocations = {}
    for i, sym in enumerate(symbols):
        if sym in signals and signals[sym][0] != 0 and signals[sym][1] >= SignalStrength.WEAK.value:
            allocations[sym] = balance * weights[i]
        else:
            allocations[sym] = 0.0
    return allocations

def advanced_slippage_prediction(price: float, size: float, volume_20: float, volatility: float, imbalance: float) -> float:
    if isinstance(volume_20, pd.Series):
        volume_20 = float(volume_20.iloc[-1]) if not volume_20.empty else 1.0
    else:
        volume_20 = float(volume_20) if volume_20 != 0 else 1.0
    base_slippage = dynamic_slippage(price, size, volume_20, volatility, imbalance)
    market_impact = (size / max(volume_20, 1)) ** 0.5 * volatility * price * 0.3
    return base_slippage + market_impact

def dynamic_slippage(price: float, size: float, volume: float, volatility: float, imbalance: float = 0.0) -> float:
    if isinstance(volume, pd.Series):
        volume = float(volume.iloc[-1]) if not volume.empty else 1.0
    else:
        volume = float(volume) if volume != 0 else 1.0
    base = price * CONFIG.slippage_base
    impact = CONFIG.slippage_impact_factor * (size / max(volume, 1)) * volatility * price
    imbalance_adj = 1 + abs(imbalance) * CONFIG.slippage_imbalance_factor
    return (base + impact) * imbalance_adj

def portfolio_var(weights: np.ndarray, cov: np.ndarray, confidence: float = 0.95, method: str = "HISTORICAL", historical_returns: Optional[np.ndarray] = None) -> float:
    if weights is None or cov is None or len(weights) == 0:
        return 0.0
    if method == "HISTORICAL" and historical_returns is not None and historical_returns.shape[1] > 20:
        port_rets = weights @ historical_returns
        var = np.percentile(port_rets, (1 - confidence) * 100)
        return abs(var)
    elif method == "EXTREME" and historical_returns is not None and historical_returns.shape[1] > 50:
        port_rets = weights @ historical_returns
        threshold = np.percentile(port_rets, 10)
        excess = port_rets[port_rets < threshold] - threshold
        if len(excess) < 10:
            return 0.0
        params = genpareto.fit(excess)
        var = threshold + genpareto.ppf(1 - confidence, *params)
        return abs(var)
    elif method == "VOLCONE":
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        var = port_vol * norm.ppf(confidence)
        return abs(var)
    else:
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        var = port_vol * norm.ppf(confidence)
        return abs(var)

def portfolio_cvar(weights: np.ndarray, historical_returns: np.ndarray, confidence: float = 0.95) -> float:
    if historical_returns is None or historical_returns.shape[1] < 20:
        return 0.0
    port_rets = weights @ historical_returns
    var = np.percentile(port_rets, (1 - confidence) * 100)
    cvar = port_rets[port_rets <= var].mean()
    return abs(cvar)

def get_dynamic_var_limit() -> float:
    base_limit = CONFIG.portfolio_risk_target * 100
    if st.session_state.get('aggressive_mode', False):
        base_limit = CONFIG.var_aggressive_threshold
    if is_night_time():
        base_limit *= CONFIG.night_risk_multiplier
    return base_limit

def is_night_time() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(pytz.utc).astimezone(eastern)
    hour = now_eastern.hour
    return CONFIG.night_start_hour <= hour < CONFIG.night_end_hour

def funding_rate_blocked(symbol: str, direction: int) -> bool:
    rate = st.session_state.funding_rates.get(symbol, 0.0)
    if abs(rate) > CONFIG.funding_rate_threshold:
        if (rate > 0 and direction == 1) or (rate < 0 and direction == -1):
            log_execution(f"资金费率阻止开仓 {symbol} 方向 {'多' if direction==1 else '空'} 费率 {rate*100:.4f}%")
            return True
    return False

def is_range_market(df_dict: Dict[str, pd.DataFrame]) -> bool:
    if '15m' not in df_dict:
        return False
    df = df_dict['15m']
    if df.empty:
        return False
    last = df.iloc[-1]
    try:
        if hasattr(last, 'get') and last.get('bb_width') is not None and not pd.isna(last.get('bb_width')):
            if last['bb_width'] < CONFIG.bb_width_threshold:
                return True
        if hasattr(last, 'get') and last.get('rsi') is not None and not pd.isna(last.get('rsi')):
            if CONFIG.rsi_range_low < last['rsi'] < CONFIG.rsi_range_high:
                return True
        if hasattr(last, 'get') and last.get('adx') is not None and not pd.isna(last.get('adx')):
            if last['adx'] < 25:
                return True
    except Exception as e:
        log_error(f"is_range_market 判断出错: {e}")
        return False
    return False

def multi_timeframe_confirmation(df_dict: Dict[str, pd.DataFrame], direction: int) -> bool:
    count = 0
    for tf in CONFIG.confirm_timeframes:
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

def can_open_position(regime: MarketRegime) -> bool:
    return regime in CONFIG.regime_allow_trade

def update_factor_correlation(ic_records: Dict[str, List[float]]) -> None:
    glob = st.session_state.globals
    if len(ic_records) < 2:
        return
    all_factors = list(glob['factor_weights'].keys())
    df_dict = {}
    for f in all_factors:
        if f in ic_records and ic_records[f]:
            df_dict[f] = pd.Series(ic_records[f])
        else:
            df_dict[f] = pd.Series([np.nan])
    ic_df = pd.DataFrame(df_dict)
    corr = ic_df.corr().fillna(0)
    glob['factor_corr_matrix'] = corr.values

def apply_factor_correlation_penalty() -> None:
    glob = st.session_state.globals
    if glob['factor_corr_matrix'] is None:
        return
    factors = list(glob['factor_weights'].keys())
    n = len(factors)
    corr_mat = glob['factor_corr_matrix']
    if corr_mat.shape[0] < n or corr_mat.shape[1] < n:
        return
    for i in range(n):
        for j in range(i+1, n):
            if corr_mat[i, j] > CONFIG.factor_corr_threshold:
                glob['factor_weights'][factors[i]] *= CONFIG.factor_corr_penalty
                glob['factor_weights'][factors[j]] *= CONFIG.factor_corr_penalty

def eliminate_poor_factors() -> None:
    glob = st.session_state.globals
    for factor, stats in st.session_state.factor_ic_stats.items():
        if stats['p_value'] > CONFIG.factor_eliminate_pvalue and stats['mean'] < CONFIG.factor_eliminate_ic and len(glob['ic_decay_records'][factor]) > 30:
            glob['factor_weights'][factor] = CONFIG.factor_min_weight
            log_execution(f"因子淘汰：{factor} 权重降至{CONFIG.factor_min_weight}")

def generate_simulated_data(symbol: str, limit: int = 2000) -> Dict[str, pd.DataFrame]:
    cache_key = f"sim_data_{symbol}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
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
    for tf in ['1m', '5m', '1h', '4h', '1d']:
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
    st.session_state[cache_key] = data_dict
    return data_dict

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    min_periods = max(20, CONFIG.bb_window)
    if len(df) < min_periods:
        df['ema20'] = np.nan
        df['ema50'] = np.nan
        df['ema200'] = np.nan
        df['rsi'] = np.nan
        df['atr'] = np.nan
        df['atr_ma'] = np.nan
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_diff'] = np.nan
        df['adx'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_width'] = np.nan
        df['bb_factor'] = np.nan
        df['volume_sma'] = np.nan
        df['volume_ratio'] = np.nan
        df['trend_factor'] = np.nan
        df['future_ret'] = np.nan
        return df

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
    if len(df) >= CONFIG.bb_window:
        bb = ta.volatility.BollingerBands(df['close'], window=CONFIG.bb_window, window_dev=2)
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed() -> int:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        return int(r.json()['data'][0]['value'])
    except Exception:
        return 50

@st.cache_resource
def get_fetcher() -> 'AggregatedDataFetcher':
    return AggregatedDataFetcher()

class SignalEngine:
    def __init__(self):
        pass

    def detect_market_regime(self, df_dict: Dict[str, pd.DataFrame], symbol: str) -> MarketRegime:
        return detect_market_regime_advanced(df_dict, symbol)

    def calc_signal(self, df_dict: Dict[str, pd.DataFrame], symbol: str) -> Tuple[int, float]:
        if not df_dict:
            return 0, 0.0
        return self._calc_signal_impl(df_dict, symbol)

    def _calc_signal_impl(self, df_dict: Dict[str, pd.DataFrame], symbol: str) -> Tuple[int, float]:
        glob = st.session_state.globals
        total_score = 0
        total_weight = 0
        tf_votes = []
        regime = st.session_state.market_regime
        ic_dict = {}

        try:
            range_penalty = 0.5 if is_range_market(df_dict) else 1.0
        except Exception as e:
            log_error(f"is_range_market 调用异常: {e}")
            range_penalty = 1.0

        for tf, df in df_dict.items():
            if df.empty or len(df) < 2:
                continue
            last = df.iloc[-1]
            weight = CONFIG.timeframe_weights.get(tf, 1) * range_penalty
            if regime == MarketRegime.TREND:
                if tf in ['4h', '1d']:
                    weight *= 1.5
            elif regime == MarketRegime.RANGE:
                if tf in ['15m', '1h']:
                    weight *= 1.3
            if pd.isna(last.get('ema20', np.nan)):
                continue

            factor_scores = {}
            if last['close'] > last['ema20']:
                factor_scores['trend'] = 1 * glob['factor_weights']['trend']
            elif last['close'] < last['ema20']:
                factor_scores['trend'] = -1 * glob['factor_weights']['trend']
            else:
                factor_scores['trend'] = 0

            if last['rsi'] > 70:
                factor_scores['rsi'] = -0.7 * glob['factor_weights']['rsi']
            elif last['rsi'] < 30:
                factor_scores['rsi'] = 0.7 * glob['factor_weights']['rsi']
            else:
                factor_scores['rsi'] = 0

            if last['macd_diff'] > 0:
                factor_scores['macd'] = 0.8 * glob['factor_weights']['macd']
            elif last['macd_diff'] < 0:
                factor_scores['macd'] = -0.8 * glob['factor_weights']['macd']
            else:
                factor_scores['macd'] = 0

            if not pd.isna(last.get('bb_upper')):
                if last['close'] > last['bb_upper']:
                    factor_scores['bb'] = -0.5 * glob['factor_weights']['bb']
                elif last['close'] < last['bb_lower']:
                    factor_scores['bb'] = 0.5 * glob['factor_weights']['bb']
                else:
                    factor_scores['bb'] = 0
            else:
                factor_scores['bb'] = 0

            if not pd.isna(last.get('volume_ratio')):
                factor_scores['volume'] = (1.2 if last['volume_ratio'] > 1.5 else 0) * glob['factor_weights']['volume']
            else:
                factor_scores['volume'] = 0

            adx = last.get('adx', 25)
            if pd.isna(adx):
                factor_scores['adx'] = 0
            else:
                factor_scores['adx'] = (0.3 if adx > 30 else -0.2 if adx < 20 else 0) * glob['factor_weights']['adx']

            if CONFIG.use_ml_factor:
                ml_score = get_ml_factor(symbol, df_dict)
                factor_scores['ml'] = ml_score * glob['factor_weights']['ml']

            if not pd.isna(last.get('orderbook_imbalance')):
                imbalance = last['orderbook_imbalance']
                factor_scores['imbalance'] = imbalance * glob['factor_weights'].get('imbalance', 0.5)
            else:
                factor_scores['imbalance'] = 0

            if not pd.isna(last.get('funding_rate_change')):
                funding_change = last['funding_rate_change']
                factor_scores['funding_change'] = np.tanh(funding_change * 100) * glob['factor_weights'].get('funding_change', 0.3)
            else:
                factor_scores['funding_change'] = 0

            for fname in factor_scores.keys():
                col = FACTOR_TO_COL.get(fname)
                if col and col in df.columns:
                    ic = calculate_ic(df, col)
                    if not np.isnan(ic):
                        if fname not in ic_dict:
                            ic_dict[fname] = []
                        ic_dict[fname].append(ic)
                        glob['ic_decay_records'][fname].append(ic)

            tf_score = sum(factor_scores.values()) * weight
            total_score += tf_score
            total_weight += weight
            if tf_score > 0:
                tf_votes.append(1)
            elif tf_score < 0:
                tf_votes.append(-1)

        update_factor_ic_stats(ic_dict)
        update_factor_correlation(ic_dict)
        apply_factor_correlation_penalty()
        eliminate_poor_factors()

        if total_weight == 0:
            return 0, 0.0
        max_possible = sum(CONFIG.timeframe_weights.values()) * 3.5
        prob_raw = min(1.0, abs(total_score) / max_possible) if max_possible > 0 else 0.5
        prob = 0.5 + 0.45 * prob_raw

        prob = apply_calibration(symbol, prob)

        direction_candidate = 1 if total_score > 0 else -1 if total_score < 0 else 0
        if direction_candidate != 0 and not multi_timeframe_confirmation(df_dict, direction_candidate):
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

class RiskManager:
    def __init__(self):
        pass

    def check_cooldown(self) -> bool:
        until = st.session_state.get('cooldown_until')
        return until is not None and datetime.now() < until

    def update_losses(self, win: bool, loss_amount: float = 0.0) -> None:
        if not win:
            st.session_state.consecutive_losses += 1
            st.session_state.daily_risk_consumed += abs(loss_amount)
            if st.session_state.consecutive_losses >= CONFIG.cooldown_losses:
                st.session_state.cooldown_until = datetime.now() + timedelta(hours=CONFIG.cooldown_hours)
        else:
            st.session_state.consecutive_losses = 0
            st.session_state.cooldown_until = None

    def check_circuit_breaker(self, atr_pct: float, fear_greed: int) -> bool:
        return atr_pct > CONFIG.circuit_breaker_atr or fear_greed <= CONFIG.circuit_breaker_fg_extreme[0] or fear_greed >= CONFIG.circuit_breaker_fg_extreme[1]

    def check_max_drawdown(self) -> bool:
        current_dd, _ = calculate_drawdown()
        return current_dd > CONFIG.max_drawdown_pct

    def calc_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        if len(returns) < 10:
            return 0.02
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)

    def get_adaptive_risk_multiplier(self) -> float:
        consecutive_losses = st.session_state.consecutive_losses
        current_dd, _ = calculate_drawdown()
        if consecutive_losses >= 4:
            return 0.0
        elif consecutive_losses == 3:
            mult = 0.4
        elif consecutive_losses == 2:
            mult = 0.6
        elif consecutive_losses == 1:
            mult = 0.8
        else:
            mult = 1.0
        if current_dd > 5.0:
            mult *= 0.3
        elif current_dd > 3.0:
            mult *= 0.5
        elif current_dd > 1.5:
            mult *= 0.7
        return max(0.0, mult)

    def calc_position_size(self, balance: float, prob: float, atr: float, price: float, price_history: pd.Series, is_aggressive: bool = False) -> float:
        if price <= 0 or prob < 0.5:
            return 0.0
        kelly_risk = get_kelly_fraction()
        risk_amount = balance * kelly_risk
        adaptive_mult = self.get_adaptive_risk_multiplier()
        risk_amount *= adaptive_mult
        if is_aggressive:
            risk_amount *= 1.5
        if atr == 0 or np.isnan(atr) or atr < price * CONFIG.min_atr_pct / 100:
            stop_distance = price * 0.01
        else:
            stop_distance = atr * adaptive_atr_multiplier(price_history, timeframe_minutes=15)
        size = risk_amount / stop_distance
        used_margin = sum([p.size * p.entry_price / CONFIG.max_leverage_global for p in st.session_state.positions.values()])
        available_balance = balance - used_margin
        max_size_by_leverage = available_balance * CONFIG.max_leverage_global / price
        size = min(size, max_size_by_leverage)
        return max(size, 0.001)

    def allocate_portfolio(self, symbol_signals: Dict[str, Tuple[int, float, float, float, pd.Series]], balance: float, cov: Optional[np.ndarray] = None) -> Dict[str, float]:
        # ========== 协方差稳定性处理 ==========
        cov = safe_cov_matrix(cov)

        if not symbol_signals:
            return {}
        if cov is None or len(symbol_signals) == 1:
            allocations = {}
            for sym, (direction, prob, atr, price, price_hist) in symbol_signals.items():
                is_aggressive = prob > 0.7 and st.session_state.get('aggressive_mode', False)
                size = self.calc_position_size(balance, prob, atr, price, price_hist, is_aggressive)
                allocations[sym] = size
            return allocations

        symbols_list = list(symbol_signals.keys())
        n = len(symbols_list)
        atr_list = [symbol_signals[sym][2] for sym in symbols_list]
        price_list = [symbol_signals[sym][3] for sym in symbols_list]
        prob_list = [symbol_signals[sym][1] for sym in symbols_list]

        risk_per_trade = balance * get_kelly_fraction()
        prob_mult = [max(0.5, min(1.5, p * 2)) for p in prob_list]
        risk_amounts = [risk_per_trade * mult for mult in prob_mult]
        nominal_weights = [risk_amounts[i] / (atr_list[i] if atr_list[i] > 0 else price_list[i]*0.01) for i in range(n)]

        total_nominal = sum(nominal_weights)
        if total_nominal == 0:
            return {sym: 0.0 for sym in symbols_list}
        init_weights = np.array(nominal_weights) / total_nominal

        port_vol = np.sqrt(np.dot(init_weights.T, np.dot(cov, init_weights)))
        if port_vol == 0:
            return {sym: 0.0 for sym in symbols_list}

        marginal_contrib = np.dot(cov, init_weights) / port_vol
        risk_contrib = init_weights * marginal_contrib
        target_rc = port_vol / n

        adjustment = target_rc / (risk_contrib + 1e-8)
        new_weights = init_weights * adjustment
        new_weights = new_weights / np.sum(new_weights)

        total_risk_available = risk_per_trade * sum(prob_mult)
        allocations = {}
        for i, sym in enumerate(symbols_list):
            sym_risk = total_risk_available * new_weights[i]
            stop_dist = atr_list[i] * adaptive_atr_multiplier(symbol_signals[sym][4]) if atr_list[i] > 0 else price_list[i]*0.01
            size = sym_risk / stop_dist
            allocations[sym] = size
        return allocations

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    CLOSED = "closed"
    CANCELED = "canceled"
    FAILED = "failed"

@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    status: OrderStatus = OrderStatus.PENDING
    filled: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    retries: int = 0
    last_check: datetime = None
    prob: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

def create_order_and_record(symbol: str, side: str, size: float, price: float, stop: float, take: float, prob: float) -> Optional[Order]:
    try:
        order_resp = st.session_state.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=size,
            params={'reduceOnly': False}
        )
        order_id = order_resp['id']
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            status=OrderStatus.OPEN,
            filled=order_resp.get('filled', 0.0),
            prob=prob,
            stop_loss=stop,
            take_profit=take
        )
        st.session_state.pending_orders[order_id] = order
        log_execution(f"订单已创建: {symbol} {side} {size} @ {price}")
        return order
    except Exception as e:
        log_error(f"创建订单失败: {e}")
        return None

def check_pending_orders() -> None:
    if not st.session_state.exchange:
        return
    for oid, order in list(st.session_state.pending_orders.items()):
        if order.status in [OrderStatus.CLOSED, OrderStatus.CANCELED, OrderStatus.FAILED]:
            continue
        try:
            fetched = st.session_state.exchange.fetch_order(oid)
            status = fetched['status']
            filled = fetched.get('filled', 0.0)
            if status == 'closed':
                if order.symbol not in st.session_state.positions:
                    pos = Position(
                        symbol=order.symbol,
                        direction=1 if order.side == 'buy' else -1,
                        entry_price=fetched['average'] or order.price,
                        entry_time=datetime.now(),
                        size=filled,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        initial_atr=0,
                        real=True,
                        prob=order.prob
                    )
                    st.session_state.positions[order.symbol] = pos
                    log_execution(f"轮询检测订单成交，创建持仓: {order.symbol}")
                del st.session_state.pending_orders[oid]
            elif status == 'partially_filled':
                order.filled = filled
                order.status = OrderStatus.PARTIALLY_FILLED
            elif status == 'canceled':
                if order.retries < 3:
                    new_order = create_order_and_record(order.symbol, order.side, order.size - order.filled, order.price, order.stop_loss, order.take_profit, order.prob)
                    if new_order:
                        order.status = OrderStatus.CANCELED
                        st.session_state.pending_orders[oid] = order
                else:
                    order.status = OrderStatus.FAILED
                    log_error(f"订单 {oid} 重试失败，已放弃")
                    del st.session_state.pending_orders[oid]
            elif status == 'open':
                order.status = OrderStatus.OPEN
            order.last_check = datetime.now()
        except Exception as e:
            log_error(f"检查订单状态失败 {oid}: {e}")

@dataclass
class Position:
    symbol: str
    direction: int
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    take_profit: float
    initial_atr: float
    prob: float = 0.0
    partial_taken: bool = False
    real: bool = False
    highest_price: float = 0.0
    lowest_price: float = 1e9
    atr_mult: float = CONFIG.atr_multiplier_base
    slippage_paid: float = 0.0
    price_history: deque = field(default_factory=lambda: deque(maxlen=CONFIG.atr_price_history_len))
    impact_cost: float = 0.0
    next_funding_time: Optional[datetime] = None  # 精确资金费结算时间

    def __post_init__(self):
        if self.direction == 1:
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price
        self.price_history.append(self.entry_price)
        # 初始化下一次资金费时间
        self.next_funding_time = get_next_funding_time(self.entry_time)

    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size * self.direction

    def stop_distance(self) -> float:
        if self.direction == 1:
            return self.entry_price - self.stop_loss
        else:
            return self.stop_loss - self.entry_price

    def update_stops(self, current_price: float, atr: float) -> None:
        self.price_history.append(current_price)
        if len(self.price_history) >= CONFIG.adapt_window:
            price_series = pd.Series(list(self.price_history))
            self.atr_mult = adaptive_atr_multiplier(price_series, timeframe_minutes=15)
        else:
            self.atr_mult = CONFIG.atr_multiplier_base
        if self.direction == 1:
            if current_price > self.highest_price:
                self.highest_price = current_price
            trailing_stop = self.highest_price - atr * self.atr_mult
            self.stop_loss = max(self.stop_loss, trailing_stop)
            new_tp = current_price + atr * self.atr_mult * CONFIG.tp_min_ratio
            self.take_profit = max(self.take_profit, new_tp)
            if current_price >= self.entry_price + self.stop_distance() * CONFIG.breakeven_trigger_pct:
                self.stop_loss = max(self.stop_loss, self.entry_price)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            trailing_stop = self.lowest_price + atr * self.atr_mult
            self.stop_loss = min(self.stop_loss, trailing_stop)
            new_tp = current_price - atr * self.atr_mult * CONFIG.tp_min_ratio
            self.take_profit = min(self.take_profit, new_tp)
            if current_price <= self.entry_price - self.stop_distance() * CONFIG.breakeven_trigger_pct:
                self.stop_loss = min(self.stop_loss, self.entry_price)

    def should_close(self, high: float, low: float, current_time: datetime) -> Tuple[bool, str, float, Optional[float]]:
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
        if hold_hours > CONFIG.max_hold_hours:
            return True, "超时", (high + low) / 2, self.size
        if not self.partial_taken:
            if self.direction == 1 and high >= self.entry_price + self.stop_distance() * CONFIG.partial_tp_r_multiple:
                self.partial_taken = True
                partial_size = self.size * CONFIG.partial_tp_ratio
                self.size *= (1 - CONFIG.partial_tp_ratio)
                self.stop_loss = max(self.stop_loss, self.entry_price)
                return True, "部分止盈", self.entry_price + self.stop_distance() * CONFIG.partial_tp_r_multiple, partial_size
            if self.direction == -1 and low <= self.entry_price - self.stop_distance() * CONFIG.partial_tp_r_multiple:
                self.partial_taken = True
                partial_size = self.size * CONFIG.partial_tp_ratio
                self.size *= (1 - CONFIG.partial_tp_ratio)
                self.stop_loss = min(self.stop_loss, self.entry_price)
                return True, "部分止盈", self.entry_price - self.stop_distance() * CONFIG.partial_tp_r_multiple, partial_size
        return False, "", 0.0, None

def set_leverage(symbol: str) -> None:
    if not st.session_state.exchange or not st.session_state.use_real:
        return
    try:
        leverage = int(CONFIG.max_leverage_global)
        exchange_name = st.session_state.exchange_choice.lower()
        if 'binance' in exchange_name:
            st.session_state.exchange.fapiPrivate_post_leverage({
                'symbol': symbol.replace('/', ''),
                'leverage': leverage
            })
        elif 'bybit' in exchange_name:
            st.session_state.exchange.private_linear_post_position_set_leverage({
                'symbol': symbol.replace('/', ''),
                'buy_leverage': leverage,
                'sell_leverage': leverage
            })
        elif 'okx' in exchange_name:
            st.session_state.exchange.privatePostAccountSetLeverage({
                'instId': symbol.replace('/', '-'),
                'lever': leverage,
                'mgnMode': 'cross'
            })
        log_execution(f"设置杠杆 {symbol} → {leverage}x")
    except Exception as e:
        log_error(f"设置杠杆失败 {symbol}: {e}")

def get_current_price(symbol: str) -> float:
    return st.session_state.symbol_current_prices.get(symbol, 0.0)

# ==================== 精确资金费扣除 ====================
def apply_funding_fees() -> None:
    now_utc = datetime.now(timezone.utc)
    for pos in st.session_state.positions.values():
        if pos.next_funding_time is None:
            pos.next_funding_time = get_next_funding_time(pos.entry_time)
        while now_utc >= pos.next_funding_time:
            rate = st.session_state.funding_rates.get(pos.symbol, 0.0)
            fee = pos.size * pos.entry_price * rate * pos.direction * -1  # 多头付正费，空头收正费
            st.session_state.account_balance += fee
            st.session_state.daily_pnl += fee
            log_execution(f"资金费 {pos.symbol}: {fee:.4f} (方向{'多' if pos.direction>0 else '空'})")
            append_to_db('funding_rates', {'time': now_utc.isoformat(), 'symbol': pos.symbol, 'rate': rate})
            pos.next_funding_time += timedelta(hours=8)

def split_and_execute(symbol: str, direction: int, total_size: float, price: float, stop: float, take: float, prob: float) -> None:
    imbalance = st.session_state.get('orderbook_imbalance', {}).get(symbol, 0.0)
    if abs(imbalance) > 0.3:
        splits = CONFIG.max_order_split * 2
    else:
        splits = CONFIG.max_order_split
    if total_size <= CONFIG.min_order_size * splits:
        execute_order(symbol, direction, total_size, price, stop, take, prob)
        return
    split_size = total_size / splits
    for i in range(splits):
        if i > 0:
            time.sleep(CONFIG.split_delay_seconds)
        current_price = get_current_price(symbol)
        stop_dist = stop - price if direction == 1 else price - stop
        new_stop = current_price - stop_dist if direction == 1 else current_price + stop_dist
        new_take = current_price + stop_dist * CONFIG.tp_min_ratio if direction == 1 else current_price - stop_dist * CONFIG.tp_min_ratio
        order = execute_order(symbol, direction, split_size, current_price, new_stop, new_take, prob)
        if order is None and st.session_state.use_real:
            log_error(f"第{i+1}次拆分订单失败，停止后续")
            break

def execute_order(symbol: str, direction: int, size: float, price: float, stop: float, take: float, prob: float) -> Optional[Order]:
    sym = symbol.strip()
    dir_str = "多" if direction == 1 else "空"
    side = 'buy' if direction == 1 else 'sell'

    volume_series = st.session_state.multi_df[sym]['15m']['volume'] if sym in st.session_state.multi_df and not st.session_state.multi_df[sym]['15m'].empty else pd.Series([1])
    volume_20 = volume_series.iloc[-1] if not volume_series.empty else 1.0
    vola_series = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:] if sym in st.session_state.multi_df else [0.02]
    vola = np.std(vola_series) if len(vola_series) > 0 else 0.02
    imbalance = st.session_state.get('orderbook_imbalance', {}).get(sym, 0.0)
    slippage = advanced_slippage_prediction(price, size, volume_20, vola, imbalance)
    exec_price = price + slippage if direction == 1 else price - slippage
    market_impact = (size / max(volume_20, 1)) ** 0.5 * vola * price * 0.3

    if st.session_state.use_real and st.session_state.exchange:
        order = create_order_and_record(sym, side, size, exec_price, stop, take, prob)
        if order is None:
            log_error(f"实盘开仓失败 {sym}")
            return None
        return order
    else:
        actual_price = exec_price
        actual_size = size
        st.session_state.positions[sym] = Position(
            symbol=sym,
            direction=direction,
            entry_price=actual_price,
            entry_time=datetime.now(),
            size=actual_size,
            stop_loss=stop,
            take_profit=take,
            initial_atr=0,
            real=False,
            slippage_paid=slippage,
            impact_cost=market_impact,
            prob=prob
        )
        st.session_state.daily_trades += 1
        log_execution(f"开仓 {sym} {dir_str} 仓位 {actual_size:.4f} @ {actual_price:.2f} 止损 {stop:.2f} 止盈 {take:.2f}")
        return True

def close_position(symbol: str, exit_price: float, reason: str, close_size: Optional[float] = None) -> None:
    sym = symbol.strip()
    pos = st.session_state.positions.get(sym)
    if not pos:
        return

    close_size = min(close_size or pos.size, pos.size)
    side = 'sell' if pos.direction == 1 else 'buy'

    volume_series = st.session_state.multi_df[sym]['15m']['volume'] if sym in st.session_state.multi_df else pd.Series([1])
    volume_20 = volume_series.iloc[-1] if not volume_series.empty else 1.0
    vola_series = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:] if sym in st.session_state.multi_df else [0.02]
    vola = np.std(vola_series) if len(vola_series) > 0 else 0.02
    imbalance = st.session_state.get('orderbook_imbalance', {}).get(sym, 0.0)
    slippage = advanced_slippage_prediction(exit_price, close_size, volume_20, vola, imbalance)
    exec_exit = exit_price - slippage if pos.direction == 1 else exit_price + slippage

    if pos.real and st.session_state.exchange:
        try:
            order = st.session_state.exchange.create_order(
                symbol=sym,
                type='market',
                side=side,
                amount=close_size,
                params={'reduceOnly': True}
            )
            actual_exit = float(order['average'] or order['price'] or exit_price)
            actual_size = float(order['amount'])
            log_execution(f"【实盘平仓成功】 {sym} {reason} {actual_size:.4f} @ {actual_exit:.2f}")
            send_telegram(f"【实盘】平仓 {reason} {sym}\n价格: {actual_exit:.2f}", msg_type="trade")
        except Exception as e:
            log_error(f"实盘平仓失败 {sym}: {e}")
            send_telegram(f"⚠️ 平仓失败 {sym} {reason}: {str(e)}", msg_type="risk")
            return
    else:
        actual_exit = exec_exit
        actual_size = close_size

    pnl = (actual_exit - pos.entry_price) * actual_size * pos.direction - actual_exit * actual_size * CONFIG.fee_rate * 2
    st.session_state.daily_pnl += pnl
    st.session_state.account_balance += pnl
    if st.session_state.account_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = st.session_state.account_balance
    st.session_state.net_value_history.append({'time': datetime.now(), 'value': st.session_state.account_balance})
    st.session_state.daily_returns.append(pnl / st.session_state.account_balance)

    trade_record = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': sym,
        'direction': '多' if pos.direction == 1 else '空',
        'entry': pos.entry_price,
        'exit': actual_exit,
        'size': actual_size,
        'pnl': pnl,
        'reason': reason,
        'slippage_entry': pos.slippage_paid,
        'slippage_exit': slippage,
        'impact_cost': pos.impact_cost,
        'raw_prob': pos.prob
    }
    st.session_state.trade_log.append(trade_record)
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log.pop(0)
    append_to_db('trades', trade_record)
    st.session_state.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage, 'impact': (close_size / max(volume_20,1))**0.5 * vola * exit_price * 0.3})
    append_to_db('slippage_log', {'time': datetime.now().isoformat(), 'symbol': sym, 'slippage': slippage, 'impact': (close_size / max(volume_20,1))**0.5 * vola * exit_price * 0.3})

    update_regime_stats(st.session_state.market_regime, pnl)
    update_consistency_stats(is_backtest=False, slippage=slippage, win=pnl>0)

    win_flag = pnl > 0
    RiskManager().update_losses(win_flag, loss_amount=pnl if not win_flag else 0)

    if actual_size >= pos.size:
        del st.session_state.positions[sym]
    else:
        pos.size -= actual_size
        log_execution(f"部分平仓 {sym} {reason} 数量 {actual_size:.4f}，剩余 {pos.size:.4f}")

    log_execution(f"平仓 {sym} {reason} 盈亏 {pnl:.2f} 余额 {st.session_state.account_balance:.2f}")
    send_telegram(f"平仓 {reason}\n盈亏: {pnl:.2f}", msg_type="trade")

def fix_data_consistency(symbols: List[str]) -> None:
    to_remove = []
    for sym in list(st.session_state.positions.keys()):
        if sym not in symbols or sym not in st.session_state.multi_df:
            to_remove.append(sym)
    for sym in to_remove:
        log_execution(f"数据修复：移除无效持仓 {sym}")
        del st.session_state.positions[sym]
    st.session_state.positions = {k: v for k, v in st.session_state.positions.items() if v.size > 0}

def generate_equity_chart() -> Optional[go.Figure]:
    if not st.session_state.equity_curve:
        return None
    df = pd.DataFrame(list(st.session_state.equity_curve)[-200:])
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

class BacktestRiskManager:
    def __init__(self, initial_balance):
        self.consecutive_losses = 0
        self.daily_risk_consumed = 0.0
        self.balance = initial_balance
        self.trade_log = []

    def update_losses(self, win: bool, loss_amount: float = 0.0):
        if not win:
            self.consecutive_losses += 1
            self.daily_risk_consumed += abs(loss_amount)
        else:
            self.consecutive_losses = 0

    def get_adaptive_risk_multiplier(self, current_dd: float) -> float:
        if self.consecutive_losses >= 4:
            return 0.0
        elif self.consecutive_losses == 3:
            mult = 0.4
        elif self.consecutive_losses == 2:
            mult = 0.6
        elif self.consecutive_losses == 1:
            mult = 0.8
        else:
            mult = 1.0
        if current_dd > 5.0:
            mult *= 0.3
        elif current_dd > 3.0:
            mult *= 0.5
        elif current_dd > 1.5:
            mult *= 0.7
        return max(0.0, mult)

def run_backtest_combined(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]], 
                           initial_balance: float = 10000, long_thresh: float = 0.53, short_thresh: float = 0.47,
                           atr_multiplier: float = CONFIG.atr_multiplier_base,
                           funding_rate: float = 0.0001) -> Dict[str, Any]:
    balance = initial_balance
    equity_curve = [balance]
    trade_log = []
    positions = {}
    current_date = None
    engine = SignalEngine()
    bt_risk = BacktestRiskManager(initial_balance)

    first_sym = symbols[0]
    df_ref = data_dicts[first_sym]['15m'].copy()
    df_ref['date'] = df_ref['timestamp'].dt.date
    total_steps = len(df_ref) - 200

    with st.progress(0) as progress_bar:
        for i in range(200, total_steps):
            progress_bar.progress((i-200) / (total_steps-200))
            current_time = df_ref['timestamp'].iloc[i]
            date = current_time.date()
            if current_date != date:
                bt_risk.daily_risk_consumed = 0.0
                current_date = date

            slice_dicts = {}
            for sym in symbols:
                df_sym = data_dicts[sym]['15m']
                mask = df_sym['timestamp'] <= current_time
                if not mask.any():
                    continue
                idx = mask.idxmax()
                slice_dict = {}
                for tf in data_dicts[sym]:
                    df_tf = data_dicts[sym][tf]
                    mask_tf = df_tf['timestamp'] <= current_time
                    if mask_tf.any():
                        slice_dict[tf] = df_tf.loc[mask_tf].copy()
                    else:
                        slice_dict[tf] = pd.DataFrame()
                slice_dicts[sym] = slice_dict

            symbol_signals = {}
            for sym in symbols:
                if sym not in slice_dicts or slice_dicts[sym].get('15m', pd.DataFrame()).empty:
                    continue
                direction, prob = engine.calc_signal(slice_dicts[sym], sym)
                if direction != 0 and prob >= SignalStrength.WEAK.value:
                    row = data_dicts[sym]['15m'].loc[data_dicts[sym]['15m']['timestamp'] <= current_time].iloc[-1]
                    atr = row['atr'] if not pd.isna(row['atr']) else 0
                    price = row['close']
                    price_hist = data_dicts[sym]['15m']['close'][data_dicts[sym]['15m']['timestamp'] <= current_time].iloc[-CONFIG.adapt_window:]
                    symbol_signals[sym] = (direction, prob, atr, price, price_hist)

            for sym in symbols:
                if sym not in positions and sym in symbol_signals:
                    direction, prob, atr, price, price_hist = symbol_signals[sym]
                    if direction == 1 and prob < long_thresh:
                        continue
                    if direction == -1 and prob > short_thresh:
                        continue
                    atr_series = data_dicts[sym]['15m']['atr'][data_dicts[sym]['15m']['timestamp'] <= current_time]
                    if len(atr_series) >= 20:
                        atr_ma = atr_series.rolling(20).mean().iloc[-1]
                        if not pd.isna(atr_ma) and atr > atr_ma * 1.5:
                            continue
                    ema200_series = data_dicts[sym]['15m']['ema200'][data_dicts[sym]['15m']['timestamp'] <= current_time]
                    if not ema200_series.empty:
                        ema200 = ema200_series.iloc[-1]
                        if direction == 1 and price < ema200:
                            continue
                        if direction == -1 and price > ema200:
                            continue
                    risk_per_trade_amount = balance * CONFIG.risk_per_trade
                    if bt_risk.daily_risk_consumed + risk_per_trade_amount > balance * CONFIG.daily_risk_budget_ratio:
                        continue

                    if price <= 0 or prob < 0.5:
                        size = 0
                    else:
                        risk_amount = risk_per_trade_amount
                        equity_series = pd.Series(equity_curve)
                        peak = equity_series.cummax()
                        dd = (peak - equity_series) / peak * 100
                        current_dd = dd.iloc[-1] if len(dd) > 0 else 0
                        adaptive_mult = bt_risk.get_adaptive_risk_multiplier(current_dd)
                        risk_amount *= adaptive_mult
                        if prob > 0.7:
                            risk_amount *= 1.5
                        if atr == 0 or np.isnan(atr) or atr < price * CONFIG.min_atr_pct / 100:
                            stop_distance = price * 0.01
                        else:
                            stop_distance = atr * atr_multiplier
                        size = risk_amount / stop_distance
                    stop_dist = atr * atr_multiplier if atr > 0 else price * 0.01
                    stop = price - stop_dist if direction == 1 else price + stop_dist
                    take = price + stop_dist * CONFIG.tp_min_ratio if direction == 1 else price - stop_dist * CONFIG.tp_min_ratio
                    volume_series = data_dicts[sym]['15m']['volume'][data_dicts[sym]['15m']['timestamp'] <= current_time]
                    volume = volume_series.iloc[-1] if not volume_series.empty else 1.0
                    vola = atr / price if atr > 0 else 0.02
                    slippage = dynamic_slippage(price, size, volume, vola, 0)
                    exec_price = price + slippage if direction == 1 else price - slippage
                    positions[sym] = Position(sym, direction, exec_price, current_time, size, stop, take, atr, prob)
                    bt_risk.daily_risk_consumed += risk_per_trade_amount

            for sym, pos in list(positions.items()):
                row = data_dicts[sym]['15m'].loc[data_dicts[sym]['15m']['timestamp'] <= current_time].iloc[-1]
                high = row['high']
                low = row['low']
                atr = row['atr'] if not pd.isna(row['atr']) else pos.initial_atr
                pos.update_stops(row['close'], atr)
                should_close, reason, exit_price, close_size = pos.should_close(high, low, current_time)
                if should_close:
                    volume_series = data_dicts[sym]['15m']['volume'][data_dicts[sym]['15m']['timestamp'] <= current_time]
                    volume = volume_series.iloc[-1] if not volume_series.empty else 1.0
                    slippage = dynamic_slippage(exit_price, close_size, volume, vola, 0)
                    exec_exit = exit_price - slippage if pos.direction == 1 else exit_price + slippage
                    pnl = (exec_exit - pos.entry_price) * close_size * pos.direction - exec_exit * close_size * CONFIG.fee_rate * 2
                    balance += pnl
                    trade_log.append({"sym": sym, "pnl": pnl, "reason": reason})
                    bt_risk.trade_log.append(pnl)
                    win_flag = pnl > 0
                    bt_risk.update_losses(win_flag, loss_amount=pnl if not win_flag else 0)
                    if close_size >= pos.size:
                        del positions[sym]
                    else:
                        pos.size -= close_size

            if current_time.hour % 8 == 0 and current_time.minute == 0:
                for sym, pos in positions.items():
                    fund_fee = pos.size * pos.entry_price * funding_rate
                    balance -= fund_fee
                    trade_log.append({"sym": sym, "pnl": -fund_fee, "reason": "funding"})

            equity_curve.append(balance)

    trades_df = pd.DataFrame(trade_log)
    win_rate = (trades_df['pnl'] > 0).mean() if not trades_df.empty else 0
    total_return = (balance - initial_balance) / initial_balance * 100
    equity_series = pd.Series(equity_curve)
    max_dd = (equity_series.cummax() - equity_series).max() / equity_series.cummax().max() * 100 if len(equity_series) > 0 else 0
    profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if any(trades_df['pnl'] < 0) else 0
    sharpe = (pd.Series(equity_curve).pct_change().dropna().mean() / pd.Series(equity_curve).pct_change().dropna().std() * np.sqrt(252)) if len(equity_curve) > 20 else 0

    return {
        "final_balance": balance,
        "equity_curve": equity_curve,
        "metrics": {
            "win_rate": win_rate,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "trade_log": trade_log
        }
    }

def bayesian_optimize_params(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, float]:
    def objective(params):
        long_thresh, short_thresh, atr_mult = params
        res = run_backtest_combined(symbols, data_dicts, 
                                    long_thresh=long_thresh,
                                    short_thresh=short_thresh,
                                    atr_multiplier=atr_mult)
        return -res['metrics']['sharpe']

    space = [Real(0.5, 0.6, name='long_thresh'),
             Real(0.4, 0.5, name='short_thresh'),
             Real(1.0, 3.0, name='atr_mult')]

    with st.spinner("贝叶斯优化进行中，请稍候..."):
        result = gp_minimize(objective, space, n_calls=30, random_state=42)
    best_params = {
        'long_thresh': result.x[0],
        'short_thresh': result.x[1],
        'atr_mult': result.x[2],
        'best_sharpe': -result.fun
    }
    return best_params

class UIRenderer:
    def __init__(self):
        self.fetcher = get_fetcher()

    def can_open_by_rules(self, symbol: str, direction: int, prob: float, atr: float, price: float, 
                          df_dict: Dict[str, pd.DataFrame], risk_budget_remaining: float, risk_per_trade_amount: float) -> Tuple[bool, str]:
        if direction == 1 and prob < 0.53:
            return False, f"做多概率{prob:.1%}<53%"
        if direction == -1 and prob > 0.47:
            return False, f"做空概率{prob:.1%}>47% (应≤47%)"
        atr_series = df_dict['15m']['atr']
        if len(atr_series) >= 20:
            atr_ma = atr_series.rolling(20).mean().iloc[-1]
            if not pd.isna(atr_ma) and atr > atr_ma * 1.5:
                return False, f"ATR过高 ({atr:.2f} > {atr_ma*1.5:.2f})"
        ema200 = df_dict['15m']['ema200'].iloc[-1]
        if direction == 1 and price < ema200:
            return False, "价格在EMA200下方，禁止做多"
        if direction == -1 and price > ema200:
            return False, "价格在EMA200上方，禁止做空"
        if risk_budget_remaining < risk_per_trade_amount:
            return False, "风险预算不足"
        if funding_rate_blocked(symbol, direction):
            return False, "资金费率不利"
        return True, "通过"

    def render_sidebar(self) -> Tuple[List[str], None, bool]:
        with st.sidebar:
            st.header("⚙️ 配置")
            mode = st.radio("模式", ['实盘', '回测'], index=0, key="mode_radio")
            st.session_state.mode = 'live' if mode == '实盘' else 'backtest'

            selected_symbols = st.multiselect("交易品种", CONFIG.symbols, default=['ETH/USDT', 'BTC/USDT'], key="symbol_multiselect")
            st.session_state.current_symbols = selected_symbols

            use_sim = st.checkbox("使用模拟数据（离线模式）", value=st.session_state.use_simulated_data, key="use_sim_checkbox")
            if use_sim != st.session_state.use_simulated_data:
                st.session_state.use_simulated_data = use_sim
                for key in list(st.session_state.keys()):
                    if key.startswith('sim_data_'):
                        del st.session_state[key]
                st.cache_data.clear()
                st.rerun()

            if st.session_state.use_simulated_data:
                st.info("📡 当前数据源：模拟数据")
            else:
                if st.session_state.data_source_failed:
                    st.error("📡 真实数据获取失败，已回退到模拟")
                else:
                    st.success("📡 当前数据源：币安实时数据")

            st.write(f"单笔风险: {CONFIG.risk_per_trade*100:.1f}% (动态凯利将覆盖)")
            st.write(f"每日风险预算: {CONFIG.daily_risk_budget_ratio*100:.1f}%")

            st.number_input("余额 USDT", value=st.session_state.account_balance, disabled=True, key="balance_display")

            if st.button("🔄 同步实盘余额", key="sync_balance_button"):
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

            # 从环境变量读取密钥，但允许用户覆盖
            api_key = st.text_input("API Key", value=st.session_state.binance_api_key or BINANCE_API_KEY, type="password", key="api_key_input")
            secret_key = st.text_input("Secret Key", value=st.session_state.binance_secret_key or BINANCE_SECRET_KEY, type="password", key="secret_key_input")
            passphrase = st.text_input("Passphrase (仅OKX需要)", type="password", key="passphrase_input") if "OKX" in exchange_choice else None
            testnet = st.checkbox("测试网", value=st.session_state.testnet, key="testnet_checkbox")
            use_real = st.checkbox("实盘交易", value=st.session_state.use_real, key="use_real_checkbox")

            if st.button("🔌 测试连接", key="test_connection_button"):
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
                    ticker = ex.fetch_ticker(selected_symbols[0])
                    st.success(f"连接成功！{selected_symbols[0]} 价格: {ticker['last']}")
                    st.session_state.exchange = ex
                    st.session_state.binance_api_key = api_key
                    st.session_state.binance_secret_key = secret_key
                    st.session_state.testnet = testnet
                    st.session_state.use_real = use_real
                    if WS_AVAILABLE and st.session_state.ws_fetcher:
                        st.session_state.ws_fetcher.stop()
                        start_ws_fetcher()
                except Exception as e:
                    st.error(f"连接失败: {e}")

            st.session_state.auto_enabled = st.checkbox("自动交易", value=True, key="auto_enabled_checkbox")
            st.session_state.aggressive_mode = st.checkbox("进攻模式 (允许更高风险)", value=False, key="aggressive_mode_checkbox")

            with st.expander("📱 通知与工具"):
                token = st.text_input("Bot Token", type="password", key="telegram_token_input")
                chat_id = st.text_input("Chat ID", key="telegram_chat_input")
                if token and chat_id:
                    st.session_state.telegram_token = token
                    st.session_state.telegram_chat_id = chat_id

                if st.button("📂 查看历史交易记录", key="view_history_button"):
                    df_trades = load_from_db('trades', limit=20)
                    st.dataframe(df_trades)

                if st.button("🔧 数据修复", key="fix_data_button"):
                    fix_data_consistency(st.session_state.current_symbols)
                    st.success("数据一致性已修复")

                if st.button("📤 发送权益曲线", key="send_equity_button"):
                    fig = generate_equity_chart()
                    if fig:
                        send_telegram("当前权益曲线", image=fig)
                        st.success("权益曲线已发送")
                    else:
                        st.warning("无权益数据")

                if st.button("🗑️ 重置所有状态", key="reset_state_button"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

            if st.button("🚨 一键紧急平仓", key="emergency_close_button"):
                for sym in list(st.session_state.positions.keys()):
                    if sym in st.session_state.symbol_current_prices:
                        close_position(sym, st.session_state.symbol_current_prices[sym], "紧急平仓")
                st.rerun()

            if st.session_state.error_log:
                with st.expander("⚠️ 错误日志（实时）"):
                    for err in list(st.session_state.error_log)[-10:]:
                        st.text(err)

            if st.session_state.execution_log:
                with st.expander("📋 执行日志（实时）"):
                    for log in list(st.session_state.execution_log)[-10:]:
                        st.text(log)

        return selected_symbols, None, use_real

    def render_main_panel(self, symbols: List[str], mode: str, use_real: bool) -> None:
        if not symbols:
            st.warning("请至少选择一个交易品种")
            return

        check_and_reset_daily()
        check_pending_orders()
        schedule_model_training()
        # 定时更新因子权重（每小时）
        if int(time.time()) % 3600 < 60:  # 每小时的前60秒内执行
            update_factor_weights_scheduled()
        apply_funding_fees()

        multi_data = {}
        for sym in symbols:
            data = self.fetcher.get_symbol_data(sym)
            if data is None:
                st.error(f"获取 {sym} 数据失败")
                return
            multi_data[sym] = data
            st.session_state.symbol_current_prices[sym] = data['current_price']
            if 'orderbook_imbalance' not in st.session_state:
                st.session_state.orderbook_imbalance = {}
            st.session_state.orderbook_imbalance[sym] = data.get('orderbook_imbalance', 0.0)

        st.session_state.multi_df = {sym: data['data_dict'] for sym, data in multi_data.items()}
        first_sym = symbols[0]
        st.session_state.fear_greed = multi_data[first_sym]['fear_greed']

        df_first = multi_data[first_sym]['data_dict']
        st.session_state.market_regime = SignalEngine().detect_market_regime(df_first, first_sym)

        cov = calculate_cov_matrix(symbols, {sym: multi_data[sym]['data_dict'] for sym in symbols}, CONFIG.cov_matrix_window)
        st.session_state.cov_matrix = cov

        fix_data_consistency(symbols)

        if st.session_state.mode == 'backtest':
            self.render_backtest_panel(symbols, multi_data)
        else:
            self.render_live_panel(symbols, multi_data)

    def render_backtest_panel(self, symbols: List[str], multi_data: Dict[str, Any]) -> None:
        st.subheader("📊 回测 (新版组合回测)")
        col1, col2 = st.columns(2)
        with col1:
            long_thresh = st.number_input("做多阈值", value=0.53, min_value=0.5, max_value=0.6, step=0.01)
            short_thresh = st.number_input("做空阈值", value=0.47, min_value=0.4, max_value=0.5, step=0.01)
            atr_mult = st.number_input("ATR倍数", value=CONFIG.atr_multiplier_base, min_value=1.0, max_value=3.0, step=0.1)
            funding = st.number_input("资金费率(模拟)", value=0.0001, format="%.6f")
            if st.button("🚀 运行回测", key="run_backtest_btn"):
                with st.spinner("回测中..."):
                    res = run_backtest_combined(symbols, {sym: multi_data[sym]['data_dict'] for sym in symbols},
                                                long_thresh=long_thresh, short_thresh=short_thresh,
                                                atr_multiplier=atr_mult, funding_rate=funding)
                    st.session_state.backtest_results_new = res
                    st.success("回测完成！")
        with col2:
            if st.button("🔍 贝叶斯优化参数", key="bayes_opt_btn"):
                best = bayesian_optimize_params(symbols, {sym: multi_data[sym]['data_dict'] for sym in symbols})
                st.session_state.best_params = best
                st.write("最优参数：")
                st.json(best)

        new = st.session_state.get('backtest_results_new')
        if new:
            st.metric("最终权益", f"{new['final_balance']:.2f}")
            st.write("交易次数:", len(new['metrics']['trade_log']))
            st.write("胜率:", f"{new['metrics'].get('win_rate', 0)*100:.2f}%")
            st.write("最大回撤:", f"{new['metrics'].get('max_drawdown', 0):.2f}%")
            st.write("盈亏比:", f"{new['metrics'].get('profit_factor', 0):.2f}")
            st.write("夏普比率:", f"{new['metrics'].get('sharpe', 0):.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=new['equity_curve'], mode='lines', name='权益曲线'))
            st.plotly_chart(fig, key="backtest_chart")

    def render_live_panel(self, symbols: List[str], multi_data: Dict[str, Any]) -> None:
        st.subheader("多品种持仓")
        risk = RiskManager()
        engine = SignalEngine()
        glob = st.session_state.globals

        cooldown = risk.check_cooldown()
        risk_budget_ok = check_risk_budget()
        if cooldown:
            st.warning(f"系统冷却中，直至 {st.session_state.cooldown_until.strftime('%H:%M')}")
        if not risk_budget_ok:
            st.error(f"每日风险预算已达上限 ({CONFIG.daily_risk_budget_ratio*100:.1f}%)，今日停止开新仓")

        with st.expander("📊 因子权重与IC", expanded=False):
            if st.session_state.factor_ic_stats:
                df_ic = pd.DataFrame(st.session_state.factor_ic_stats).T.round(4)
                df_ic['权重'] = pd.Series(glob['factor_weights'])
                def highlight_p(val):
                    if isinstance(val, float) and val < 0.05:
                        return 'background-color: lightgreen'
                    return ''
                st.dataframe(df_ic.style.applymap(highlight_p, subset=['p_value']))
            else:
                st.info("暂无IC统计数据，积累更多交易后可见")

        symbol_signals = {}
        for sym in symbols:
            df_dict_sym = st.session_state.multi_df[sym]
            direction, prob = engine.calc_signal(df_dict_sym, sym)
            if direction != 0 and prob >= SignalStrength.WEAK.value:
                price = multi_data[sym]['current_price']
                atr_sym = df_dict_sym['15m']['atr'].iloc[-1] if not pd.isna(df_dict_sym['15m']['atr'].iloc[-1]) else 0
                price_hist = df_dict_sym['15m']['close'].iloc[-CONFIG.adapt_window:]
                symbol_signals[sym] = (direction, prob, atr_sym, price, price_hist)

        allocations = risk.allocate_portfolio(symbol_signals, st.session_state.account_balance, st.session_state.cov_matrix)

        can_open_global = not (cooldown or not risk_budget_ok)
        for sym in symbols:
            train_calibration_model(sym)

            if sym not in st.session_state.positions and allocations.get(sym, 0) > 0:
                dir, prob, atr_sym, price, price_hist = symbol_signals[sym]
                risk_per_trade_amount = st.session_state.account_balance * get_kelly_fraction()
                risk_budget_remaining = st.session_state.account_balance * CONFIG.daily_risk_budget_ratio - st.session_state.daily_risk_consumed
                df_dict_sym = st.session_state.multi_df[sym]
                
                can_open, reason = self.can_open_by_rules(
                    sym, dir, prob, atr_sym, price, df_dict_sym,
                    risk_budget_remaining, risk_per_trade_amount
                )
                
                if can_open and can_open_global:
                    if atr_sym == 0 or np.isnan(atr_sym):
                        stop_dist = price * 0.01
                    else:
                        stop_dist = atr_sym * adaptive_atr_multiplier(price_hist, timeframe_minutes=15)
                    stop = price - stop_dist if dir == 1 else price + stop_dist
                    take = price + stop_dist * CONFIG.tp_min_ratio if dir == 1 else price - stop_dist * CONFIG.tp_min_ratio
                    size = allocations[sym]
                    split_and_execute(sym, dir, size, price, stop, take, prob)
                else:
                    log_execution(f"开仓被阻止：{sym}，原因：{reason}")

        for sym, pos in list(st.session_state.positions.items()):
            if sym not in symbols:
                continue
            df_dict_sym = st.session_state.multi_df[sym]
            current_price = multi_data[sym]['current_price']
            high = df_dict_sym['15m']['high'].iloc[-1]
            low = df_dict_sym['15m']['low'].iloc[-1]
            atr_sym = df_dict_sym['15m']['atr'].iloc[-1] if not pd.isna(df_dict_sym['15m']['atr'].iloc[-1]) else 0
            should_close, reason, exit_price, close_size = pos.should_close(high, low, datetime.now())
            if should_close:
                close_position(sym, exit_price, reason, close_size)
            else:
                if not pd.isna(atr_sym) and atr_sym > 0:
                    pos.update_stops(current_price, atr_sym)

        total_floating = 0.0
        for sym, pos in st.session_state.positions.items():
            if sym in multi_data:
                total_floating += pos.pnl(multi_data[sym]['current_price'])

        historical_rets = None
        if len(symbols) > 1:
            ret_arrays = []
            for sym in symbols:
                rets = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-100:]
                ret_arrays.append(rets)
            min_len = min(len(arr) for arr in ret_arrays)
            hist_rets = np.array([arr[-min_len:] for arr in ret_arrays])
            historical_rets = hist_rets

        portfolio_var_value = 0.0
        portfolio_cvar_value = 0.0
        if st.session_state.cov_matrix is not None and len(symbols) > 1:
            total_value = st.session_state.account_balance
            weights = []
            for sym in symbols:
                if sym in st.session_state.positions:
                    pos = st.session_state.positions[sym]
                    value = pos.size * multi_data[sym]['current_price']
                    weight = value / total_value
                else:
                    weight = 0.0
                weights.append(weight)
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                method = st.session_state.get('var_method', CONFIG.var_method.value)
                portfolio_var_value = portfolio_var(weights, st.session_state.cov_matrix, CONFIG.var_confidence, method, historical_rets)
                portfolio_cvar_value = portfolio_cvar(weights, historical_rets, CONFIG.var_confidence)

        record_equity_point()
        current_dd, max_dd = calculate_drawdown()

        risk_budget_total = st.session_state.account_balance * CONFIG.daily_risk_budget_ratio
        risk_budget_remaining = max(0, risk_budget_total - st.session_state.daily_risk_consumed)

        st.session_state.advanced_metrics = calculate_advanced_metrics()

        col1, col2 = st.columns([1.2, 1.8])
        with col1:
            st.markdown("### 📊 市场状态")
            first_sym = symbols[0]
            prob_first = engine.calc_signal(st.session_state.multi_df[first_sym], first_sym)[1]
            fear_greed_display = multi_data[first_sym]['fear_greed']
            c1, c2, c3 = st.columns(3)
            c1.metric("恐惧贪婪", fear_greed_display)
            c2.metric("信号概率", f"{prob_first:.1%}")
            c3.metric("当前价格", f"{multi_data[first_sym]['current_price']:.2f}")

            price_lines = " | ".join([f"{sym}: {multi_data[sym]['current_price']:.2f}" for sym in symbols])
            st.caption(price_lines)

            cal_status = []
            for sym in symbols:
                if sym in glob['ml_calibrators']:
                    cnt = glob['ml_calibrators_count'].get(sym, 0)
                    cal_status.append(f"{sym}: ✅ 已校准({cnt}笔)")
                else:
                    cal_status.append(f"{sym}: ⏳ 待校准")
            st.caption("校准状态: " + " | ".join(cal_status))

            if st.session_state.positions:
                st.markdown("### 📈 当前持仓")
                pos_list = []
                for sym, pos in st.session_state.positions.items():
                    current = multi_data[sym]['current_price']
                    pnl = pos.pnl(current)
                    pnl_pct = (current - pos.entry_price) / pos.entry_price * 100 * pos.direction
                    hold_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
                    pos_list.append({
                        "品种": sym,
                        "方向": "多" if pos.direction==1 else "空",
                        "入场价": f"{pos.entry_price:.2f}",
                        "数量": f"{pos.size:.4f}",
                        "浮动盈亏": f"{pnl:.2f} ({pnl_pct:+.2f}%)",
                        "持仓时长": f"{hold_hours:.1f}h",
                        "止损": f"{pos.stop_loss:.2f}",
                        "止盈": f"{pos.take_profit:.2f}"
                    })
                df_pos = pd.DataFrame(pos_list)
                st.dataframe(df_pos, height=200, use_container_width=True)
            else:
                st.markdown("### 无持仓")
                st.info("等待信号...")

            st.markdown("### 📉 风险监控")
            row1 = st.columns(4)
            row1[0].metric("实时盈亏", f"{st.session_state.daily_pnl + total_floating:.2f} USDT")
            row1[1].metric("当前回撤", f"{current_dd:.2f}%")
            row1[2].metric("最大回撤", f"{max_dd:.2f}%")
            row1[3].metric("连亏次数", st.session_state.consecutive_losses)

            row2 = st.columns(4)
            row2[0].metric("今日风险消耗", f"{st.session_state.daily_risk_consumed:.2f} USDT")
            row2[1].metric("剩余预算", f"{risk_budget_remaining:.2f} USDT")
            row2[2].metric("组合VaR", f"{portfolio_var_value*100:.2f}%")
            row2[3].metric("组合CVaR", f"{portfolio_cvar_value*100:.2f}%")

            used_ratio = st.session_state.daily_risk_consumed / (st.session_state.account_balance * CONFIG.daily_risk_budget_ratio)
            st.progress(min(used_ratio, 1.0), text=f"今日风险预算已用 {used_ratio*100:.1f}%")

            if st.session_state.cooldown_until:
                st.warning(f"冷却至 {st.session_state.cooldown_until.strftime('%H:%M')}")
            if is_night_time():
                st.info("🌙 当前为美东夜间时段，风险预算已降低")

            with st.expander("📈 高级绩效指标"):
                am = st.session_state.advanced_metrics
                if am:
                    st.write(f"夏普比率: {am['sharpe']:.2f}")
                    st.write(f"索提诺比率: {am['sortino']:.2f}")
                    st.write(f"卡玛比率: {am['calmar']:.2f}")
                    st.write(f"盈亏比: {am['profit_factor']:.2f}")
                    st.write(f"胜率: {am['win_rate']:.2%}")
                    st.write(f"总收益率: {am['total_return']:.2f}%")
                    st.write(f"最大回撤: {am['max_drawdown']:.2f}%")
                else:
                    st.info("暂无足够数据计算高级绩效指标")

            with st.expander("📈 市场状态统计"):
                if st.session_state.regime_stats:
                    df_reg = pd.DataFrame(st.session_state.regime_stats).T
                    df_reg['胜率'] = df_reg['wins'] / df_reg['trades'] * 100
                    df_reg['平均盈亏'] = df_reg['total_pnl'] / df_reg['trades']
                    st.dataframe(df_reg[['trades', '胜率', '平均盈亏']].round(2))
                else:
                    st.info("暂无统计数据")

            with st.expander("🔄 实盘一致性"):
                if st.session_state.consistency_stats:
                    cons = st.session_state.consistency_stats
                    bt = cons.get('backtest', {})
                    lv = cons.get('live', {})
                    if bt and lv:
                        st.write(f"回测滑点: {bt.get('avg_slippage', 0):.4f} 实盘滑点: {lv.get('avg_slippage', 0):.4f}")
                        st.write(f"回测胜率: {bt.get('win_rate', 0):.2%} 实盘胜率: {lv.get('win_rate', 0):.2%}")
                        if lv.get('avg_slippage', 0) > bt.get('avg_slippage', 0) * 2:
                            st.warning("⚠️ 实盘滑点显著高于回测，请检查流动性或调整滑点模型")
                    else:
                        st.write("暂无足够实盘数据对比")
                else:
                    st.info("暂无一致性数据")

            with st.expander("📊 因子IC统计"):
                if st.session_state.factor_ic_stats:
                    df_ic = pd.DataFrame(st.session_state.factor_ic_stats).T.round(4)
                    def highlight_p(val):
                        if val < 0.05:
                            return 'background-color: lightgreen'
                        return ''
                    st.dataframe(df_ic.style.applymap(highlight_p, subset=['p_value']))
                else:
                    st.info("暂无IC统计数据")

            if st.session_state.net_value_history and st.session_state.equity_curve:
                hist_df = pd.DataFrame(st.session_state.net_value_history[-200:])
                equity_df = pd.DataFrame(list(st.session_state.equity_curve)[-200:])
                fig_nv = go.Figure()
                fig_nv.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['value'], mode='lines', name='已平仓净值', line=dict(color='cyan')))
                fig_nv.add_trace(go.Scatter(x=equity_df['time'], y=equity_df['equity'], mode='lines', name='当前权益', line=dict(color='yellow')))
                fig_nv.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), template='plotly_dark')
                st.plotly_chart(fig_nv, use_container_width=True, key="equity_chart_fixed")

        with col2:
            df_plot = st.session_state.multi_df[first_sym]['15m'].tail(120).copy()
            if not df_plot.empty:
                if not pd.api.types.is_datetime64_any_dtype(df_plot['timestamp']):
                    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')
                df_plot = df_plot.dropna(subset=['timestamp'])
                if df_plot.empty:
                    st.warning("图表数据无效")
                    return
            else:
                st.warning("无图表数据")
                return

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.2, 0.2, 0.2], vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(x=df_plot['timestamp'], open=df_plot['open'], high=df_plot['high'],
                                          low=df_plot['low'], close=df_plot['close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema20'], line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema50'], line=dict(color="blue")), row=1, col=1)
            if first_sym in st.session_state.positions:
                pos = st.session_state.positions[first_sym]
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
            fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, key="kline_fixed")

def main() -> None:
    st.set_page_config(page_title="终极量化终端 · 职业版 48.3 (终极融合版)", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 职业版 48.3 (终极融合版)")
    st.caption("融合 48.1 与 48.2 所有优势 · 资金费精确 · 协方差稳定 · WebSocket降级 · 数据库连接池 · 环境变量安全 · 定时因子权重 · 回测进度条")

    init_session_state()
    check_and_fix_anomalies()
    renderer = UIRenderer()
    symbols, mode, use_real = renderer.render_sidebar()

    if symbols:
        renderer.render_main_panel(symbols, mode, use_real)
    else:
        st.warning("请在左侧选择至少一个交易品种")

    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

if __name__ == "__main__":
    main()
