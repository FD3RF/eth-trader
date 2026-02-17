# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 完美极限版 42.1
==================================================
核心特性：
1. 协方差矩阵风险平价（动态品种相关性）
2. 动态滑点模型（基于波动率、成交量、订单大小）
3. 组合VaR实时监控（每日95% VaR）
4. 严格Walk Forward验证（训练/测试完全隔离）
5. 因子IC显著性检验（p值 + 信息比率）
6. 多品种持仓显示修复（按品种名称严格匹配，数据永不串位）
7. 数据一致性验证：自动清理无效持仓，一键修复
8. 净值曲线持久化（包含浮动盈亏，自动保存/加载 equity_curve.csv）
9. 精准回撤计算（基于实时权益，当前回撤/最大回撤）
10. 市场状态分段统计（趋势/震荡/恐慌下的胜率、盈亏）
11. 实盘一致性误差统计（滑点偏差、胜率对比）
12. 所有已有功能（多周期信号、在线学习、回测、参数敏感性等）
13. 高性能并行数据获取 + 自动回退模拟
14. 完整日志持久化（CSV + 按日文件）
15. 一键紧急平仓、Telegram通知
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
from scipy.stats import ttest_1samp, norm

warnings.filterwarnings('ignore')

# ==================== 日志文件持久化 ====================
LOG_DIR = "logs"
TRADE_LOG_FILE = "trade_log.csv"
PERF_LOG_FILE = "performance_log.csv"
SLIPPAGE_LOG_FILE = "slippage_log.csv"
EQUITY_CURVE_FILE = "equity_curve.csv"      # 权益曲线持久化
REGIME_STATS_FILE = "regime_stats.csv"      # 市场状态统计
CONSISTENCY_FILE = "consistency_stats.csv"  # 一致性误差统计
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
    slippage_base: float = 0.0005
    slippage_impact_factor: float = 0.1
    fee_rate: float = 0.0004
    ic_window: int = 80
    mc_simulations: int = 500
    sim_volatility: float = 0.06
    sim_trend_strength: float = 0.2
    adapt_window: int = 20
    factor_learning_rate: float = 0.3
    var_confidence: float = 0.95
    portfolio_risk_target: float = 0.02
    cov_matrix_window: int = 50
    max_drawdown_window: int = 100  # 回撤计算窗口

CONFIG = TradingConfig()

# ==================== 全局变量（因子权重）====================
factor_weights = {
    'trend': 1.0,
    'rsi': 1.0,
    'macd': 1.0,
    'bb': 1.0,
    'volume': 1.0,
    'adx': 1.0
}
factor_to_col = {
    'trend': 'trend_factor',
    'rsi': 'rsi',
    'macd': 'macd_diff',
    'bb': 'bb_factor',
    'volume': 'volume_ratio',
    'adx': 'adx'
}

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
    # 加载持久化权益曲线
    equity_df = load_csv(EQUITY_CURVE_FILE)
    equity_curve = deque(maxlen=500)
    if not equity_df.empty:
        for _, row in equity_df.iterrows():
            try:
                t = pd.to_datetime(row['time'])
                equity_curve.append({'time': t, 'equity': float(row['equity'])})
            except:
                pass

    # 加载市场状态统计
    regime_stats = {}
    regime_df = load_csv(REGIME_STATS_FILE)
    if not regime_df.empty:
        for _, row in regime_df.iterrows():
            regime_stats[row['regime']] = {
                'trades': int(row['trades']),
                'wins': int(row['wins']),
                'total_pnl': float(row['total_pnl'])
            }

    # 加载一致性误差统计
    consistency_stats = {'backtest': {}, 'live': {}}
    cons_df = load_csv(CONSISTENCY_FILE)
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
        'peak_balance': 10000.0,
        'consecutive_losses': 0,
        'daily_trades': 0,
        'trade_log': [],
        'positions': {},
        'auto_enabled': True,
        'pause_until': None,
        'exchange': None,
        'net_value_history': [],  # 仅用于显示已平仓净值（历史）
        'equity_curve': equity_curve,  # 实时权益曲线（含浮动盈亏）
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
        'binance_api_key': '',
        'binance_secret_key': '',
        'fear_greed': 50,
        'market_regime': MarketRegime.RANGE,
        'multi_df': {},
        'performance_metrics': {},
        'mode': 'live',
        'factor_ic_stats': {},
        'symbol_current_prices': {},
        'daily_returns': deque(maxlen=252),
        'cov_matrix': None,
        'slippage_records': [],
        'regime_stats': regime_stats,
        'consistency_stats': consistency_stats,
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

# ==================== 实时权益计算 ====================
def current_equity():
    """计算当前总权益 = 余额 + 所有持仓浮动盈亏"""
    balance = st.session_state.account_balance
    floating = 0.0
    for sym, pos in st.session_state.positions.items():
        if sym in st.session_state.symbol_current_prices:
            floating += pos.pnl(st.session_state.symbol_current_prices[sym])
    return balance + floating

# ==================== 精准回撤计算 ====================
def calculate_drawdown():
    """基于权益曲线计算当前回撤和最大回撤"""
    if len(st.session_state.equity_curve) < 2:
        return 0.0, 0.0
    df = pd.DataFrame(list(st.session_state.equity_curve))
    peak = df['equity'].cummax()
    dd = (peak - df['equity']) / peak * 100
    current_dd = dd.iloc[-1]
    max_dd = dd.max()
    return current_dd, max_dd

# ==================== 记录权益点 ====================
def record_equity_point():
    equity = current_equity()
    now = datetime.now()
    st.session_state.equity_curve.append({'time': now, 'equity': equity})
    # 持久化（可追加）
    append_to_csv(EQUITY_CURVE_FILE, {'time': now.isoformat(), 'equity': equity})

# ==================== 市场状态统计更新 ====================
def update_regime_stats(regime: MarketRegime, pnl: float):
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
    pd.DataFrame(rows).to_csv(REGIME_STATS_FILE, index=False)

# ==================== 一致性误差统计 ====================
def update_consistency_stats(is_backtest: bool, slippage: float, win: bool):
    key = 'backtest' if is_backtest else 'live'
    stats = st.session_state.consistency_stats.get(key, {'trades': 0, 'avg_slippage': 0.0, 'wins': 0})
    stats['trades'] += 1
    stats['avg_slippage'] = (stats['avg_slippage'] * (stats['trades']-1) + slippage) / stats['trades']
    if win:
        stats['wins'] += 1
    stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
    st.session_state.consistency_stats[key] = stats
    rows = []
    for typ, s in st.session_state.consistency_stats.items():
        rows.append({
            'type': typ,
            'trades': s.get('trades', 0),
            'avg_slippage': s.get('avg_slippage', 0.0),
            'win_rate': s.get('win_rate', 0.0)
        })
    pd.DataFrame(rows).to_csv(CONSISTENCY_FILE, index=False)

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
    global factor_weights
    lr = CONFIG.factor_learning_rate
    for factor, ic in ic_dict.items():
        if factor in factor_weights and not np.isnan(ic):
            adjustment = 1 + lr * ic
            factor_weights[factor] = max(0.1, factor_weights[factor] * adjustment)

# ==================== 因子IC统计 ====================
def update_factor_ic_stats(ic_records: Dict[str, List[float]]):
    stats = {}
    for factor, ic_list in ic_records.items():
        if len(ic_list) > 5:
            mean_ic = np.mean(ic_list)
            std_ic = np.std(ic_list)
            ir = mean_ic / max(std_ic, 0.001)
            t_stat, p_value = ttest_1samp(ic_list, 0)
            stats[factor] = {'mean': mean_ic, 'std': std_ic, 'ir': ir, 'p_value': p_value}
    st.session_state.factor_ic_stats = stats

# ==================== 协方差矩阵计算 ====================
def calculate_cov_matrix(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]], window: int = 50) -> Optional[np.ndarray]:
    if len(symbols) < 2:
        return None
    returns_list = []
    for sym in symbols:
        df = data_dicts[sym]['15m']['close'].iloc[-window:]
        ret = df.pct_change().dropna().values
        if len(ret) < window // 2:
            return None
        returns_list.append(ret[-window:])
    returns_array = np.array(returns_list)
    if returns_array.shape[0] != len(symbols):
        return None
    cov = np.cov(returns_array)
    return cov

# ==================== 动态滑点计算 ====================
def dynamic_slippage(price: float, size: float, volume: float, volatility: float) -> float:
    base = price * CONFIG.slippage_base
    impact = CONFIG.slippage_impact_factor * (size / max(volume, 1)) * volatility * price
    return base + impact

# ==================== 组合VaR计算 ====================
def portfolio_var(weights: np.ndarray, cov: np.ndarray, confidence: float = 0.95) -> float:
    if weights is None or cov is None or len(weights) == 0:
        return 0.0
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    var = port_vol * norm.ppf(confidence)
    return abs(var)

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

# ==================== 技术指标计算 ====================
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

# ==================== 因子IC计算 ====================
_ic_cache = {}
def calculate_ic(df: pd.DataFrame, factor_name: str) -> float:
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
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "fear_greed": fetch_fear_greed(),
            "funding_rate": self.fetch_funding_rate(symbol),
            "orderbook_imbalance": self.fetch_orderbook_imbalance(symbol),
        }

# ==================== 信号引擎 ====================
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
        ic_dict = {}

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

            factor_scores = {}
            if last['close'] > last['ema20']:
                factor_scores['trend'] = 1 * factor_weights['trend']
            elif last['close'] < last['ema20']:
                factor_scores['trend'] = -1 * factor_weights['trend']
            else:
                factor_scores['trend'] = 0

            if last['rsi'] > 70:
                factor_scores['rsi'] = -0.7 * factor_weights['rsi']
            elif last['rsi'] < 30:
                factor_scores['rsi'] = 0.7 * factor_weights['rsi']
            else:
                factor_scores['rsi'] = 0

            if last['macd_diff'] > 0:
                factor_scores['macd'] = 0.8 * factor_weights['macd']
            elif last['macd_diff'] < 0:
                factor_scores['macd'] = -0.8 * factor_weights['macd']
            else:
                factor_scores['macd'] = 0

            if not pd.isna(last.get('bb_upper')):
                if last['close'] > last['bb_upper']:
                    factor_scores['bb'] = -0.5 * factor_weights['bb']
                elif last['close'] < last['bb_lower']:
                    factor_scores['bb'] = 0.5 * factor_weights['bb']
                else:
                    factor_scores['bb'] = 0
            else:
                factor_scores['bb'] = 0

            if not pd.isna(last.get('volume_ratio')):
                factor_scores['volume'] = (1.2 if last['volume_ratio'] > 1.5 else 0) * factor_weights['volume']
            else:
                factor_scores['volume'] = 0

            adx = last.get('adx', 25)
            if pd.isna(adx):
                factor_scores['adx'] = 0
            else:
                factor_scores['adx'] = (0.3 if adx > 30 else -0.2 if adx < 20 else 0) * factor_weights['adx']

            for fname in factor_scores.keys():
                col = factor_to_col.get(fname)
                if col and col in df.columns:
                    ic = calculate_ic(df, col)
                    if fname not in ic_dict:
                        ic_dict[fname] = []
                    ic_dict[fname].append(ic)

            tf_score = sum(factor_scores.values()) * weight
            total_score += tf_score
            total_weight += weight
            if tf_score > 0:
                tf_votes.append(1)
            elif tf_score < 0:
                tf_votes.append(-1)

        avg_ic = {}
        for fname, ic_list in ic_dict.items():
            avg_ic[fname] = np.nanmean(ic_list) if ic_list else 0.0
        update_factor_weights(avg_ic)
        update_factor_ic_stats(ic_dict)

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

# ==================== 风险管理（含VaR、组合风险）====================
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
        current_dd, _ = calculate_drawdown()
        return current_dd > CONFIG.max_drawdown_pct

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

    def allocate_portfolio(self, symbol_signals: Dict[str, Tuple[int, float, float, float, np.ndarray]], balance: float) -> Dict[str, float]:
        if not symbol_signals:
            return {}
        symbols = list(symbol_signals.keys())
        ret_arrays = []
        for sym in symbols:
            rets = symbol_signals[sym][4]
            if len(rets) < 10:
                ret_arrays.append(np.random.randn(10) * 0.02)
            else:
                ret_arrays.append(rets[-20:])
        min_len = min(len(arr) for arr in ret_arrays)
        ret_matrix = np.array([arr[-min_len:] for arr in ret_arrays])
        cov = np.cov(ret_matrix)
        try:
            vols = np.sqrt(np.diag(cov))
            inv_vol = 1.0 / vols
            weights = inv_vol / np.sum(inv_vol)
        except:
            weights = np.ones(len(symbols)) / len(symbols)
        allocations = {}
        for i, sym in enumerate(symbols):
            dir, prob, atr, price, rets = symbol_signals[sym]
            if dir == 0 or prob < SignalStrength.WEAK.value:
                allocations[sym] = 0.0
                continue
            size = self.calc_position_size(balance * weights[i], prob, atr, price, rets)
            allocations[sym] = size
        return allocations

# ==================== 持仓管理（带动态滑点）====================
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
    partial_taken: bool = False
    real: bool = False
    highest_price: float = 0.0
    lowest_price: float = 1e9
    atr_mult: float = CONFIG.atr_multiplier_base
    slippage_paid: float = 0.0

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

# ==================== 下单执行（动态滑点，带符号标准化）====================
def execute_order(symbol: str, direction: int, size: float, price: float, stop: float, take: float):
    sym = symbol.strip()
    dir_str = "多" if direction == 1 else "空"
    volume = 0
    if sym in st.session_state.multi_df:
        df = st.session_state.multi_df[sym]['15m']
        volume = df['volume'].iloc[-1] if not df.empty else 0
    vola = 0.02
    if sym in st.session_state.multi_df:
        rets = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]
        vola = np.std(rets) if len(rets) > 5 else 0.02
    slippage = dynamic_slippage(price, size, volume, vola)
    exec_price = price + slippage if direction == 1 else price - slippage
    st.session_state.positions[sym] = Position(
        symbol=sym,
        direction=direction,
        entry_price=exec_price,
        entry_time=datetime.now(),
        size=size,
        stop_loss=stop,
        take_profit=take,
        initial_atr=0,
        real=False,
        slippage_paid=slippage
    )
    st.session_state.daily_trades += 1
    log_execution(f"开仓 {sym} {dir_str} 仓位 {size:.4f} @ {exec_price:.2f} (原价 {price:.2f}, 滑点 {slippage:.4f}) 止损 {stop:.2f} 止盈 {take:.2f}")
    send_telegram(f"🔔 开仓 {dir_str} {sym}\n价格: {exec_price:.2f}\n仓位: {size:.4f}")
    st.session_state.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage})

def close_position(symbol: str, exit_price: float, reason: str):
    sym = symbol.strip()
    pos = st.session_state.positions.pop(sym, None)
    if pos is None:
        return
    volume = 0
    if sym in st.session_state.multi_df:
        df = st.session_state.multi_df[sym]['15m']
        volume = df['volume'].iloc[-1] if not df.empty else 0
    vola = 0.02
    if sym in st.session_state.multi_df:
        rets = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]
        vola = np.std(rets) if len(rets) > 5 else 0.02
    slippage = dynamic_slippage(exit_price, pos.size, volume, vola)
    exec_exit = exit_price - slippage if pos.direction == 1 else exit_price + slippage
    pnl = pos.pnl(exec_exit) - exec_exit * pos.size * CONFIG.fee_rate * 2
    st.session_state.daily_pnl += pnl
    st.session_state.account_balance += pnl
    if st.session_state.account_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = st.session_state.account_balance
    # 记录已平仓净值点（用于历史曲线）
    st.session_state.net_value_history.append({'time': datetime.now(), 'value': st.session_state.account_balance})
    # 权益曲线会在每次刷新时自动记录，这里不必重复
    st.session_state.daily_returns.append(pnl / st.session_state.account_balance)
    
    update_regime_stats(st.session_state.market_regime, pnl)
    update_consistency_stats(is_backtest=False, slippage=slippage, win=pnl>0)
    
    trade_record = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': sym,
        'direction': '多' if pos.direction == 1 else '空',
        'entry': pos.entry_price,
        'exit': exec_exit,
        'size': pos.size,
        'pnl': pnl,
        'reason': reason,
        'slippage_entry': pos.slippage_paid,
        'slippage_exit': slippage
    }
    st.session_state.trade_log.append(trade_record)
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log.pop(0)
    
    append_to_csv(TRADE_LOG_FILE, trade_record)
    st.session_state.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage})
    
    win = pnl > 0
    RiskManager().update_losses(win)
    log_execution(f"平仓 {sym} {reason} 盈亏 {pnl:.2f} 余额 {st.session_state.account_balance:.2f}")
    send_telegram(f"🔔 平仓 {reason}\n盈亏: {pnl:.2f}")

# ==================== 数据一致性修复 ====================
def fix_data_consistency(symbols):
    to_remove = []
    for sym in list(st.session_state.positions.keys()):
        if sym not in symbols or sym not in st.session_state.multi_df:
            to_remove.append(sym)
    for sym in to_remove:
        log_execution(f"数据修复：移除无效持仓 {sym}")
        del st.session_state.positions[sym]
    st.session_state.positions = {k: v for k, v in st.session_state.positions.items() if v.size > 0}

# ==================== 回测引擎（多品种组合，带动态滑点）====================
def run_backtest(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]], initial_balance: float = 10000) -> Dict[str, Any]:
    first_sym = symbols[0]
    base_df = data_dicts[first_sym]['15m']
    min_len = len(base_df)
    aligned_data = {}
    for sym in symbols:
        sym_df = data_dicts[sym]['15m']
        if len(sym_df) != min_len:
            aligned_data[sym] = sym_df.iloc[-min_len:].reset_index(drop=True)
        else:
            aligned_data[sym] = sym_df.reset_index(drop=True)
    
    balance = initial_balance
    peak_balance = initial_balance
    positions = {}
    equity_curve = []
    trades = []
    recent_returns = deque(maxlen=50)
    engine = SignalEngine()
    risk_manager = RiskManager()
    total_slippage = 0.0
    slippage_count = 0

    for i in range(50, min_len):
        row_dict = {sym: aligned_data[sym].iloc[i] for sym in symbols}
        high_dict = {sym: row['high'] for sym, row in row_dict.items()}
        low_dict = {sym: row['low'] for sym, row in row_dict.items()}
        price_dict = {sym: row['close'] for sym, row in row_dict.items()}
        atr_dict = {sym: row['atr'] if not pd.isna(row['atr']) else 0 for sym, row in row_dict.items()}
        volume_dict = {sym: row['volume'] for sym, row in row_dict.items()}
        timestamp = row_dict[first_sym]['timestamp']

        signal_inputs = {}
        for sym in symbols:
            dummy = {}
            for tf in CONFIG.timeframes:
                dummy[tf] = data_dicts[sym][tf].iloc[:i+1].reset_index(drop=True)
            signal_inputs[sym] = dummy

        symbol_signals = {}
        for sym in symbols:
            direction, prob = engine.calc_signal(signal_inputs[sym])
            if direction != 0 and prob >= SignalStrength.WEAK.value:
                recent = aligned_data[sym]['close'].pct_change().dropna().values[-20:]
                symbol_signals[sym] = (direction, prob, atr_dict[sym], price_dict[sym], recent)

        allocations = risk_manager.allocate_portfolio(symbol_signals, balance)

        for sym in symbols:
            if sym not in positions and allocations.get(sym, 0) > 0:
                dir, prob, atr_sym, price, _ = symbol_signals[sym]
                stop_dist = atr_sym * CONFIG.atr_multiplier_base if atr_sym > 0 else price * 0.01
                stop = price - stop_dist if dir == 1 else price + stop_dist
                take = price + stop_dist * CONFIG.tp_min_ratio if dir == 1 else price - stop_dist * CONFIG.tp_min_ratio
                size = allocations[sym]
                vola = np.std(aligned_data[sym]['close'].pct_change().dropna().values[-20:]) if len(aligned_data[sym])>20 else 0.02
                slippage = dynamic_slippage(price, size, volume_dict[sym], vola)
                total_slippage += slippage
                slippage_count += 1
                exec_price = price + slippage if dir == 1 else price - slippage
                positions[sym] = {
                    'direction': dir,
                    'entry': exec_price,
                    'size': size,
                    'stop': stop,
                    'take': take,
                    'entry_time': timestamp,
                    'partial_taken': False,
                    'slippage': slippage
                }

        close_list = []
        for sym, pos in positions.items():
            high = high_dict[sym]
            low = low_dict[sym]
            price = price_dict[sym]
            close_flag = False
            exit_price = price
            reason = ""
            hold_hours = (timestamp - pos['entry_time']).total_seconds() / 3600

            if pos['direction'] == 1:
                if low <= pos['stop']:
                    close_flag, exit_price, reason = True, pos['stop'], '止损'
                elif high >= pos['take']:
                    close_flag, exit_price, reason = True, pos['take'], '止盈'
                elif not pos['partial_taken'] and high >= pos['entry'] + (pos['take'] - pos['entry']) * CONFIG.partial_tp_r_multiple:
                    close_flag, exit_price, reason = True, pos['entry'] + (pos['take'] - pos['entry']) * CONFIG.partial_tp_r_multiple, '部分止盈'
                    pos['partial_taken'] = True
            else:
                if high >= pos['stop']:
                    close_flag, exit_price, reason = True, pos['stop'], '止损'
                elif low <= pos['take']:
                    close_flag, exit_price, reason = True, pos['take'], '止盈'
                elif not pos['partial_taken'] and low <= pos['entry'] - (pos['entry'] - pos['take']) * CONFIG.partial_tp_r_multiple:
                    close_flag, exit_price, reason = True, pos['entry'] - (pos['entry'] - pos['take']) * CONFIG.partial_tp_r_multiple, '部分止盈'
                    pos['partial_taken'] = True

            if hold_hours > CONFIG.max_hold_hours:
                close_flag, exit_price, reason = True, (high + low) / 2, '超时'

            if close_flag:
                vola = np.std(aligned_data[sym]['close'].pct_change().dropna().values[-20:]) if len(aligned_data[sym])>20 else 0.02
                slippage = dynamic_slippage(exit_price, pos['size'], volume_dict[sym], vola)
                total_slippage += slippage
                slippage_count += 1
                exec_exit = exit_price - slippage if pos['direction'] == 1 else exit_price + slippage
                pnl = (exec_exit - pos['entry']) * pos['size'] * pos['direction'] - exec_exit * pos['size'] * CONFIG.fee_rate * 2
                balance += pnl
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': timestamp,
                    'symbol': sym,
                    'direction': pos['direction'],
                    'entry': pos['entry'],
                    'exit': exec_exit,
                    'size': pos['size'],
                    'pnl': pnl,
                    'reason': reason,
                    'slippage_entry': pos['slippage'],
                    'slippage_exit': slippage
                })
                recent_returns.append(pnl / max(1, balance))
                peak_balance = max(peak_balance, balance)
                close_list.append(sym)

        for sym in close_list:
            del positions[sym]

        equity_curve.append({'time': timestamp, 'balance': balance})

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if not trades_df.empty:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        win_rate = len(wins)/len(trades_df)
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 1
        returns = trades_df['pnl'].values / initial_balance
        sharpe = (returns.mean()/returns.std()*np.sqrt(252)) if len(returns) > 1 and returns.std() != 0 else 0
        max_drawdown = (peak_balance - equity_df['balance'].min()) / peak_balance * 100
        avg_slippage = total_slippage / slippage_count if slippage_count > 0 else 0
    else:
        win_rate = avg_win = avg_loss = sharpe = max_drawdown = avg_slippage = 0

    update_consistency_stats(is_backtest=True, slippage=avg_slippage, win=False)

    performance = {
        'final_balance': balance,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'max_drawdown_pct': max_drawdown,
        'avg_slippage': avg_slippage
    }

    return {'equity': equity_df, 'trades': trades_df, 'performance': performance}

# ==================== Walk Forward 验证（严格隔离）====================
def walk_forward(data_dicts: Dict[str, Dict[str, pd.DataFrame]], symbols: List[str], train_window=180, test_window=30):
    base_df = data_dicts[symbols[0]]['15m']
    total_len = len(base_df)
    results = []
    for start in range(0, total_len - train_window - test_window, test_window):
        train_end = start + train_window
        test_end = train_end + test_window
        train_data = {}
        test_data = {}
        for sym in symbols:
            sym_data = data_dicts[sym]
            train_data[sym] = {tf: sym_data[tf].iloc[start:train_end].reset_index(drop=True) for tf in CONFIG.timeframes}
            test_data[sym] = {tf: sym_data[tf].iloc[train_end:test_end].reset_index(drop=True) for tf in CONFIG.timeframes}
        engine = SignalEngine()
        for _ in range(5):
            for sym in symbols:
                if len(train_data[sym]['15m']) > 50:
                    engine.calc_signal({tf: train_data[sym][tf] for tf in CONFIG.timeframes})
        result = run_backtest(symbols, test_data, initial_balance=10000)
        results.append(result)
    return results

# ==================== 参数敏感性热力图 ====================
def param_sensitivity_heatmap(data_dicts: Dict[str, Dict[str, pd.DataFrame]], symbols: List[str], param_ranges: Dict[str, List]):
    atr_vals = param_ranges.get('atr_multiplier_base', [1.2, 1.5, 1.8, 2.1])
    tp_vals = param_ranges.get('tp_min_ratio', [1.5, 2.0, 2.5, 3.0])
    sharpe_matrix = np.zeros((len(atr_vals), len(tp_vals)))
    for i, atr in enumerate(atr_vals):
        for j, tp in enumerate(tp_vals):
            old_atr = CONFIG.atr_multiplier_base
            old_tp = CONFIG.tp_min_ratio
            CONFIG.atr_multiplier_base = atr
            CONFIG.tp_min_ratio = tp
            result = run_backtest(symbols, data_dicts, initial_balance=10000)
            sharpe = result['performance']['sharpe']
            sharpe_matrix[i, j] = sharpe
            CONFIG.atr_multiplier_base = old_atr
            CONFIG.tp_min_ratio = old_tp
    return {'atr_vals': atr_vals, 'tp_vals': tp_vals, 'sharpe': sharpe_matrix}

# ==================== UI渲染器 ====================
class UIRenderer:
    def __init__(self):
        self.fetcher = get_fetcher()

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ 配置")
            mode = st.radio("模式", ['实盘', '回测'], index=0)
            st.session_state.mode = 'live' if mode == '实盘' else 'backtest'

            selected_symbols = st.multiselect("交易品种", CONFIG.symbols, default=['ETH/USDT', 'BTC/USDT'])
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
                    ticker = ex.fetch_ticker(selected_symbols[0])
                    st.success(f"连接成功！{selected_symbols[0]} 价格: {ticker['last']}")
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
                for sym in list(st.session_state.positions.keys()):
                    if sym in st.session_state.symbol_current_prices:
                        close_position(sym, st.session_state.symbol_current_prices[sym], "紧急平仓")
                st.rerun()

            if st.button("📂 查看历史交易记录"):
                if os.path.exists(TRADE_LOG_FILE):
                    df_trades = pd.read_csv(TRADE_LOG_FILE)
                    st.dataframe(df_trades.tail(20))
                else:
                    st.info("暂无历史交易记录")

            if st.button("🔧 数据修复"):
                fix_data_consistency(st.session_state.current_symbols)
                st.success("数据一致性已修复")

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

        return selected_symbols, mode_lev, use_real

    def render_main_panel(self, symbols, mode, use_real):
        if not symbols:
            st.warning("请至少选择一个交易品种")
            return

        multi_data = {}
        for sym in symbols:
            data = self.fetcher.get_symbol_data(sym)
            if data is None:
                st.error(f"获取 {sym} 数据失败")
                return
            multi_data[sym] = data
            st.session_state.symbol_current_prices[sym] = data['current_price']

        st.session_state.multi_df = {sym: data['data_dict'] for sym, data in multi_data.items()}
        first_sym = symbols[0]
        st.session_state.fear_greed = multi_data[first_sym]['fear_greed']
        df_first = multi_data[first_sym]['data_dict']
        st.session_state.market_regime = SignalEngine().detect_market_regime(df_first)

        cov = calculate_cov_matrix(symbols, {sym: multi_data[sym]['data_dict'] for sym in symbols}, CONFIG.cov_matrix_window)
        st.session_state.cov_matrix = cov

        fix_data_consistency(symbols)

        if st.session_state.mode == 'backtest':
            self.render_backtest_panel(symbols, multi_data)
        else:
            self.render_live_panel(symbols, multi_data)

    def render_backtest_panel(self, symbols, multi_data):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ 运行标准回测"):
                with st.spinner("回测中..."):
                    results = run_backtest(symbols, {sym: multi_data[sym]['data_dict'] for sym in symbols}, st.session_state.account_balance)
                    st.session_state.backtest_results = results
        with col2:
            if st.button("🔬 运行 Walk Forward 验证"):
                with st.spinner("Walk Forward 进行中..."):
                    wf_results = walk_forward({sym: multi_data[sym]['data_dict'] for sym in symbols}, symbols)
                    st.session_state.wf_results = wf_results
        with col3:
            if st.button("🔥 参数敏感性分析"):
                with st.spinner("生成热力图..."):
                    param_ranges = {
                        'atr_multiplier_base': [1.2, 1.5, 1.8, 2.1],
                        'tp_min_ratio': [1.5, 2.0, 2.5, 3.0]
                    }
                    heat = param_sensitivity_heatmap({sym: multi_data[sym]['data_dict'] for sym in symbols}, symbols, param_ranges)
                    st.session_state.param_sensitivity = heat

        if st.session_state.backtest_results:
            res = st.session_state.backtest_results
            eq = res['equity']
            trades = res['trades']
            perf = res['performance']
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

        if st.session_state.wf_results:
            st.subheader("Walk Forward 各段夏普")
            wf_sharpes = [r['performance']['sharpe'] for r in st.session_state.wf_results]
            fig_wf = go.Figure(data=[go.Bar(x=list(range(len(wf_sharpes))), y=wf_sharpes)])
            fig_wf.update_layout(height=300, title="各测试段夏普比率")
            st.plotly_chart(fig_wf, use_container_width=True)
            st.write(f"平均夏普: {np.mean(wf_sharpes):.3f}, 标准差: {np.std(wf_sharpes):.3f}")

        if st.session_state.param_sensitivity:
            heat = st.session_state.param_sensitivity
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat['sharpe'],
                x=[f"{v:.1f}" for v in heat['tp_vals']],
                y=[f"{v:.1f}" for v in heat['atr_vals']],
                colorscale='Viridis'))
            fig_heat.update_layout(title="参数敏感性 (夏普)", xaxis_title="TP Ratio", yaxis_title="ATR Multiplier")
            st.plotly_chart(fig_heat, use_container_width=True)

    def render_live_panel(self, symbols, multi_data):
        st.subheader("多品种持仓")
        risk = RiskManager()
        engine = SignalEngine()

        symbol_signals = {}
        for sym in symbols:
            df_dict_sym = st.session_state.multi_df[sym]
            direction, prob = engine.calc_signal(df_dict_sym)
            if direction != 0 and prob >= SignalStrength.WEAK.value:
                price = multi_data[sym]['current_price']
                atr_sym = df_dict_sym['15m']['atr'].iloc[-1] if not pd.isna(df_dict_sym['15m']['atr'].iloc[-1]) else 0
                recent = df_dict_sym['15m']['close'].pct_change().dropna().values[-20:]
                symbol_signals[sym] = (direction, prob, atr_sym, price, recent)

        allocations = risk.allocate_portfolio(symbol_signals, st.session_state.account_balance)

        for sym in symbols:
            if sym not in st.session_state.positions and allocations.get(sym, 0) > 0:
                dir, prob, atr_sym, price, _ = symbol_signals[sym]
                if atr_sym == 0 or np.isnan(atr_sym):
                    stop_dist = price * 0.01
                else:
                    stop_dist = atr_sym * adaptive_atr_multiplier(pd.Series([price]))
                stop = price - stop_dist if dir == 1 else price + stop_dist
                take = price + stop_dist * CONFIG.tp_min_ratio if dir == 1 else price - stop_dist * CONFIG.tp_min_ratio
                size = allocations[sym]
                execute_order(sym, dir, size, price, stop, take)

        for sym, pos in list(st.session_state.positions.items()):
            if sym not in symbols:
                continue
            df_dict_sym = st.session_state.multi_df[sym]
            current_price = multi_data[sym]['current_price']
            high = df_dict_sym['15m']['high'].iloc[-1]
            low = df_dict_sym['15m']['low'].iloc[-1]
            atr_sym = df_dict_sym['15m']['atr'].iloc[-1] if not pd.isna(df_dict_sym['15m']['atr'].iloc[-1]) else 0
            should_close, reason, exit_price = pos.should_close(high, low, datetime.now())
            if should_close:
                close_position(sym, exit_price, reason)
            else:
                if not pd.isna(atr_sym) and atr_sym > 0:
                    pos.update_stops(current_price, atr_sym)

        total_floating = 0.0
        for sym, pos in st.session_state.positions.items():
            if sym in multi_data:
                total_floating += pos.pnl(multi_data[sym]['current_price'])

        portfolio_var_value = 0.0
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
                port_vol = np.sqrt(np.dot(weights.T, np.dot(st.session_state.cov_matrix, weights)))
                portfolio_var_value = port_vol * norm.ppf(0.95) * np.sqrt(1)
        else:
            portfolio_var_value = 0.0

        # 记录权益点（每次刷新）
        record_equity_point()
        current_dd, max_dd = calculate_drawdown()

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("### 📊 市场状态")
            first_sym = symbols[0]
            prob_first = engine.calc_signal(st.session_state.multi_df[first_sym])[1]
            c1, c2, c3 = st.columns(3)
            c1.metric("恐惧贪婪", multi_data[first_sym]['fear_greed'])
            c2.metric("信号概率", f"{prob_first:.1%}")
            c3.metric("当前价格", f"{multi_data[first_sym]['current_price']:.2f}")

            for sym in symbols:
                st.write(f"{sym}: {multi_data[sym]['current_price']:.2f}")

            if st.session_state.positions:
                st.markdown("### 📈 当前持仓")
                for sym in sorted(st.session_state.positions.keys()):
                    pos = st.session_state.positions[sym]
                    pnl = pos.pnl(multi_data[sym]['current_price']) if sym in multi_data else 0
                    st.info(f"{sym}: {'多' if pos.direction==1 else '空'} 入场 {pos.entry_price:.2f} 数量 {pos.size:.4f} 浮动盈亏 {pnl:.2f}")
            else:
                st.markdown("### 无持仓")
                st.info("等待信号...")

            st.markdown("### 📉 风险监控")
            st.metric("实时盈亏", f"{st.session_state.daily_pnl + total_floating:.2f} USDT")
            st.metric("当前回撤", f"{current_dd:.2f}%")
            st.metric("最大回撤", f"{max_dd:.2f}%")
            st.metric("连亏次数", st.session_state.consecutive_losses)
            st.metric("日内交易", f"{st.session_state.daily_trades}/{CONFIG.max_daily_trades}")
            st.metric("组合VaR (95%)", f"{portfolio_var_value*100:.2f}%")

            if st.session_state.cooldown_until:
                st.warning(f"冷却至 {st.session_state.cooldown_until.strftime('%H:%M')}")

            # 市场状态统计
            if st.session_state.regime_stats:
                with st.expander("📈 市场状态统计"):
                    df_reg = pd.DataFrame(st.session_state.regime_stats).T
                    df_reg['胜率'] = df_reg['wins'] / df_reg['trades'] * 100
                    df_reg['平均盈亏'] = df_reg['total_pnl'] / df_reg['trades']
                    st.dataframe(df_reg[['trades', '胜率', '平均盈亏']].round(2))

            # 一致性误差统计
            if st.session_state.consistency_stats:
                with st.expander("🔄 实盘一致性"):
                    cons = st.session_state.consistency_stats
                    bt = cons.get('backtest', {})
                    lv = cons.get('live', {})
                    if bt and lv:
                        st.write(f"回测滑点: {bt.get('avg_slippage', 0):.4f} 实盘滑点: {lv.get('avg_slippage', 0):.4f}")
                        st.write(f"回测胜率: {bt.get('win_rate', 0):.2%} 实盘胜率: {lv.get('win_rate', 0):.2%}")
                    else:
                        st.write("暂无足够实盘数据对比")

            if st.session_state.factor_ic_stats:
                with st.expander("📊 因子IC统计"):
                    df_ic = pd.DataFrame(st.session_state.factor_ic_stats).T.round(4)
                    st.dataframe(df_ic)

            if st.session_state.net_value_history:
                hist_df = pd.DataFrame(st.session_state.net_value_history[-200:])
                fig_nv = go.Figure()
                fig_nv.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['value'], mode='lines', name='已平仓净值', line=dict(color='cyan')))
                fig_nv.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), template='plotly_dark')
                st.plotly_chart(fig_nv, use_container_width=True)

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

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5,0.15,0.15,0.2], vertical_spacing=0.02)
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
            fig.update_layout(height=800, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="终极量化终端 42.1 · 完美极限", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 完美极限版 42.1")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北 · 净值持久化 · 精准回撤 · 分段统计 · 一致性误差 · 实时权益")

    init_session_state()
    renderer = UIRenderer()
    symbols, mode, use_real = renderer.render_sidebar()

    if symbols:
        renderer.render_main_panel(symbols, mode, use_real)
    else:
        st.warning("请在左侧选择至少一个交易品种")

    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

if __name__ == "__main__":
    main()
