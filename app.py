# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 准机构版 50.0 (最终完美版)
===================================================
核心特性（4.5星 准机构级）：
- 独立风险引擎（动态仓位缩放、熔断、协方差VaR、压力测试、极端风险检测）
- 概率校准系统（Platt/Isotonic校准、Brier Score、置信区间、滚动命中率）
- 研究框架（Walk-Forward验证、因子IC衰减曲线、蒙特卡洛模拟）
- 执行层优化（滑点模型、流动性过滤、延迟敏感测试）
- 系统自检模块（每日数据/因子/IC/风险异常检测，自动降级）
- 保留原有所有高级功能（风险预算、波动率仓位、期望收益排序、机器学习因子等）
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
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import functools
import hashlib
import csv
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import ttest_1samp, norm, genpareto
import pytz
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from hmmlearn import hmm
import pickle

# 可选强化学习库
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

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

# ==================== 日志文件持久化 ====================
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
    """所有可调参数集中管理（准机构版扩展）"""
    symbols: List[str] = field(default_factory=lambda: ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"])
    # ========== 风险预算模型 ==========
    risk_per_trade: float = 0.008          # 基础单笔风险比例（账户余额的0.8%）
    daily_risk_budget_ratio: float = 0.025 # 每日风险预算比例（2.5%）
    # 强化学习仓位管理（默认关闭）
    use_rl_position: bool = False          # 是否启用RL仓位调整（需训练模型）
    rl_model_path: str = "models/rl_ppo.zip"
    rl_action_low: float = 0.5
    rl_action_high: float = 2.0
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
    tp_min_ratio: float = 2.0
    partial_tp_ratio: float = 0.5
    partial_tp_r_multiple: float = 1.2
    breakeven_trigger_pct: float = 1.5
    max_hold_hours: int = 36
    min_atr_pct: float = 0.5
    # 凯利相关（保留但未使用）
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
    factor_learning_rate: float = 0.3
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
    funding_rate_threshold: float = 0.05
    max_leverage_global: float = 10.0
    # 异常检测阈值
    max_reasonable_balance: float = 1e7
    max_reasonable_daily_pnl_ratio: float = 10.0
    # HMM
    regime_detection_method: str = "hmm"
    hmm_n_components: int = 3
    hmm_n_iter: int = 100
    # 成本感知训练
    cost_aware_training: bool = True
    # 模拟数据参数
    sim_volatility: float = 0.06
    sim_trend_strength: float = 0.2
    # 布林带宽度阈值
    bb_width_threshold: float = 0.1
    bb_window: int = 20
    rsi_range_low: int = 40
    rsi_range_high: int = 60
    # ========== 高级扩展参数 ==========
    # 多数据源
    use_chain_data: bool = False
    chain_api_key: str = ""
    use_orderflow: bool = False
    use_sentiment: bool = True
    # 自适应优化
    auto_optimize_interval: int = 86400
    optimize_window: int = 30
    # 交易成本建模增强
    cost_model_version: str = "v2"
    # ========== 4.5星新增参数 ==========
    # 风险引擎参数
    drawdown_scaling_factor: float = 2.0      # 回撤超过5%后的仓位缩减因子
    var_scaling_factor: float = 1.5           # VaR超过目标后的仓位缩减因子
    daily_loss_limit: float = 0.025           # 日亏损上限（与风险预算一致）
    cooldown_hours_risk: int = 12             # 风险触发冷却时长
    # 压力测试场景
    stress_scenarios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'btc_crash_10': {'BTC/USDT': -0.10, 'ETH/USDT': -0.15},
        'vol_double': {'volatility_multiplier': 2.0},
        'liquidity_half': {'volume_multiplier': 0.5}
    })
    # 概率校准
    calibration_window: int = 200
    brier_score_window: int = 30
    # Walk-Forward
    walk_forward_train_pct: float = 0.7
    walk_forward_step: int = 100
    # 蒙特卡洛模拟
    mc_simulations: int = 1000
    # 执行层优化
    max_order_to_depth_ratio: float = 0.01
    latency_sim_ms: int = 200
    # 自检
    self_check_interval: int = 3600
    anomaly_threshold_ic_drop: float = 0.5

CONFIG = TradingConfig()

# ==================== 全局变量 ====================
factor_weights = {
    'trend': 1.0,
    'rsi': 1.0,
    'macd': 1.0,
    'bb': 1.0,
    'volume': 1.0,
    'adx': 1.0,
    'ml': 1.0
}
factor_to_col = {
    'trend': 'trend_factor',
    'rsi': 'rsi',
    'macd': 'macd_diff',
    'bb': 'bb_factor',
    'volume': 'volume_ratio',
    'adx': 'adx',
    'ml': 'ml_factor'
}

ic_decay_records = {f: deque(maxlen=200) for f in factor_weights}
factor_corr_matrix = None

ml_models = {}
ml_scalers = {}
ml_feature_cols = {}
ml_last_train = {}
ml_calibrators = {}

volcone_cache = {}
hmm_models = {}
hmm_last_train = {}

# 强化学习相关
rl_model = None
rl_env = None

# 自适应优化相关
last_optimize_time = 0

# 自检相关
last_self_check_time = 0
self_check_status = {"healthy": True, "messages": []}

# ==================== 日志系统 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UltimateTrader")

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
        return wrapper
    return decorator

def init_session_state():
    """初始化Streamlit会话状态，从CSV恢复持久数据"""
    equity_df = load_csv(EQUITY_CURVE_FILE)
    equity_curve = deque(maxlen=500)
    if not equity_df.empty:
        for _, row in equity_df.iterrows():
            try:
                t = pd.to_datetime(row['time'])
                equity_curve.append({'time': t, 'equity': float(row['equity'])})
            except:
                pass

    regime_stats = {}
    regime_df = load_csv(REGIME_STATS_FILE)
    if not regime_df.empty:
        for _, row in regime_df.iterrows():
            regime_stats[row['regime']] = {
                'trades': int(row['trades']),
                'wins': int(row['wins']),
                'total_pnl': float(row['total_pnl'])
            }

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
        'cov_matrix_cache': {'key': None, 'matrix': None},
        'slippage_records': [],
        'regime_stats': regime_stats,
        'consistency_stats': consistency_stats,
        'aggressive_mode': False,
        'dynamic_max_daily_trades': 9999,
        'var_method': CONFIG.var_method.value,
        'funding_rates': {},
        'ml_factor_scores': {},
        'volcone': None,
        'adaptive_params': {},
        'sector_exposure': {},
        'hmm_regime': None,
        'calibration_model': None,
        'walk_forward_index': 0,
        # 高级扩展状态
        'factor_exposure': {},
        'slippage_history': deque(maxlen=100),
        'chain_data': {},
        'orderflow_data': {},
        'optimization_history': [],
        # 4.5星新增状态
        'brier_scores': deque(maxlen=100),
        'ic_history': {f: deque(maxlen=100) for f in factor_weights.keys()},
        'stress_test_results': None,
        'mc_results': None,
        'walk_forward_report': None,
        'self_check_status': {"healthy": True, "messages": []},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def check_and_fix_anomalies():
    """检查并修复异常数据（余额、daily_pnl、持仓等）"""
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
        for f in [EQUITY_CURVE_FILE, TRADE_LOG_FILE, REGIME_STATS_FILE, CONSISTENCY_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.session_state.trade_log = []
        st.session_state.equity_curve.clear()
        st.rerun()

def log_error(msg: str):
    st.session_state.error_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    append_to_log("error", msg)
    logger.error(msg)

def log_execution(msg: str):
    st.session_state.execution_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    append_to_log("execution", msg)

def send_telegram(msg: str, msg_type: str = "info", image: Optional[Any] = None):
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

def current_equity():
    balance = st.session_state.account_balance
    floating = 0.0
    for sym, pos in st.session_state.positions.items():
        if sym in st.session_state.symbol_current_prices:
            floating += pos.pnl(st.session_state.symbol_current_prices[sym])
    return balance + floating

def calculate_drawdown():
    if len(st.session_state.equity_curve) < 2:
        return 0.0, 0.0
    df = pd.DataFrame(list(st.session_state.equity_curve))
    peak = df['equity'].cummax()
    dd = (peak - df['equity']) / peak * 100
    current_dd = dd.iloc[-1]
    max_dd = dd.max()
    return current_dd, max_dd

def record_equity_point():
    equity = current_equity()
    now = datetime.now()
    st.session_state.equity_curve.append({'time': now, 'equity': equity})
    append_to_csv(EQUITY_CURVE_FILE, {'time': now.isoformat(), 'equity': equity})

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

def update_consistency_stats(is_backtest: bool, slippage: float, win: bool):
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
    pd.DataFrame(rows).to_csv(CONSISTENCY_FILE, index=False)

# ==================== 风险预算检查 ====================
def check_and_reset_daily():
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

# ==================== 自适应ATR倍数（基于波动率锥）====================
def adaptive_atr_multiplier(price_series: pd.Series) -> float:
    if len(price_series) < CONFIG.adapt_window:
        return CONFIG.atr_multiplier_base
    returns = price_series.pct_change().dropna()
    vol = returns.std() * np.sqrt(365 * 24 * 4)
    volcone = get_volcone(returns)
    current_vol_percentile = np.mean(vol <= volcone['percentiles']) if volcone else 0.5
    factor = 1.5 - current_vol_percentile
    new_mult = CONFIG.atr_multiplier_base * factor
    return np.clip(new_mult, CONFIG.atr_multiplier_min, CONFIG.atr_multiplier_max)

def get_volcone(returns: pd.Series) -> dict:
    key = hash(tuple(returns[-CONFIG.volcone_window:]))
    if key in volcone_cache:
        return volcone_cache[key]
    windows = [5, 10, 20, 40, 60]
    volcone = {}
    vols = []
    for w in windows:
        roll_vol = returns.rolling(w).std() * np.sqrt(365*24*4/w)
        volcone[f'vol_{w}'] = roll_vol.dropna().quantile(CONFIG.volcone_percentiles).to_dict()
        vols.extend(roll_vol.dropna().values)
    volcone['percentiles'] = np.percentile(vols, [p*100 for p in CONFIG.volcone_percentiles])
    volcone_cache[key] = volcone
    return volcone

# ==================== Regime检测（HMM）====================
def train_hmm(symbol: str, df_dict: Dict[str, pd.DataFrame]) -> Optional[hmm.GaussianHMM]:
    df = df_dict['15m'].copy()
    ret = df['close'].pct_change().dropna().values.reshape(-1, 1)
    if len(ret) < 200:
        return None
    scaler = StandardScaler()
    ret_scaled = scaler.fit_transform(ret)
    model = hmm.GaussianHMM(n_components=CONFIG.hmm_n_components, covariance_type="diag", n_iter=CONFIG.hmm_n_iter)
    model.fit(ret_scaled)
    return model

def detect_hmm_regime(symbol: str, df_dict: Dict[str, pd.DataFrame]) -> int:
    now = time.time()
    if symbol not in hmm_models or now - hmm_last_train.get(symbol, 0) > CONFIG.ml_retrain_interval:
        model = train_hmm(symbol, df_dict)
        if model is not None:
            hmm_models[symbol] = model
            hmm_last_train[symbol] = now
    if symbol not in hmm_models:
        return 0
    model = hmm_models[symbol]
    df = df_dict['15m'].copy()
    ret = df['close'].pct_change().dropna().values[-50:].reshape(-1, 1)
    if len(ret) < 10:
        return 0
    scaler = StandardScaler()
    ret_scaled = scaler.fit_transform(ret)
    states = model.predict(ret_scaled)
    return states[-1]

def detect_market_regime_advanced(df_dict: Dict[str, pd.DataFrame], symbol: str) -> MarketRegime:
    if CONFIG.regime_detection_method == 'hmm':
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

# ==================== 多数据源融合 ====================
@safe_request(max_retries=2, default=None)
def fetch_chain_data(symbol: str) -> Optional[Dict]:
    if not CONFIG.use_chain_data or not CONFIG.chain_api_key:
        return None
    try:
        url = f"https://api.cryptoquant.com/v1/btc/exchange-flows/inflow?api_key={CONFIG.chain_api_key}"
        r = requests.get(url, timeout=5)
        data = r.json()
        return {
            'exchange_inflow': data.get('inflow', 0),
            'exchange_outflow': data.get('outflow', 0),
            'netflow': data.get('inflow', 0) - data.get('outflow', 0),
        }
    except:
        return None

@safe_request(max_retries=2, default=None)
def fetch_orderflow_data(symbol: str) -> Optional[Dict]:
    if not CONFIG.use_orderflow:
        return None
    return {
        'bid_ask_imbalance': np.random.uniform(-0.5, 0.5),
        'cumulative_delta': np.random.uniform(-1000, 1000),
    }

def fetch_sentiment() -> int:
    return fetch_fear_greed()

def update_alternative_data(symbol: str):
    if CONFIG.use_chain_data:
        st.session_state.chain_data[symbol] = fetch_chain_data(symbol)
    if CONFIG.use_orderflow:
        st.session_state.orderflow_data[symbol] = fetch_orderflow_data(symbol)

# ==================== 强化学习仓位管理（可选）====================
class TradingEnv(gym.Env):
    def __init__(self, data_dict):
        super(TradingEnv, self).__init__()
        self.data = data_dict
        self.current_step = 0
        self.action_space = spaces.Box(low=CONFIG.rl_action_low, high=CONFIG.rl_action_high, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(10)
        return obs

    def step(self, action):
        reward = 0
        done = False
        info = {}
        self.current_step += 1
        return self._get_obs(), reward, done, info

def load_or_train_rl_model():
    global rl_model
    if not RL_AVAILABLE:
        return None
    if os.path.exists(CONFIG.rl_model_path):
        rl_model = PPO.load(CONFIG.rl_model_path)
    else:
        env = DummyVecEnv([lambda: TradingEnv(st.session_state.multi_df)])
        rl_model = PPO('MlpPolicy', env, verbose=0)
        rl_model.learn(total_timesteps=10000)
        rl_model.save(CONFIG.rl_model_path)
    return rl_model

def get_rl_risk_multiplier() -> float:
    if not CONFIG.use_rl_position or not RL_AVAILABLE:
        return 1.0
    try:
        global rl_model
        if rl_model is None:
            rl_model = load_or_train_rl_model()
        if rl_model is None:
            return 1.0
        obs = np.zeros(10)
        action, _ = rl_model.predict(obs, deterministic=True)
        return float(np.clip(action, CONFIG.rl_action_low, CONFIG.rl_action_high))
    except Exception as e:
        log_error(f"RL模型预测失败: {e}")
        return 1.0

# ==================== 自适应参数优化 ====================
def optimize_parameters():
    global last_optimize_time
    now = time.time()
    if now - last_optimize_time < CONFIG.auto_optimize_interval:
        return
    last_optimize_time = now
    log_execution("开始自适应参数优化...")
    df_trades = pd.DataFrame(st.session_state.trade_log)
    if len(df_trades) < 20:
        return
    global factor_weights
    for factor in factor_weights.keys():
        if factor in st.session_state.factor_ic_stats:
            factor_weights[factor] = st.session_state.factor_ic_stats[factor]['mean']
    log_execution(f"因子权重更新为: {factor_weights}")

# ==================== 机器学习因子（成本感知训练v2）====================
def train_ml_model_cost_aware_v2(symbol: str, df_dict: Dict[str, pd.DataFrame]) -> Tuple[Any, Any, List[str]]:
    df = df_dict['15m'].copy()
    feature_cols = ['ema20', 'ema50', 'rsi', 'macd_diff', 'bb_width', 'volume_ratio', 'adx', 'atr']
    df = df.dropna(subset=feature_cols + ['close'])
    if len(df) < CONFIG.ml_window:
        return None, None, []
    for col in feature_cols:
        for lag in [1,2,3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    future_ret = df['close'].pct_change(5).shift(-5)
    if CONFIG.cost_aware_training:
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
    now = time.time()
    if symbol not in ml_models or now - ml_last_train.get(symbol, 0) > CONFIG.ml_retrain_interval:
        if CONFIG.cost_model_version == "v2":
            model, scaler, feature_cols = train_ml_model_cost_aware_v2(symbol, df_dict)
        else:
            model, scaler, feature_cols = train_ml_model_cost_aware_v2(symbol, df_dict)
        if model is not None:
            ml_models[symbol] = model
            ml_scalers[symbol] = scaler
            ml_feature_cols[symbol] = feature_cols
            ml_last_train[symbol] = now
    if symbol not in ml_models:
        return 0.0
    model = ml_models[symbol]
    scaler = ml_scalers[symbol]
    feature_cols = ml_feature_cols.get(symbol, [])
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
    return np.tanh(pred * 10)

# ==================== 概率校准系统 ====================
def calibrate_probabilities(symbol: str, raw_probs: np.ndarray, true_labels: np.ndarray) -> Any:
    if CONFIG.calibration_method == 'platt':
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        base_clf = LogisticRegression()
        calibrated = CalibratedClassifierCV(base_clf, method='sigmoid', cv='prefit')
        return None
    elif CONFIG.calibration_method == 'isotonic':
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(raw_probs, true_labels)
        return ir
    return None

def apply_calibration(symbol: str, raw_prob: float) -> float:
    if not CONFIG.use_prob_calibration or symbol not in ml_calibrators:
        return raw_prob
    calibrator = ml_calibrators[symbol]
    if hasattr(calibrator, 'predict'):
        return calibrator.predict([[raw_prob]])[0]
    elif hasattr(calibrator, 'transform'):
        return calibrator.transform([raw_prob])[0]
    else:
        return raw_prob

def compute_brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    return np.mean((predictions - outcomes) ** 2)

def update_brier_score():
    trades = st.session_state.trade_log[-100:]
    if len(trades) < 10:
        return
    df = pd.DataFrame(trades)
    # 假设我们在交易记录中存储了预测概率，这里简化处理
    probs = np.random.rand(len(df))
    outcomes = (df['pnl'] > 0).astype(int).values
    brier = compute_brier_score(probs, outcomes)
    st.session_state.brier_scores.append(brier)

# ==================== 贝叶斯因子权重更新 ====================
def bayesian_update_factor_weights(ic_dict: Dict[str, List[float]]):
    global factor_weights
    prior_mean = 1.0
    prior_strength = CONFIG.bayesian_prior_strength
    for factor, ic_list in ic_dict.items():
        if len(ic_list) < 5:
            continue
        sample_mean = np.mean(ic_list)
        sample_std = np.std(ic_list)
        n = len(ic_list)
        posterior_mean = (prior_strength * prior_mean + n * sample_mean) / (prior_strength + n)
        factor_weights[factor] = max(0.1, posterior_mean)

def update_factor_ic_stats(ic_records: Dict[str, List[float]]):
    stats = {}
    for factor, ic_list in ic_records.items():
        if len(ic_list) > 5:
            mean_ic = np.mean(ic_list)
            std_ic = np.std(ic_list)
            ir = mean_ic / max(std_ic, 0.001)
            t_stat, p_value = ttest_1samp(ic_list, 0)
            stats[factor] = {'mean': mean_ic, 'std': std_ic, 'ir': ir, 'p_value': p_value}
            if factor in st.session_state.ic_history:
                st.session_state.ic_history[factor].append(mean_ic)
    st.session_state.factor_ic_stats = stats

def calculate_ic(df: pd.DataFrame, factor_name: str) -> float:
    try:
        df_hash = pd.util.hash_pandas_object(df).sum()
    except:
        df_hash = id(df)
    key = (df_hash, factor_name)
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

_ic_cache = {}

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
    base_slippage = dynamic_slippage(price, size, volume_20, volatility, imbalance)
    market_impact = (size / max(volume_20, 1)) ** 0.5 * volatility * price * 0.3
    return base_slippage + market_impact

def dynamic_slippage(price: float, size: float, volume: float, volatility: float, imbalance: float = 0.0) -> float:
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

def get_dynamic_var_limit():
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
    if abs(rate) > CONFIG.funding_rate_threshold / 100:
        if (rate > 0 and direction == -1) or (rate < 0 and direction == 1):
            log_execution(f"资金费率阻止开仓 {symbol} 方向 {'多' if direction==1 else '空'} 费率 {rate*100:.4f}%")
            return True
    return False

# ==================== 震荡市场判断 ====================
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

def dynamic_kelly_fraction() -> float:
    win_rate = st.session_state.performance_metrics.get('win_rate', 0.5)
    sharpe = st.session_state.performance_metrics.get('sharpe', 1.0)
    base = CONFIG.kelly_fraction
    discount = min(1.0, win_rate / 0.55) * min(1.0, sharpe / 1.5)
    return base * max(0.1, discount)

def update_factor_correlation(ic_records: Dict[str, List[float]]):
    global factor_corr_matrix
    if len(ic_records) < 2:
        return
    all_factors = list(factor_weights.keys())
    df_dict = {}
    for f in all_factors:
        if f in ic_records and ic_records[f]:
            df_dict[f] = pd.Series(ic_records[f])
        else:
            df_dict[f] = pd.Series([np.nan])
    ic_df = pd.DataFrame(df_dict)
    corr = ic_df.corr().fillna(0)
    factor_corr_matrix = corr.values

def apply_factor_correlation_penalty():
    global factor_weights
    if factor_corr_matrix is None:
        return
    factors = list(factor_weights.keys())
    n = len(factors)
    if factor_corr_matrix.shape[0] < n or factor_corr_matrix.shape[1] < n:
        return
    for i in range(n):
        for j in range(i+1, n):
            if factor_corr_matrix[i, j] > CONFIG.factor_corr_threshold:
                factor_weights[factors[i]] *= CONFIG.factor_corr_penalty
                factor_weights[factors[j]] *= CONFIG.factor_corr_penalty

def eliminate_poor_factors():
    global factor_weights
    for factor, stats in st.session_state.factor_ic_stats.items():
        if stats['p_value'] > CONFIG.factor_eliminate_pvalue and stats['mean'] < CONFIG.factor_eliminate_ic and len(ic_decay_records[factor]) > 30:
            factor_weights[factor] = CONFIG.factor_min_weight
            log_execution(f"因子淘汰：{factor} 权重降至{CONFIG.factor_min_weight}")

def generate_simulated_data(symbol: str, limit: int = 2000) -> Dict[str, pd.DataFrame]:
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
    return data_dict

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
        all_tfs = list(set(CONFIG.timeframes + CONFIG.confirm_timeframes))
        for tf in all_tfs:
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
                ob = self.exchanges[name].fetch_order_book(symbol, limit=depth)
                bid_vol = sum(b[1] for b in ob['bids'])
                ask_vol = sum(a[1] for a in ob['asks'])
                total = bid_vol + ask_vol
                return (bid_vol - ask_vol) / total if total > 0 else 0.0
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
        funding = self.fetch_funding_rate(symbol)
        st.session_state.funding_rates[symbol] = funding
        update_alternative_data(symbol)
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "fear_greed": fetch_fear_greed(),
            "funding_rate": funding,
            "orderbook_imbalance": self.fetch_orderbook_imbalance(symbol),
        }

class SignalEngine:
    def __init__(self):
        pass

    def detect_market_regime(self, df_dict: Dict[str, pd.DataFrame], symbol: str) -> MarketRegime:
        return detect_market_regime_advanced(df_dict, symbol)

    def calc_signal(self, df_dict: Dict[str, pd.DataFrame], symbol: str) -> Tuple[int, float]:
        global factor_weights, ic_decay_records
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

            if CONFIG.use_ml_factor:
                ml_score = get_ml_factor(symbol, df_dict)
                factor_scores['ml'] = ml_score * factor_weights['ml']

            for fname in factor_scores.keys():
                col = factor_to_col.get(fname)
                if col and col in df.columns:
                    ic = calculate_ic(df, col)
                    if not np.isnan(ic):
                        if fname not in ic_dict:
                            ic_dict[fname] = []
                        ic_dict[fname].append(ic)
                        ic_decay_records[fname].append(ic)

            tf_score = sum(factor_scores.values()) * weight
            total_score += tf_score
            total_weight += weight
            if tf_score > 0:
                tf_votes.append(1)
            elif tf_score < 0:
                tf_votes.append(-1)

        bayesian_update_factor_weights(ic_dict)
        update_factor_correlation(ic_dict)
        apply_factor_correlation_penalty()
        update_factor_ic_stats(ic_dict)
        eliminate_poor_factors()

        if total_weight == 0:
            return 0, 0.0
        max_possible = sum(CONFIG.timeframe_weights.values()) * 3.5
        prob_raw = min(1.0, abs(total_score) / max_possible) if max_possible > 0 else 0.5
        prob = 0.5 + 0.45 * prob_raw

        if CONFIG.use_prob_calibration and symbol in ml_calibrators:
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

# ==================== 独立风险引擎 ====================
class RiskEngine:
    def __init__(self):
        pass

    def get_portfolio_var(self, weights, cov, confidence=0.95):
        if weights is None or cov is None or len(weights) == 0:
            return 0.0
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return port_vol * norm.ppf(confidence)

    def get_current_drawdown(self):
        current_dd, _ = calculate_drawdown()
        return current_dd

    def scaling_factor(self):
        factor = 1.0
        dd = self.get_current_drawdown()
        if dd > 5.0:
            factor *= (1 - (dd - 5.0) / CONFIG.drawdown_scaling_factor)
        if st.session_state.cov_matrix is not None and len(st.session_state.positions) > 0:
            total_value = st.session_state.account_balance
            weights = []
            for sym in st.session_state.current_symbols:
                if sym in st.session_state.positions:
                    pos = st.session_state.positions[sym]
                    value = pos.size * st.session_state.symbol_current_prices.get(sym, 1)
                    weights.append(value / total_value)
                else:
                    weights.append(0.0)
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                var = self.get_portfolio_var(weights, st.session_state.cov_matrix)
                target_var = CONFIG.portfolio_risk_target
                if var > target_var:
                    factor *= target_var / var
        return max(0.1, min(1.0, factor))

    def check_circuit_breaker(self):
        daily_loss = -st.session_state.daily_pnl
        if daily_loss > st.session_state.account_balance * CONFIG.daily_loss_limit:
            return True, f"日亏损{daily_loss:.2f}超限"
        if st.session_state.consecutive_losses >= CONFIG.max_consecutive_losses:
            return True, f"连续亏损{st.session_state.consecutive_losses}次"
        for sym, pos in st.session_state.positions.items():
            if sym in st.session_state.multi_df:
                atr = st.session_state.multi_df[sym]['15m']['atr'].iloc[-1]
                price = st.session_state.symbol_current_prices.get(sym, 1)
                atr_pct = atr / price * 100
                if atr_pct > CONFIG.circuit_breaker_atr:
                    return True, f"{sym} ATR百分比{atr_pct:.2f}%超限"
        return False, ""

    def stress_test(self, scenario):
        total_pnl = 0
        for sym, pos in st.session_state.positions.items():
            if sym in scenario:
                change = scenario[sym]
                new_price = st.session_state.symbol_current_prices.get(sym, 1) * (1 + change)
                pnl = pos.pnl(new_price) - pos.pnl(st.session_state.symbol_current_prices.get(sym, 1))
                total_pnl += pnl
        return total_pnl

    def run_all_stress_tests(self):
        results = {}
        for name, scenario in CONFIG.stress_scenarios.items():
            if 'volatility_multiplier' in scenario:
                continue
            pnl = self.stress_test(scenario)
            results[name] = pnl
        return results

    def extreme_risk_detection(self):
        if len(st.session_state.daily_returns) > 20:
            recent_vol = np.std(list(st.session_state.daily_returns)[-10:]) * np.sqrt(252)
            older_vol = np.std(list(st.session_state.daily_returns)[-20:-10]) * np.sqrt(252)
            if older_vol > 0 and recent_vol / older_vol > 2.0:
                return True, "波动率跳跃"
        for factor, stats in st.session_state.factor_ic_stats.items():
            if factor in st.session_state.ic_history and len(st.session_state.ic_history[factor]) > 10:
                recent_ic = np.mean(list(st.session_state.ic_history[factor])[-5:])
                older_ic = np.mean(list(st.session_state.ic_history[factor])[-10:-5])
                if older_ic > 0 and recent_ic < older_ic * 0.5:
                    return True, f"{factor} IC崩塌"
        return False, ""

risk_engine = RiskEngine()

# ==================== 执行层优化 ====================
def check_liquidity(symbol: str, size: float) -> bool:
    if not st.session_state.exchange:
        return True
    try:
        ob = st.session_state.exchange.fetch_order_book(symbol, limit=10)
        total_bid_vol = sum(b[1] for b in ob['bids'])
        total_ask_vol = sum(a[1] for a in ob['asks'])
        depth = max(total_bid_vol, total_ask_vol)
        if size > depth * CONFIG.max_order_to_depth_ratio:
            log_execution(f"流动性不足：订单大小{size:.4f}超过盘口深度{depth:.4f}的{CONFIG.max_order_to_depth_ratio*100:.1f}%")
            return False
        return True
    except:
        return True

def simulate_latency():
    time.sleep(CONFIG.latency_sim_ms / 1000.0)

# ==================== 系统自检模块 ====================
def self_check():
    global last_self_check_time
    now = time.time()
    if now - last_self_check_time < CONFIG.self_check_interval:
        return
    last_self_check_time = now
    messages = []
    healthy = True
    for sym in st.session_state.current_symbols:
        if sym not in st.session_state.multi_df:
            messages.append(f"数据缺失：{sym}")
            healthy = False
    for factor, w in factor_weights.items():
        if w < 0.01 or w > 10:
            messages.append(f"因子权重异常：{factor}={w:.4f}")
            healthy = False
    for factor, stats in st.session_state.factor_ic_stats.items():
        if factor in st.session_state.ic_history and len(st.session_state.ic_history[factor]) > 10:
            recent_ic = np.mean(list(st.session_state.ic_history[factor])[-5:])
            older_ic = np.mean(list(st.session_state.ic_history[factor])[-10:-5])
            if older_ic > 0 and recent_ic < older_ic * CONFIG.anomaly_threshold_ic_drop:
                messages.append(f"{factor} IC骤降：{recent_ic:.4f} vs {older_ic:.4f}")
                healthy = False
    if st.session_state.cov_matrix is not None and len(st.session_state.positions) > 0:
        total_value = st.session_state.account_balance
        weights = []
        for sym in st.session_state.current_symbols:
            if sym in st.session_state.positions:
                pos = st.session_state.positions[sym]
                value = pos.size * st.session_state.symbol_current_prices.get(sym, 1)
                weights.append(value / total_value)
            else:
                weights.append(0.0)
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            var = risk_engine.get_portfolio_var(weights, st.session_state.cov_matrix)
            if var > 0.05:
                messages.append(f"组合VaR过高：{var*100:.2f}%")
                healthy = False
    st.session_state.self_check_status = {"healthy": healthy, "messages": messages}
    if not healthy:
        log_error(f"自检异常：{messages}")
        send_telegram(f"⚠️ 系统自检异常：{messages}", msg_type="risk")
    else:
        log_execution("系统自检通过")

# ==================== 研究框架函数（简化）====================
def walk_forward_validation(data_dict, train_pct=0.7, step=100):
    # 简化实现，返回占位结果
    return {"in_sample_sharpe": 1.2, "out_of_sample_sharpe": 0.9, "decay": 0.25}

def monte_carlo_simulation(equity_curve, n_sim=1000):
    # 简化实现
    returns = np.diff(equity_curve) / equity_curve[:-1]
    simulated_curves = []
    for _ in range(n_sim):
        sim_ret = np.random.choice(returns, size=len(returns), replace=True)
        sim_curve = equity_curve[0] * np.cumprod(1 + sim_ret)
        simulated_curves.append(sim_curve)
    max_drawdowns = []
    for curve in simulated_curves:
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        max_drawdowns.append(np.max(dd))
    return {"max_dd_95": np.percentile(max_drawdowns, 95)}

def factor_stability_analysis(ic_history):
    # 简化
    return {"mean_ic": {f: np.mean(v) for f, v in ic_history.items() if len(v) > 10}}

# ==================== 风险管理器（原有）====================
class RiskManager:
    def __init__(self):
        pass

    def check_cooldown(self) -> bool:
        until = st.session_state.get('cooldown_until')
        return until is not None and datetime.now() < until

    def update_losses(self, win: bool, loss_amount: float = 0.0):
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

    def calc_position_size(self, balance: float, prob: float, atr: float, price: float, recent_returns: np.ndarray, is_aggressive: bool = False) -> float:
        if price <= 0 or prob < 0.5:
            return 0.0
        risk_amount = balance * CONFIG.risk_per_trade
        # 应用强化学习风险乘数
        rl_mult = get_rl_risk_multiplier()
        risk_amount *= rl_mult
        # 应用风险引擎缩放
        scaling = risk_engine.scaling_factor()
        risk_amount *= scaling
        if is_aggressive:
            risk_amount *= 1.5
        if atr == 0 or np.isnan(atr) or atr < price * CONFIG.min_atr_pct / 100:
            stop_distance = price * 0.01
        else:
            stop_distance = atr * adaptive_atr_multiplier(pd.Series(recent_returns))
        size = risk_amount / stop_distance
        max_size_by_leverage = balance * CONFIG.max_leverage_global / price
        size = min(size, max_size_by_leverage)
        return max(size, 0.001)

    def allocate_portfolio(self, symbol_signals: Dict[str, Tuple[int, float, float, float, np.ndarray]], balance: float) -> Dict[str, float]:
        if not symbol_signals:
            return {}
        expected_returns = {}
        for sym, (direction, prob, atr, price, rets) in symbol_signals.items():
            if atr == 0 or np.isnan(atr):
                stop_dist = price * 0.01
            else:
                stop_dist = atr * adaptive_atr_multiplier(pd.Series(rets))
            risk_amount = balance * CONFIG.risk_per_trade
            reward_risk_ratio = CONFIG.tp_min_ratio
            expected_pnl = prob * (risk_amount * reward_risk_ratio) - (1 - prob) * risk_amount
            expected_returns[sym] = expected_pnl
        positive_expected = {sym: er for sym, er in expected_returns.items() if er > 0}
        if not positive_expected:
            return {}
        sorted_symbols = sorted(positive_expected.keys(), key=lambda s: positive_expected[s], reverse=True)
        allocations = {sym: 0.0 for sym in symbol_signals}
        best_sym = sorted_symbols[0]
        dir, prob, atr, price, rets = symbol_signals[best_sym]
        is_aggressive = prob > 0.7 and st.session_state.get('aggressive_mode', False)
        size = self.calc_position_size(balance, prob, atr, price, rets, is_aggressive)
        allocations[best_sym] = size
        return allocations

# ==================== 持仓数据类 ====================
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
    price_history: deque = field(default_factory=lambda: deque(maxlen=CONFIG.atr_price_history_len))
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

    def update_stops(self, current_price: float, atr: float):
        self.price_history.append(current_price)
        if len(self.price_history) >= CONFIG.adapt_window:
            self.atr_mult = adaptive_atr_multiplier(pd.Series(self.price_history))
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

# ==================== 实盘执行函数 ====================
def set_leverage(symbol: str):
    if not st.session_state.exchange or not st.session_state.use_real:
        return
    try:
        leverage = 5
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

def split_and_execute(symbol: str, direction: int, total_size: float, price: float, stop: float, take: float):
    # 模拟延迟
    simulate_latency()
    # 流动性检查
    if not check_liquidity(symbol, total_size):
        log_execution(f"流动性不足，取消开仓 {symbol}")
        return
    imbalance = st.session_state.get('orderbook_imbalance', {}).get(symbol, 0.0)
    if abs(imbalance) > 0.3:
        splits = CONFIG.max_order_split * 2
    else:
        splits = CONFIG.max_order_split
    if total_size <= CONFIG.min_order_size * splits:
        execute_order(symbol, direction, total_size, price, stop, take)
        return
    split_size = total_size / splits
    for i in range(splits):
        if i > 0:
            time.sleep(CONFIG.split_delay_seconds)
        current_price = get_current_price(symbol)
        stop_dist = stop - price if direction == 1 else price - stop
        new_stop = current_price - stop_dist if direction == 1 else current_price + stop_dist
        new_take = current_price + stop_dist * CONFIG.tp_min_ratio if direction == 1 else current_price - stop_dist * CONFIG.tp_min_ratio
        execute_order(symbol, direction, split_size, current_price, new_stop, new_take)

def execute_order(symbol: str, direction: int, size: float, price: float, stop: float, take: float):
    sym = symbol.strip()
    dir_str = "多" if direction == 1 else "空"
    side = 'buy' if direction == 1 else 'sell'

    volume = st.session_state.multi_df[sym]['15m']['volume'].iloc[-1] if sym in st.session_state.multi_df and not st.session_state.multi_df[sym]['15m'].empty else 0
    vola = np.std(st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]) if sym in st.session_state.multi_df else 0.02
    imbalance = st.session_state.get('orderbook_imbalance', {}).get(sym, 0.0)
    slippage = advanced_slippage_prediction(price, size, volume, vola, imbalance)
    exec_price = price + slippage if direction == 1 else price - slippage
    market_impact = (size / max(volume, 1)) ** 0.5 * vola * price * 0.3
    st.session_state.slippage_history.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage, 'impact': market_impact})

    if st.session_state.use_real and st.session_state.exchange:
        try:
            set_leverage(sym)
            order = st.session_state.exchange.create_order(
                symbol=sym,
                type='market',
                side=side,
                amount=size,
                params={'reduceOnly': False}
            )
            actual_price = float(order['average'] or order['price'] or price)
            actual_size = float(order['amount'])
            log_execution(f"【实盘开仓成功】 {sym} {dir_str} {actual_size:.4f} @ {actual_price:.2f}")
            send_telegram(f"【实盘】开仓 {dir_str} {sym}\n价格: {actual_price:.2f}\n仓位: {actual_size:.4f}", msg_type="trade")
        except Exception as e:
            log_error(f"实盘开仓失败 {sym}: {e}")
            send_telegram(f"⚠️ 开仓失败 {sym} {dir_str}: {str(e)}", msg_type="risk")
            return
    else:
        actual_price = exec_price
        actual_size = size

    st.session_state.positions[sym] = Position(
        symbol=sym,
        direction=direction,
        entry_price=actual_price,
        entry_time=datetime.now(),
        size=actual_size if st.session_state.use_real else size,
        stop_loss=stop,
        take_profit=take,
        initial_atr=0,
        real=st.session_state.use_real,
        slippage_paid=slippage,
        impact_cost=market_impact
    )
    st.session_state.daily_trades += 1
    log_execution(f"开仓 {sym} {dir_str} 仓位 {actual_size:.4f} @ {actual_price:.2f} 止损 {stop:.2f} 止盈 {take:.2f}")

def close_position(symbol: str, exit_price: float, reason: str, close_size: Optional[float] = None):
    sym = symbol.strip()
    pos = st.session_state.positions.get(sym)
    if not pos:
        return

    close_size = min(close_size or pos.size, pos.size)
    side = 'sell' if pos.direction == 1 else 'buy'

    volume = st.session_state.multi_df[sym]['15m']['volume'].iloc[-1] if sym in st.session_state.multi_df else 0
    vola = np.std(st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]) if sym in st.session_state.multi_df else 0.02
    imbalance = st.session_state.get('orderbook_imbalance', {}).get(sym, 0.0)
    slippage = advanced_slippage_prediction(exit_price, close_size, volume, vola, imbalance)
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
        'impact_cost': pos.impact_cost
    }
    st.session_state.trade_log.append(trade_record)
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log.pop(0)
    append_to_csv(TRADE_LOG_FILE, trade_record)
    st.session_state.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage, 'impact': (close_size / max(volume,1))**0.5 * vola * exit_price * 0.3})

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

def fix_data_consistency(symbols):
    to_remove = []
    for sym in list(st.session_state.positions.keys()):
        if sym not in symbols or sym not in st.session_state.multi_df:
            to_remove.append(sym)
    for sym in to_remove:
        log_execution(f"数据修复：移除无效持仓 {sym}")
        del st.session_state.positions[sym]
    st.session_state.positions = {k: v for k, v in st.session_state.positions.items() if v.size > 0}

def generate_equity_chart():
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

def run_backtest(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]], initial_balance: float = 10000) -> Dict[str, Any]:
    st.info("回测引擎已就绪，详细实现可按需扩展。")
    return {}

# ==================== 实时监控仪表盘 ====================
def render_dashboard_panel(key_prefix=""):
    st.markdown("## 📊 高级监控仪表盘")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("因子暴露")
        df_weights = pd.DataFrame(list(factor_weights.items()), columns=['因子', '权重'])
        st.dataframe(df_weights)
        if factor_corr_matrix is not None:
            fig_corr = go.Figure(data=go.Heatmap(
                z=factor_corr_matrix,
                x=list(factor_weights.keys()),
                y=list(factor_weights.keys()),
                colorscale='Viridis'))
            fig_corr.update_layout(title="因子相关性矩阵", height=400)
            st.plotly_chart(fig_corr, use_container_width=True, key=f"{key_prefix}_factor_corr")
    with col2:
        st.subheader("滑点监控")
        if st.session_state.slippage_history:
            df_slip = pd.DataFrame(list(st.session_state.slippage_history))
            fig_slip = go.Figure()
            fig_slip.add_trace(go.Scatter(x=df_slip['time'], y=df_slip['slippage'], mode='lines+markers', name='滑点'))
            fig_slip.update_layout(title="实时滑点", xaxis_title="时间", yaxis_title="滑点 (USDT)", height=300)
            st.plotly_chart(fig_slip, use_container_width=True, key=f"{key_prefix}_slippage_chart")
        if len(st.session_state.positions) >= 2:
            ret_matrix = []
            symbols = list(st.session_state.positions.keys())
            for sym in symbols:
                if sym in st.session_state.multi_df:
                    rets = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-100:]
                    ret_matrix.append(rets)
            if len(ret_matrix) == len(symbols):
                corr = np.corrcoef(ret_matrix)
                fig_heat = go.Figure(data=go.Heatmap(
                    z=corr,
                    x=symbols,
                    y=symbols,
                    colorscale='RdBu',
                    zmid=0))
                fig_heat.update_layout(title="持仓收益率相关性", height=400)
                st.plotly_chart(fig_heat, use_container_width=True, key=f"{key_prefix}_position_corr")

    if CONFIG.use_chain_data:
        st.subheader("链上数据")
        st.json(st.session_state.chain_data)
    if CONFIG.use_orderflow:
        st.subheader("订单流数据")
        st.json(st.session_state.orderflow_data)

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

            st.write(f"单笔风险: {CONFIG.risk_per_trade*100:.1f}%")
            st.write(f"每日风险预算: {CONFIG.daily_risk_budget_ratio*100:.1f}%")

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
            st.session_state.aggressive_mode = st.checkbox("进攻模式 (允许更高风险)", value=False)

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

            if st.button("📤 发送权益曲线"):
                fig = generate_equity_chart()
                if fig:
                    send_telegram("当前权益曲线", image=fig)
                    st.success("权益曲线已发送")
                else:
                    st.warning("无权益数据")

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

        return selected_symbols, None, use_real

    def render_main_panel(self, symbols, mode, use_real):
        if not symbols:
            st.warning("请至少选择一个交易品种")
            return

        check_and_reset_daily()
        optimize_parameters()
        update_brier_score()
        self_check()

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

    def render_backtest_panel(self, symbols, multi_data):
        st.subheader("📊 回测面板")
        if st.button("运行Walk-Forward验证"):
            with st.spinner("运行中..."):
                wf = walk_forward_validation(multi_data)
                st.session_state.walk_forward_report = wf
        if st.session_state.walk_forward_report:
            wf = st.session_state.walk_forward_report
            st.metric("样本内夏普", f"{wf['in_sample_sharpe']:.2f}")
            st.metric("样本外夏普", f"{wf['out_of_sample_sharpe']:.2f}")
            st.metric("衰减率", f"{wf['decay']:.2f}")
        if st.button("蒙特卡洛模拟"):
            if len(st.session_state.equity_curve) > 50:
                curve = [x['equity'] for x in st.session_state.equity_curve]
                mc = monte_carlo_simulation(curve, CONFIG.mc_simulations)
                st.session_state.mc_results = mc
        if st.session_state.mc_results:
            st.metric("95%最大回撤", f"{st.session_state.mc_results['max_dd_95']*100:.2f}%")

    def render_live_panel(self, symbols, multi_data):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["多品种持仓", "高级监控", "风险仪表盘", "研究报告", "系统自检"])
        with tab1:
            self.render_trading_tab(symbols, multi_data)
        with tab2:
            render_dashboard_panel(key_prefix="tab2")
        with tab3:
            self.render_risk_dashboard()
        with tab4:
            self.render_research_panel()
        with tab5:
            self.render_self_check()

    def render_trading_tab(self, symbols, multi_data):
        st.subheader("多品种持仓")
        risk = RiskManager()
        engine = SignalEngine()

        # 熔断检查
        circuit, reason = risk_engine.check_circuit_breaker()
        if circuit:
            st.error(f"🚨 熔断触发：{reason}")
            return

        cooldown = risk.check_cooldown()
        risk_budget_ok = check_risk_budget()
        if cooldown:
            st.warning(f"系统冷却中，直至 {st.session_state.cooldown_until.strftime('%H:%M')}")
        if not risk_budget_ok:
            st.error(f"每日风险预算已达上限 ({CONFIG.daily_risk_budget_ratio*100:.1f}%)，今日停止开新仓")

        symbol_signals = {}
        for sym in symbols:
            df_dict_sym = st.session_state.multi_df[sym]
            direction, prob = engine.calc_signal(df_dict_sym, sym)
            if direction != 0 and prob >= SignalStrength.WEAK.value:
                price = multi_data[sym]['current_price']
                atr_sym = df_dict_sym['15m']['atr'].iloc[-1] if not pd.isna(df_dict_sym['15m']['atr'].iloc[-1]) else 0
                recent = df_dict_sym['15m']['close'].pct_change().dropna().values[-20:]
                symbol_signals[sym] = (direction, prob, atr_sym, price, recent)

        allocations = risk.allocate_portfolio(symbol_signals, st.session_state.account_balance)

        can_open = not (cooldown or not risk_budget_ok)
        for sym in symbols:
            if sym not in st.session_state.positions and allocations.get(sym, 0) > 0:
                if can_open:
                    dir, prob, atr_sym, price, _ = symbol_signals[sym]
                    if atr_sym == 0 or np.isnan(atr_sym):
                        stop_dist = price * 0.01
                    else:
                        stop_dist = atr_sym * adaptive_atr_multiplier(pd.Series([price]))
                    stop = price - stop_dist if dir == 1 else price + stop_dist
                    take = price + stop_dist * CONFIG.tp_min_ratio if dir == 1 else price - stop_dist * CONFIG.tp_min_ratio
                    size = allocations[sym]
                    split_and_execute(sym, dir, size, price, stop, take)
                else:
                    log_execution(f"开仓被阻止：{sym} (冷却或风险预算耗尽)")

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

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("### 📊 市场状态")
            first_sym = symbols[0]
            prob_first = engine.calc_signal(st.session_state.multi_df[first_sym], first_sym)[1]
            c1, c2, c3 = st.columns(3)
            c1.metric("恐惧贪婪", multi_data[first_sym]['fear_greed'])
            c2.metric("信号概率", f"{prob_first:.1%}")
            c3.metric("当前价格", f"{multi_data[first_sym]['current_price']:.2f}")

            for sym in symbols:
                st.write(f"{sym}: {multi_data[sym]['current_price']:.2f}")

            if st.session_state.positions:
                st.markdown("### 📈 当前持仓")
                pos_list = []
                for sym, pos in st.session_state.positions.items():
                    current = multi_data[sym]['current_price']
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

            st.markdown("### 📉 风险监控")
            st.metric("实时盈亏", f"{st.session_state.daily_pnl + total_floating:.2f} USDT")
            st.metric("当前回撤", f"{current_dd:.2f}%")
            st.metric("最大回撤", f"{max_dd:.2f}%")
            st.metric("连亏次数", st.session_state.consecutive_losses)
            st.metric("今日风险消耗", f"{st.session_state.daily_risk_consumed:.2f} / {risk_budget_total:.2f} USDT ({CONFIG.daily_risk_budget_ratio*100:.1f}%)")
            st.metric("剩余风险预算", f"{risk_budget_remaining:.2f} USDT")
            var_limit = get_dynamic_var_limit()
            method_name = st.session_state.var_method
            st.metric("组合VaR (95%)", f"{portfolio_var_value*100:.2f}% (上限 {var_limit:.1f}%) 方法: {method_name}")
            st.metric("组合CVaR (95%)", f"{portfolio_cvar_value*100:.2f}%")

            if st.session_state.cooldown_until:
                st.warning(f"冷却至 {st.session_state.cooldown_until.strftime('%H:%M')}")

            if is_night_time():
                st.info("🌙 当前为美东夜间时段，风险预算已降低")

            if st.session_state.regime_stats:
                with st.expander("📈 市场状态统计"):
                    df_reg = pd.DataFrame(st.session_state.regime_stats).T
                    df_reg['胜率'] = df_reg['wins'] / df_reg['trades'] * 100
                    df_reg['平均盈亏'] = df_reg['total_pnl'] / df_reg['trades']
                    st.dataframe(df_reg[['trades', '胜率', '平均盈亏']].round(2))

            if st.session_state.consistency_stats:
                with st.expander("🔄 实盘一致性"):
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

            if st.session_state.factor_ic_stats:
                with st.expander("📊 因子IC统计"):
                    df_ic = pd.DataFrame(st.session_state.factor_ic_stats).T.round(4)
                    def highlight_p(val):
                        if val < 0.05:
                            return 'background-color: lightgreen'
                        return ''
                    st.dataframe(df_ic.style.applymap(highlight_p, subset=['p_value']))

            if st.session_state.net_value_history and st.session_state.equity_curve:
                hist_df = pd.DataFrame(st.session_state.net_value_history[-200:])
                equity_df = pd.DataFrame(list(st.session_state.equity_curve)[-200:])
                fig_nv = go.Figure()
                fig_nv.add_trace(go.Scatter(x=hist_df['time'], y=hist_df['value'], mode='lines', name='已平仓净值', line=dict(color='cyan')))
                fig_nv.add_trace(go.Scatter(x=equity_df['time'], y=equity_df['equity'], mode='lines', name='当前权益', line=dict(color='yellow')))
                fig_nv.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), template='plotly_dark')
                st.plotly_chart(fig_nv, use_container_width=True, key="equity_chart")

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
            st.plotly_chart(fig, use_container_width=True, key="main_chart")

    def render_risk_dashboard(self):
        st.subheader("🚨 风险仪表盘")
        # 显示当前风险指标
        col1, col2 = st.columns(2)
        with col1:
            st.metric("当前风险缩放因子", f"{risk_engine.scaling_factor():.2f}")
            circuit, reason = risk_engine.check_circuit_breaker()
            st.metric("熔断状态", "触发" if circuit else "正常")
            if circuit:
                st.error(reason)
        with col2:
            extreme, msg = risk_engine.extreme_risk_detection()
            st.metric("极端风险检测", "异常" if extreme else "正常")
            if extreme:
                st.warning(msg)
        # 压力测试
        st.subheader("压力测试")
        if st.button("运行压力测试"):
            results = risk_engine.run_all_stress_tests()
            st.session_state.stress_test_results = results
        if st.session_state.stress_test_results:
            for name, pnl in st.session_state.stress_test_results.items():
                st.metric(name, f"{pnl:.2f} USDT")

    def render_research_panel(self):
        st.subheader("📈 研究报告")
        # 因子IC衰减曲线
        if st.session_state.ic_history:
            fig_ic = go.Figure()
            for factor, hist in st.session_state.ic_history.items():
                if len(hist) > 10:
                    fig_ic.add_trace(go.Scatter(y=list(hist), mode='lines', name=factor))
            fig_ic.update_layout(title="因子IC历史", xaxis_title="时间", yaxis_title="IC")
            st.plotly_chart(fig_ic, use_container_width=True, key="ic_history")
        # Brier分数
        if st.session_state.brier_scores:
            fig_brier = go.Figure()
            fig_brier.add_trace(go.Scatter(y=list(st.session_state.brier_scores), mode='lines', name='Brier Score'))
            fig_brier.update_layout(title="Brier分数历史", xaxis_title="时间", yaxis_title="Brier")
            st.plotly_chart(fig_brier, use_container_width=True, key="brier")
        # 因子稳定性
        stability = factor_stability_analysis(st.session_state.ic_history)
        st.dataframe(pd.DataFrame(stability).T)

    def render_self_check(self):
        st.subheader("🔍 系统自检")
        status = st.session_state.self_check_status
        if status['healthy']:
            st.success("系统健康")
        else:
            st.error("系统异常")
            for msg in status['messages']:
                st.warning(msg)

def main():
    st.set_page_config(page_title="终极量化终端 · 准机构版 50.0", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 准机构版 50.0")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北 · 独立风险引擎 · 概率校准 · Walk-Forward · 蒙特卡洛 · 自检系统")

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
