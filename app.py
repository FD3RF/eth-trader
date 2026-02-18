# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 智能进化版 47.0
==================================================
核心特性（100% 完美极限 + 新增三阶段优化）：
1. 多周期共振信号（1m/5m/15m/1h/4h/1d） + 动态加权
2. 震荡市场过滤器（布林带宽度 + RSI区间）抑制假信号
3. 协方差风险平价 + 组合 VaR/CVaR 实时监控（支持正态/历史模拟法）
4. 动态每日交易次数（根据波动率自适应调整）
5. 进攻模式开关（短时提升风险预算，仓位放大）
6. 动态 ATR 止损/止盈（基于近20根K线波动率，1.2x - 2.5x 自适应）
7. 净值曲线持久化（含浮动盈亏，自动保存 equity_curve.csv）
8. 精准回撤计算（当前回撤 + 最大回撤，基于实时权益）
9. 市场状态分段统计（趋势/震荡/恐慌下的胜率、盈亏）
10. 实盘一致性误差统计（滑点对比 + 胜率对比 + 自动报警）
11. Telegram 增强通知（区分信号、风险、交易类型，支持发送权益曲线截图）
12. 一键数据修复（清理无效持仓） + 重置所有状态
13. 高性能并行数据获取（多交易所自动回退）
14. 完整日志持久化（交易日志、执行日志、错误日志）
15. 回测引擎 + Walk Forward 验证 + 参数敏感性热力图
16. 因子 IC 显著性检验（均值、标准差、信息比率、p 值，p<0.05 高亮）
17. 多品种支持（ETH/BTC/SOL/BNB 等，可自由添加）
18. 滑点 + 手续费精细建模（基于订单深度、波动率、订单簿不平衡）
19. 移动止损 + 比例部分止盈 + 保本止损 + 部分止盈后止损优化
20. 熔断机制（基于 ATR 百分比 + 恐惧贪婪指数）
21. 冷却机制（连续亏损后暂停交易）
22. 实时盈亏 + 当前回撤 + 最大回撤 + VaR/CVaR 联动显示
23. 图表 K 线 + 均线 + 持仓标记 + 交易记录可视化
24. 完全可配置参数（位于 TradingConfig 类中）
==================================================
新增优化（47.0 三阶段智能进化）：
- 阶段一：市场状态开仓过滤（只在趋势/恐慌下开仓），CVaR 计算，动态 Kelly 折扣（基于近期胜率/夏普）
- 阶段二：因子相关性动态降权，IC 滚动衰减（指数加权），因子淘汰（低 IC 且高 p 值自动降权）
- 阶段三：高级滑点预测（加入市场冲击项），订单拆分（TWAP 简化），冲击成本记录
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
import pytz

warnings.filterwarnings('ignore')

# ==================== 日志文件持久化 ====================
LOG_DIR = "logs"
TRADE_LOG_FILE = "trade_log.csv"
PERF_LOG_FILE = "performance_log.csv"
SLIPPAGE_LOG_FILE = "slippage_log.csv"
EQUITY_CURVE_FILE = "equity_curve.csv"
REGIME_STATS_FILE = "regime_stats.csv"
CONSISTENCY_FILE = "consistency_stats.csv"
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

class VaRMethod(Enum):
    NORMAL = "正态法"
    HISTORICAL = "历史模拟法"

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
    daily_trades_volatility_threshold: float = 0.5
    daily_trades_boost: int = 2
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
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h', '4h', '1d'])
    confirm_timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])
    timeframe_weights: Dict[str, int] = field(default_factory=lambda: {'1d': 10, '4h': 7, '1h': 5, '15m': 3, '5m': 2, '1m': 1})
    fetch_limit: int = 2000
    auto_refresh_ms: int = 30000
    anti_duplicate_seconds: int = 180
    kelly_fraction: float = 0.25
    atr_multiplier_base: float = 1.5
    atr_multiplier_min: float = 1.2
    atr_multiplier_max: float = 2.5
    max_leverage_global: float = 10.0
    circuit_breaker_atr: float = 5.0
    circuit_breaker_fg_extreme: Tuple[int, int] = (10, 90)
    slippage_base: float = 0.0005
    slippage_impact_factor: float = 0.1
    slippage_imbalance_factor: float = 0.5
    fee_rate: float = 0.0004
    ic_window: int = 80
    mc_simulations: int = 500
    sim_volatility: float = 0.06
    sim_trend_strength: float = 0.2
    adapt_window: int = 20
    factor_learning_rate: float = 0.3
    var_confidence: float = 0.95
    var_method: VaRMethod = VaRMethod.HISTORICAL
    var_aggressive_threshold: float = 1.0
    portfolio_risk_target: float = 0.02
    cov_matrix_window: int = 50
    max_drawdown_window: int = 100
    bb_width_threshold: float = 0.1
    rsi_range_low: int = 40
    rsi_range_high: int = 60
    signal_weight_boost: float = 1.5
    atr_price_history_len: int = 20
    funding_rate_threshold: float = 0.05
    night_start_hour: int = 0
    night_end_hour: int = 8
    night_risk_multiplier: float = 0.5
    # 阶段一：市场状态开仓过滤
    regime_allow_trade: List[MarketRegime] = field(default_factory=lambda: [MarketRegime.TREND, MarketRegime.PANIC])
    # 阶段二：因子相关性降权阈值
    factor_corr_threshold: float = 0.7
    factor_corr_penalty: float = 0.7
    # 阶段二：IC衰减率
    ic_decay_rate: float = 0.99
    # 阶段二：因子淘汰阈值
    factor_eliminate_pvalue: float = 0.1
    factor_eliminate_ic: float = 0.02
    factor_min_weight: float = 0.1
    # 阶段三：订单拆分
    max_order_split: int = 3
    min_order_size: float = 0.001
    split_delay_seconds: int = 5

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

# 阶段二：IC衰减记录
ic_decay_records = {f: deque(maxlen=200) for f in factor_weights}
factor_corr_matrix = None
last_corr_update = None

# ==================== 日志系统 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UltimateTrader")

# ==================== 辅助函数 ====================
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
        'dynamic_max_daily_trades': CONFIG.max_daily_trades,
        'var_method': CONFIG.var_method.value,
        'funding_rates': {},
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

# ==================== 实时权益计算 ====================
def current_equity():
    balance = st.session_state.account_balance
    floating = 0.0
    for sym, pos in st.session_state.positions.items():
        if sym in st.session_state.symbol_current_prices:
            floating += pos.pnl(st.session_state.symbol_current_prices[sym])
    return balance + floating

# ==================== 精准回撤计算 ====================
def calculate_drawdown():
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

# ==================== 动态每日交易次数 ====================
def update_daily_trades_limit(volatility: float):
    base = CONFIG.max_daily_trades
    if volatility > CONFIG.daily_trades_volatility_threshold:
        st.session_state.dynamic_max_daily_trades = base + CONFIG.daily_trades_boost
    else:
        st.session_state.dynamic_max_daily_trades = base

# ==================== 自适应ATR倍数（基于价格序列）====================
def adaptive_atr_multiplier(price_series: pd.Series) -> float:
    if len(price_series) < CONFIG.adapt_window:
        return CONFIG.atr_multiplier_base
    returns = price_series.pct_change().dropna()
    vol = returns.std() * np.sqrt(365 * 24 * 4)
    base_vol = 0.5
    ratio = base_vol / max(vol, 0.1)
    new_mult = CONFIG.atr_multiplier_base * np.clip(ratio, 0.5, 2.0)
    return np.clip(new_mult, CONFIG.atr_multiplier_min, CONFIG.atr_multiplier_max)

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

# ==================== 协方差矩阵计算（带缓存）====================
def calculate_cov_matrix(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]], window: int = 50) -> Optional[np.ndarray]:
    if len(symbols) < 2:
        return None
    # 生成缓存键：包含符号列表、窗口大小和最新数据的时间戳哈希
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

# ==================== 动态滑点计算（加入市场冲击项）====================
def advanced_slippage_prediction(price: float, size: float, volume_20: float, volatility: float, imbalance: float) -> float:
    base_slippage = dynamic_slippage(price, size, volume_20, volatility, imbalance)
    # 冲击预测
    market_impact = (size / max(volume_20, 1)) ** 0.5 * volatility * price * 0.3
    return base_slippage + market_impact

def dynamic_slippage(price: float, size: float, volume: float, volatility: float, imbalance: float = 0.0) -> float:
    base = price * CONFIG.slippage_base
    impact = CONFIG.slippage_impact_factor * (size / max(volume, 1)) * volatility * price
    imbalance_adj = 1 + abs(imbalance) * CONFIG.slippage_imbalance_factor
    return (base + impact) * imbalance_adj

# ==================== 组合VaR/CVaR计算 ====================
def portfolio_var(weights: np.ndarray, cov: np.ndarray, confidence: float = 0.95, method: str = "HISTORICAL", historical_returns: Optional[np.ndarray] = None) -> float:
    if weights is None or cov is None or len(weights) == 0:
        return 0.0
    if method == "HISTORICAL" and historical_returns is not None and historical_returns.shape[1] > 20:
        # 历史模拟法：组合收益 = 权重 @ 历史收益矩阵
        port_rets = weights @ historical_returns
        var = np.percentile(port_rets, (1 - confidence) * 100)
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

# ==================== 夜间时段判断 ====================
def is_night_time() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(pytz.utc).astimezone(eastern)
    hour = now_eastern.hour
    if hour >= CONFIG.night_start_hour and hour < CONFIG.night_end_hour:
        return True
    return False

# ==================== 资金费率过滤 ====================
def funding_rate_blocked(symbol: str, direction: int) -> bool:
    rate = st.session_state.funding_rates.get(symbol, 0.0)
    if abs(rate) > CONFIG.funding_rate_threshold / 100:
        if (rate > 0 and direction == -1) or (rate < 0 and direction == 1):
            log_execution(f"资金费率阻止开仓 {symbol} 方向 {'多' if direction==1 else '空'} 费率 {rate*100:.4f}%")
            return True
    return False

# ==================== 震荡过滤器 ====================
def is_range_market(df_dict: Dict[str, pd.DataFrame]) -> bool:
    if '15m' not in df_dict:
        return False
    df = df_dict['15m']
    last = df.iloc[-1]
    if not pd.isna(last.get('bb_width')):
        if last['bb_width'] < CONFIG.bb_width_threshold:
            return True
    if not pd.isna(last.get('rsi')):
        if CONFIG.rsi_range_low < last['rsi'] < CONFIG.rsi_range_high:
            return True
    return False

# ==================== 多周期共振确认 ====================
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

# ==================== 阶段一：市场状态开仓过滤 ====================
def can_open_position(regime: MarketRegime) -> bool:
    return regime in CONFIG.regime_allow_trade

# ==================== 阶段一：动态Kelly折扣 ====================
def dynamic_kelly_fraction() -> float:
    win_rate = st.session_state.performance_metrics.get('win_rate', 0.5)
    sharpe = st.session_state.performance_metrics.get('sharpe', 1.0)
    base = CONFIG.kelly_fraction
    discount = min(1.0, win_rate / 0.55) * min(1.0, sharpe / 1.5)
    return base * max(0.1, discount)

# ==================== 阶段二：因子相关性动态降权 ====================
def update_factor_correlation(ic_records: Dict[str, List[float]]):
    global factor_corr_matrix
    if len(ic_records) < 2:
        return
    ic_df = pd.DataFrame({k: pd.Series(v) for k, v in ic_records.items()})
    factor_corr_matrix = ic_df.corr().fillna(0).values

def apply_factor_correlation_penalty():
    global factor_weights
    if factor_corr_matrix is None:
        return
    factors = list(factor_weights.keys())
    n = len(factors)
    for i in range(n):
        for j in range(i+1, n):
            if factor_corr_matrix[i, j] > CONFIG.factor_corr_threshold:
                factor_weights[factors[i]] *= CONFIG.factor_corr_penalty
                factor_weights[factors[j]] *= CONFIG.factor_corr_penalty

# ==================== 阶段二：因子淘汰机制 ====================
def eliminate_poor_factors():
    global factor_weights
    for factor, stats in st.session_state.factor_ic_stats.items():
        if stats['p_value'] > CONFIG.factor_eliminate_pvalue and stats['mean'] < CONFIG.factor_eliminate_ic and len(ic_decay_records[factor]) > 30:
            factor_weights[factor] = CONFIG.factor_min_weight
            log_execution(f"因子淘汰：{factor} 权重降至{CONFIG.factor_min_weight}")

# ==================== 超真实模拟数据生成器 ====================
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

# ==================== 因子IC计算（缓存键优化）====================
_ic_cache = {}
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

# ==================== 独立缓存函数（每小时刷新）====================
@st.cache_data(ttl=3600, show_spinner=False)
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
        return {
            "data_dict": data_dict,
            "current_price": current_price,
            "fear_greed": fetch_fear_greed(),
            "funding_rate": funding,
            "orderbook_imbalance": self.fetch_orderbook_imbalance(symbol),
        }

# ==================== 信号引擎（增强IC衰减和因子降权）====================
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
        global factor_weights, ic_decay_records
        total_score = 0
        total_weight = 0
        tf_votes = []
        regime = st.session_state.market_regime
        ic_dict = {}  # 用于存储当前周期各因子的IC列表

        range_penalty = 0.5 if is_range_market(df_dict) else 1.0

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

            # 收集当前周期的IC
            for fname in factor_scores.keys():
                col = factor_to_col.get(fname)
                if col and col in df.columns:
                    ic = calculate_ic(df, col)
                    if not np.isnan(ic):
                        if fname not in ic_dict:
                            ic_dict[fname] = []
                        ic_dict[fname].append(ic)
                        ic_decay_records[fname].append(ic)  # 加入衰减记录

            tf_score = sum(factor_scores.values()) * weight
            total_score += tf_score
            total_weight += weight
            if tf_score > 0:
                tf_votes.append(1)
            elif tf_score < 0:
                tf_votes.append(-1)

        # 计算加权平均IC（带衰减）
        avg_ic = {}
        for fname, ic_list in ic_dict.items():
            if ic_list:
                # 从衰减记录中获取带权重的平均IC
                decayed_list = list(ic_decay_records[fname])
                if decayed_list:
                    weights = [CONFIG.ic_decay_rate ** i for i in range(len(decayed_list))]
                    weighted_ic = np.average(decayed_list, weights=weights[::-1])
                    avg_ic[fname] = weighted_ic
                else:
                    avg_ic[fname] = np.nanmean(ic_list)

        # 更新因子权重
        update_factor_weights(avg_ic)
        # 更新因子相关性矩阵
        update_factor_correlation(ic_dict)
        # 应用相关性降权
        apply_factor_correlation_penalty()
        # 更新IC统计（用于淘汰）
        update_factor_ic_stats(ic_dict)
        # 因子淘汰
        eliminate_poor_factors()

        if total_weight == 0:
            return 0, 0.0
        max_possible = sum(CONFIG.timeframe_weights.values()) * 3.5
        prob_raw = min(1.0, abs(total_score) / max_possible) if max_possible > 0 else 0.5
        prob = 0.5 + 0.45 * prob_raw

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

# ==================== 风险管理（增强：动态Kelly）====================
class RiskManager:
    def __init__(self):
        pass

    def check_daily_limit(self) -> bool:
        today = datetime.now().date()
        if st.session_state.get('last_trade_date') != today:
            st.session_state.daily_trades = 0
            st.session_state.last_trade_date = today
        return st.session_state.daily_trades >= st.session_state.dynamic_max_daily_trades

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

    def calc_position_size(self, balance: float, prob: float, atr: float, price: float, recent_returns: np.ndarray, is_aggressive: bool = False) -> float:
        if price <= 0 or prob < 0.5:
            return 0.0
        edge = max(0.05, prob - 0.5) * 2
        var = self.calc_var(recent_returns, CONFIG.var_confidence)
        risk_mult = 1.5 if is_aggressive else 1.0
        kelly = dynamic_kelly_fraction()  # 使用动态Kelly
        risk_amount = balance * CONFIG.base_risk_per_trade * edge * kelly * (1 / max(var, 0.01)) * risk_mult
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
            is_aggressive = prob > 0.7 and st.session_state.get('aggressive_mode', False)
            if funding_rate_blocked(sym, dir):
                allocations[sym] = 0.0
                continue
            # 阶段一：市场状态开仓过滤
            if not can_open_position(st.session_state.market_regime):
                allocations[sym] = 0.0
                continue
            size = self.calc_position_size(balance * weights[i], prob, atr, price, rets, is_aggressive)
            allocations[sym] = size
        return allocations

# ==================== 持仓管理 ====================
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
    impact_cost: float = 0.0  # 新增：冲击成本

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

# ==================== 订单拆分执行 ====================
def get_current_price(symbol: str) -> float:
    return st.session_state.symbol_current_prices.get(symbol, 0.0)

def split_and_execute(symbol: str, direction: int, total_size: float, price: float, stop: float, take: float):
    """将大订单拆分为多个小订单执行，每个子订单根据当前价格重新计算止损止盈"""
    if total_size <= CONFIG.min_order_size * CONFIG.max_order_split:
        execute_order(symbol, direction, total_size, price, stop, take)
        return
    split_size = total_size / CONFIG.max_order_split
    for i in range(CONFIG.max_order_split):
        if i > 0:
            time.sleep(CONFIG.split_delay_seconds)
        current_price = get_current_price(symbol)
        # 重新计算止损止盈（基于当前价格）
        stop_dist = stop - price if direction == 1 else price - stop  # 原止损距离
        new_stop = current_price - stop_dist if direction == 1 else current_price + stop_dist
        new_take = current_price + stop_dist * CONFIG.tp_min_ratio if direction == 1 else current_price - stop_dist * CONFIG.tp_min_ratio
        execute_order(symbol, direction, split_size, current_price, new_stop, new_take)

# ==================== 下单执行（使用高级滑点）====================
def execute_order(symbol: str, direction: int, size: float, price: float, stop: float, take: float):
    sym = symbol.strip()
    dir_str = "多" if direction == 1 else "空"
    volume = 0
    imbalance = 0.0
    if sym in st.session_state.multi_df:
        df = st.session_state.multi_df[sym]['15m']
        volume = df['volume'].iloc[-1] if not df.empty else 0
    vola = 0.02
    if sym in st.session_state.multi_df:
        rets = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]
        vola = np.std(rets) if len(rets) > 5 else 0.02
    if sym in st.session_state.symbol_current_prices:
        imbalance = st.session_state.get('orderbook_imbalance', {}).get(sym, 0.0)
    # 使用高级滑点预测
    slippage = advanced_slippage_prediction(price, size, volume, vola, imbalance)
    exec_price = price + slippage if direction == 1 else price - slippage
    # 计算冲击成本（用于记录）
    market_impact = (size / max(volume, 1)) ** 0.5 * vola * price * 0.3
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
        slippage_paid=slippage,
        impact_cost=market_impact
    )
    st.session_state.daily_trades += 1
    log_execution(f"开仓 {sym} {dir_str} 仓位 {size:.4f} @ {exec_price:.2f} (原价 {price:.2f}, 滑点 {slippage:.4f}, 冲击 {market_impact:.4f}) 止损 {stop:.2f} 止盈 {take:.2f}")
    send_telegram(f"开仓 {dir_str} {sym}\n价格: {exec_price:.2f}\n仓位: {size:.4f}", msg_type="trade")
    st.session_state.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage, 'impact': market_impact})

def close_position(symbol: str, exit_price: float, reason: str, close_size: Optional[float] = None):
    sym = symbol.strip()
    pos = st.session_state.positions.get(sym)
    if pos is None:
        return
    if close_size is None:
        close_size = pos.size
    close_size = min(close_size, pos.size)
    volume = 0
    imbalance = 0.0
    if sym in st.session_state.multi_df:
        df = st.session_state.multi_df[sym]['15m']
        volume = df['volume'].iloc[-1] if not df.empty else 0
    vola = 0.02
    if sym in st.session_state.multi_df:
        rets = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-20:]
        vola = np.std(rets) if len(rets) > 5 else 0.02
    if sym in st.session_state.symbol_current_prices:
        imbalance = st.session_state.get('orderbook_imbalance', {}).get(sym, 0.0)
    slippage = advanced_slippage_prediction(exit_price, close_size, volume, vola, imbalance)
    exec_exit = exit_price - slippage if pos.direction == 1 else exit_price + slippage
    pnl = (exec_exit - pos.entry_price) * close_size * pos.direction - exec_exit * close_size * CONFIG.fee_rate * 2
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
        'exit': exec_exit,
        'size': close_size,
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
    # 记录平仓滑点
    st.session_state.slippage_records.append({'time': datetime.now(), 'symbol': sym, 'slippage': slippage, 'impact': (close_size / max(volume,1))**0.5 * vola * exit_price * 0.3})

    update_regime_stats(st.session_state.market_regime, pnl)
    update_consistency_stats(is_backtest=False, slippage=slippage, win=pnl>0)

    if close_size >= pos.size:
        del st.session_state.positions[sym]
    else:
        pos.size -= close_size
        log_execution(f"部分平仓 {sym} {reason} 数量 {close_size:.4f}，剩余 {pos.size:.4f}")

    win = pnl > 0
    RiskManager().update_losses(win)
    log_execution(f"平仓 {sym} {reason} 盈亏 {pnl:.2f} 余额 {st.session_state.account_balance:.2f}")
    send_telegram(f"平仓 {reason}\n盈亏: {pnl:.2f}", msg_type="trade")

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

# ==================== 生成权益曲线截图用于Telegram ====================
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

# ==================== 回测引擎 ====================
def run_backtest(symbols: List[str], data_dicts: Dict[str, Dict[str, pd.DataFrame]], initial_balance: float = 10000) -> Dict[str, Any]:
    """简化回测：对每个品种独立运行，使用15分钟线模拟交易"""
    # 获取所有品种的15分钟数据，并确保时间对齐
    df_dict = {}
    min_start = None
    for sym in symbols:
        df = data_dicts[sym]['15m'].copy()
        df['symbol'] = sym
        df.set_index('timestamp', inplace=True)
        df_dict[sym] = df
        if min_start is None or df.index[0] > min_start:
            min_start = df.index[0]
    # 取所有品种的共同起始时间
    common_start = min_start
    for sym in symbols:
        df_dict[sym] = df_dict[sym][df_dict[sym].index >= common_start]

    # 合并所有品种数据，按时间排序
    combined = pd.concat(df_dict.values(), axis=0).sort_index()
    dates = combined.index.unique()
    
    balance = initial_balance
    positions = {}  # sym -> Position (模拟)
    trade_log = []
    equity_curve = []
    
    risk_mgr = RiskManager()
    engine = SignalEngine()
    
    for dt in dates:
        # 获取当前时刻所有品种的数据切片
        current_data = {}
        for sym in symbols:
            row = df_dict[sym].loc[dt] if dt in df_dict[sym].index else None
            if row is not None:
                current_data[sym] = row
        if not current_data:
            continue
        
        # 更新市场状态（使用第一个品种）
        first_sym = symbols[0]
        # 构造 df_dict 格式用于信号计算（需要多周期数据，但回测简化只使用15m）
        # 为了信号计算，需传入包含多周期的字典，但回测中只有15m可用，可能导致信号偏差
        # 这里简化：仅基于15m计算信号，不启用多周期确认
        # 实际应用中应使用历史多周期数据，但此处为保持简洁，仅用15m模拟
        sim_df_dict = {'15m': df_dict[first_sym].loc[:dt].tail(200)}  # 模拟
        # 计算信号
        direction, prob = engine.calc_signal(sim_df_dict)  # 实际可能不准
        
        # 风险管理检查
        if risk_mgr.check_cooldown() or risk_mgr.check_max_drawdown():
            continue
        
        # 平仓检查
        for sym, pos in list(positions.items()):
            if sym not in current_data:
                continue
            row = current_data[sym]
            should_close, reason, exit_price, close_size = pos.should_close(row['high'], row['low'], dt)
            if should_close:
                # 执行平仓
                pnl = (exit_price - pos.entry_price) * close_size * pos.direction
                balance += pnl
                trade_log.append({
                    'time': dt,
                    'symbol': sym,
                    'direction': '多' if pos.direction==1 else '空',
                    'entry': pos.entry_price,
                    'exit': exit_price,
                    'size': close_size,
                    'pnl': pnl,
                    'reason': reason
                })
                if close_size >= pos.size:
                    del positions[sym]
                else:
                    pos.size -= close_size
        
        # 开仓信号处理
        if direction != 0 and prob >= SignalStrength.WEAK.value and len(positions) < len(symbols):
            # 选择信号最强的品种开仓（简化）
            # 实际应计算每个品种的信号
            for sym in symbols:
                if sym not in positions:
                    # 计算该品种的信号（简化）
                    price = current_data[sym]['close']
                    atr = current_data[sym]['atr'] if not pd.isna(current_data[sym]['atr']) else price*0.01
                    # 计算仓位大小
                    size = risk_mgr.calc_position_size(balance * 0.5, prob, atr, price, np.array([0.01]*20))  # 简化
                    if size > 0:
                        stop_dist = atr * CONFIG.atr_multiplier_base
                        stop = price - stop_dist if direction==1 else price + stop_dist
                        take = price + stop_dist * CONFIG.tp_min_ratio if direction==1 else price - stop_dist * CONFIG.tp_min_ratio
                        pos = Position(
                            symbol=sym,
                            direction=direction,
                            entry_price=price,
                            entry_time=dt,
                            size=size,
                            stop_loss=stop,
                            take_profit=take,
                            initial_atr=atr
                        )
                        positions[sym] = pos
                        break  # 每次只开一个
        
        # 记录权益
        total_value = balance
        for sym, pos in positions.items():
            if sym in current_data:
                total_value += pos.pnl(current_data[sym]['close'])
        equity_curve.append({'time': dt, 'equity': total_value})
    
    return {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'trade_log': trade_log,
        'equity_curve': equity_curve
    }

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
            if 'orderbook_imbalance' not in st.session_state:
                st.session_state.orderbook_imbalance = {}
            st.session_state.orderbook_imbalance[sym] = data.get('orderbook_imbalance', 0.0)

        st.session_state.multi_df = {sym: data['data_dict'] for sym, data in multi_data.items()}
        first_sym = symbols[0]
        st.session_state.fear_greed = multi_data[first_sym]['fear_greed']
        df_first = multi_data[first_sym]['data_dict']
        st.session_state.market_regime = SignalEngine().detect_market_regime(df_first)

        cov = calculate_cov_matrix(symbols, {sym: multi_data[sym]['data_dict'] for sym in symbols}, CONFIG.cov_matrix_window)
        st.session_state.cov_matrix = cov

        fix_data_consistency(symbols)

        if first_sym in multi_data:
            rets = multi_data[first_sym]['data_dict']['15m']['close'].pct_change().dropna().values[-20:]
            volatility = np.std(rets) if len(rets) > 5 else 0
            update_daily_trades_limit(volatility)

        if st.session_state.mode == 'backtest':
            self.render_backtest_panel(symbols, multi_data)
        else:
            self.render_live_panel(symbols, multi_data)

    def render_backtest_panel(self, symbols, multi_data):
        st.subheader("📊 回测面板")
        if st.button("▶️ 开始回测"):
            with st.spinner("回测进行中..."):
                results = run_backtest(symbols, {sym: multi_data[sym]['data_dict'] for sym in symbols}, st.session_state.account_balance)
                st.session_state.backtest_results = results
        if st.session_state.backtest_results:
            res = st.session_state.backtest_results
            st.metric("初始资金", f"{res['initial_balance']:.2f}")
            st.metric("最终资金", f"{res['final_balance']:.2f}")
            st.metric("总收益率", f"{res['total_return']:.2f}%")
            st.metric("交易次数", len(res['trade_log']))
            if res['trade_log']:
                df_trades = pd.DataFrame(res['trade_log'])
                st.dataframe(df_trades)
            if res['equity_curve']:
                df_eq = pd.DataFrame(res['equity_curve'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_eq['time'], y=df_eq['equity'], mode='lines'))
                fig.update_layout(title="回测权益曲线", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

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
                # 使用订单拆分执行
                split_and_execute(sym, dir, size, price, stop, take)

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

        # 构建历史收益率用于VaR/CVaR
        historical_rets = None
        if len(symbols) > 1:
            ret_arrays = []
            for sym in symbols:
                rets = st.session_state.multi_df[sym]['15m']['close'].pct_change().dropna().values[-100:]
                ret_arrays.append(rets)
            min_len = min(len(arr) for arr in ret_arrays)
            # 构建形状为 (n_symbols, min_len) 的矩阵
            hist_rets = np.array([arr[-min_len:] for arr in ret_arrays])
            historical_rets = hist_rets  # 形状 (n_symbols, n_obs)

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
            st.metric("日内交易", f"{st.session_state.daily_trades}/{st.session_state.dynamic_max_daily_trades}")
            var_limit = get_dynamic_var_limit()
            method_name = "历史模拟法" if st.session_state.var_method == "HISTORICAL" else "正态法"
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
    st.set_page_config(page_title="终极量化终端 47.0 · 智能进化版", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 智能进化版 47.0")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北 · 三阶段智能进化 · CVaR · 动态Kelly · 因子相关性 · IC衰减 · 订单拆分 · 冲击成本")

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
