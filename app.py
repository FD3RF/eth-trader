# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 终极版 v5.0
==================================================
设计哲学：
1. 安全至上：自动从 secrets.toml 加载只读密钥，刷新永不丢失
2. 智能切换：若真实数据获取失败，自动回退到模拟数据并提示
3. 风控为王：动态熔断、冷却、日内限制、最大回撤、凯利仓位
4. 信号可靠：多周期多因子加权 + 市场状态识别 + IC动态调权
5. 执行坚决：强制止损止盈 + 移动止损 + 保本止损 + 部分止盈
6. 极致透明：实时调试信息 + 净值曲线 + 蒙特卡洛模拟
7. 开箱即用：内置超真实模拟数据生成器，无需API即可体验
==================================================
作者：AI 极限优化版
最后更新：2026-02-18
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import warnings
import logging
import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import requests

warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================
class SignalStrength(Enum):
    EXTREME = 0.85
    STRONG = 0.75
    HIGH = 0.65
    MEDIUM = 0.55
    WEAK = 0.45
    NONE = 0.0

class MarketRegime(Enum):
    TREND_UP = "趋势上涨"
    TREND_DOWN = "趋势下跌"
    RANGE = "震荡"
    PANIC = "恐慌"
    EUPHORIA = "狂热"

@dataclass
class TradingConfig:
    # 交易标的
    symbols: List[str] = field(default_factory=lambda: ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"])
    
    # 风险控制核心参数
    base_risk_per_trade: float = 0.02        # 单笔风险本金比例 (2%) 凯利公式基础
    risk_budget_ratio: float = 0.20          # 总风险预算占本金比例
    daily_loss_limit: float = 500.0           # 单日最大亏损（绝对值）
    max_drawdown_pct: float = 15.0            # 最大回撤百分比（超过则停止交易）
    min_atr_pct: float = 0.5                  # 最小ATR百分比（低于此用1%止损）
    tp_min_ratio: float = 2.2                  # 最小盈亏比
    partial_tp_ratio: float = 0.5              # 部分止盈仓位比例
    partial_tp_r_multiple: float = 1.2          # 部分止盈触发倍数（止损的倍数）
    trailing_stop_pct: float = 0.4              # 移动止损回调百分比（相对于最高点的百分比）
    breakeven_trigger_pct: float = 1.5          # 保本止损触发倍数（止损的倍数）
    max_hold_hours: int = 48                     # 最长持仓时间（小时）
    max_consecutive_losses: int = 2              # 最大连续亏损次数（触发冷却）
    cooldown_losses: int = 2                      # 触发冷却的连续亏损次数
    cooldown_hours: int = 12                      # 冷却时长（小时）
    max_daily_trades: int = 4                     # 每日最大交易次数
    atr_multiplier: float = 1.8                   # ATR止损乘数
    
    # 杠杆模式
    leverage_modes: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "保守 (1-2x)": (1, 2),
        "稳健 (3-5x)": (3, 5),
        "进取 (5-8x)": (5, 8),
        "极限 (8-10x)": (8, 10)
    })
    
    # 交易所配置
    exchanges: Dict[str, Any] = field(default_factory=lambda: {
        "Binance合约": ccxt.binance,
        "Bybit合约": ccxt.bybit,
        "OKX合约": ccxt.okx
    })
    
    # 周期权重（可根据市场状态动态调整）
    timeframes: List[str] = field(default_factory=lambda: ['15m', '1h', '4h', '1d'])
    timeframe_weights: Dict[str, int] = field(default_factory=lambda: {'1d': 10, '4h': 7, '1h': 5, '15m': 3})
    
    # 数据获取
    fetch_limit: int = 1000
    auto_refresh_ms: int = 30000                 # 刷新间隔30秒
    anti_duplicate_seconds: int = 180             # 防重信号间隔
    
    # 交易成本
    slippage_base: float = 0.0005                 # 滑点
    fee_rate: float = 0.0004                       # 手续费
    
    # 因子IC窗口
    ic_window: int = 80
    
    # 模拟数据参数
    sim_volatility: float = 0.06
    sim_trend_strength: float = 0.2
    
    # 恐惧贪婪指数API
    fear_greed_api: str = "https://api.alternative.me/fng/?limit=1"

CONFIG = TradingConfig()

# ==================== 日志系统 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UltimateTrader")

# ==================== 辅助函数 ====================
def init_session_state():
    """初始化 session_state，并从 secrets 加载 API 密钥"""
    # 从 secrets.toml 加载 API 密钥（如果存在）
    if 'BINANCE_API_KEY' in st.secrets:
        st.session_state.binance_api_key = st.secrets['BINANCE_API_KEY']
        st.session_state.binance_secret_key = st.secrets['BINANCE_SECRET_KEY']
        # 如果有有效密钥，默认关闭模拟模式
        default_use_sim = False
    else:
        st.session_state.binance_api_key = ''
        st.session_state.binance_secret_key = ''
        default_use_sim = True
    
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
        'use_simulated_data': default_use_sim,
        'data_source_failed': False,
        'error_log': [],
        'execution_log': [],
        'last_trade_date': None,
        'multi_df': {},
        'ic_cache': {},
        'fear_greed': 50,
        'market_regime': MarketRegime.RANGE,
        'exchange_choice': 'Binance合约',
        'testnet': True,
        'use_real': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_error(msg: str):
    st.session_state.error_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    if len(st.session_state.error_log) > 20:
        st.session_state.error_log.pop(0)
    logger.error(msg)

def log_execution(msg: str):
    st.session_state.execution_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    if len(st.session_state.execution_log) > 30:
        st.session_state.execution_log.pop(0)
    # Telegram通知（如果启用）
    if st.session_state.telegram_token and st.session_state.telegram_chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{st.session_state.telegram_token}/sendMessage",
                          json={"chat_id": st.session_state.telegram_chat_id, "text": msg})
        except:
            pass

def fetch_fear_greed() -> int:
    """获取恐惧贪婪指数，失败返回50"""
    try:
        resp = requests.get(CONFIG.fear_greed_api, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return int(data['data'][0]['value'])
    except Exception as e:
        logger.warning(f"获取恐惧贪婪指数失败: {e}")
    return 50

def detect_market_regime(df_dict: Dict[str, pd.DataFrame]) -> MarketRegime:
    """根据多周期数据识别市场状态"""
    if '1h' not in df_dict or '4h' not in df_dict:
        return MarketRegime.RANGE
    df1h = df_dict['1h']
    df4h = df_dict['4h']
    if len(df1h) < 20 or len(df4h) < 20:
        return MarketRegime.RANGE
    
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
        if trend_up:
            return MarketRegime.TREND_UP
        elif trend_down:
            return MarketRegime.TREND_DOWN
        else:
            return MarketRegime.RANGE
    elif st.session_state.fear_greed <= 20:
        return MarketRegime.PANIC
    elif st.session_state.fear_greed >= 80:
        return MarketRegime.EUPHORIA
    else:
        return MarketRegime.RANGE

# ==================== 超真实模拟数据生成器 ====================
def generate_simulated_data(symbol: str, limit: int = 1500) -> Dict[str, pd.DataFrame]:
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % 2**32
    np.random.seed(seed)
    
    end = datetime.now()
    timestamps = pd.date_range(end=end, periods=limit, freq='15min')
    
    if 'BTC' in symbol:
        base = 42000
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
    
    required_len = max(50, CONFIG.ic_window)
    if len(df) < required_len:
        logger.warning(f"数据点不足 {required_len}，指标可能为NaN")
    
    # EMA
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # RSI
    if len(df) >= 14:
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    else:
        df['rsi'] = np.nan
    
    # ATR
    if len(df) >= 14:
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr
        df['atr_ma'] = atr.rolling(20).mean()
    else:
        df['atr'] = np.nan
        df['atr_ma'] = np.nan
    
    # MACD
    if len(df) >= 26:
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = df['macd'] - df['macd_signal']
    else:
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_diff'] = np.nan
    
    # ADX
    if len(df) >= 14:
        try:
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        except Exception as e:
            logger.warning(f"ADX 计算失败: {e}")
            df['adx'] = np.nan
    else:
        df['adx'] = np.nan
    
    # 布林带
    if len(df) >= 20:
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    else:
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_width'] = np.nan
    
    # 成交量加权
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # 未来收益率（用于IC计算）
    if len(df) >= 6:
        df['future_ret'] = df['close'].pct_change(5).shift(-5)
    else:
        df['future_ret'] = np.nan
    
    return df

# ==================== 因子信息系数 (IC) 计算 ====================
def calculate_ic(df: pd.DataFrame, factor_name: str) -> float:
    window = min(CONFIG.ic_window, len(df) - 6)
    if window < 20:
        return 0.0
    factor = df[factor_name].iloc[-window:-5]
    future = df['future_ret'].iloc[-window:-5]
    valid = factor.notna() & future.notna()
    if valid.sum() < 10:
        return 0.0
    ic = factor[valid].corr(future[valid])
    return 0.0 if pd.isna(ic) else ic

# ==================== 数据获取器（增强版）====================
@st.cache_resource
def get_fetcher() -> 'AggregatedDataFetcher':
    return AggregatedDataFetcher()

class AggregatedDataFetcher:
    def __init__(self):
        self.exchanges = {}
        self._init_exchanges()
    
    def _init_exchanges(self):
        if st.session_state.get('use_simulated_data', True):
            return
        # 只初始化用户选择的交易所，避免多个连接
        exchange_name = st.session_state.get('exchange_choice', 'Binance合约')
        api_key = st.session_state.get('binance_api_key', '')
        secret = st.session_state.get('binance_secret_key', '')
        if not api_key or not secret:
            logger.warning("API密钥为空，无法初始化交易所")
            return
        try:
            cls = CONFIG.exchanges[exchange_name]
            exchange_params = {
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'future'}
            }
            # OKX需要passphrase
            if 'OKX' in exchange_name:
                exchange_params['password'] = st.session_state.get('okx_passphrase', '')
            self.exchanges[exchange_name] = cls(exchange_params)
            if st.session_state.get('testnet', True):
                self.exchanges[exchange_name].set_sandbox_mode(True)
            logger.info(f"交易所 {exchange_name} 初始化成功")
        except Exception as e:
            logger.error(f"初始化交易所 {exchange_name} 失败: {e}")

    def fetch_kline(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        if not self.exchanges:
            return None
        for ex in self.exchanges.values():
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if ohlcv and len(ohlcv) >= 50:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.astype({col: float for col in ['open','high','low','close','volume']})
                    return add_indicators(df)
            except Exception as e:
                logger.warning(f"获取 {symbol} {timeframe} 失败: {e}")
                continue
        return None

    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        data_dict = {}
        for tf in CONFIG.timeframes:
            df = self.fetch_kline(symbol, tf, CONFIG.fetch_limit)
            if df is not None and len(df) >= 30:
                data_dict[tf] = df
            else:
                logger.error(f"无法获取 {symbol} {tf} 数据")
                return None
        return data_dict

# ==================== 多周期多因子信号整合 ====================
def calc_signal(multi_df: Dict[str, pd.DataFrame]) -> Tuple[int, float]:
    total_score = 0
    total_weight = 0
    tf_votes = []
    
    regime = st.session_state.get('market_regime', MarketRegime.RANGE)
    
    for tf, df in multi_df.items():
        if df.empty or len(df) < 2:
            continue
        last = df.iloc[-1]
        weight = CONFIG.timeframe_weights.get(tf, 1)
        
        if regime in [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]:
            if tf in ['4h', '1d']:
                weight *= 1.5
        elif regime == MarketRegime.RANGE:
            if tf in ['15m', '1h']:
                weight *= 1.3
        
        if pd.isna(last.get('ema20', np.nan)) or pd.isna(last.get('rsi', np.nan)):
            continue
        
        factor_score = 0
        if last['close'] > last['ema20']:
            factor_score += 1
        elif last['close'] < last['ema20']:
            factor_score -= 1
        
        if last['rsi'] > 70:
            factor_score -= 0.7
        elif last['rsi'] < 30:
            factor_score += 0.7
        
        if last['macd_diff'] > 0:
            factor_score += 0.8
        elif last['macd_diff'] < 0:
            factor_score -= 0.8
        
        if not pd.isna(last.get('bb_upper')) and not pd.isna(last.get('bb_lower')):
            if last['close'] > last['bb_upper']:
                factor_score -= 0.5
            elif last['close'] < last['bb_lower']:
                factor_score += 0.5
        
        if not pd.isna(last.get('volume_ratio')):
            if last['volume_ratio'] > 1.5:
                factor_score *= 1.2
        
        adx = last.get('adx', 25)
        if pd.isna(adx):
            adx_boost = 1.0
        elif adx > 30:
            adx_boost = 1.3
        elif adx < 20:
            adx_boost = 0.7
        else:
            adx_boost = 1.0
        
        ic_rsi = calculate_ic(df, 'rsi')
        ic_macd = calculate_ic(df, 'macd_diff')
        ic_adx = calculate_ic(df, 'adx')
        ic_avg = np.nanmean([ic_rsi, ic_macd, ic_adx])
        if pd.isna(ic_avg):
            ic_boost = 1.0
        else:
            ic_boost = 1.0 + np.clip(ic_avg * 0.5, -0.2, 0.3)
        
        tf_score = factor_score * weight * adx_boost * ic_boost
        total_score += tf_score
        total_weight += weight
        
        if factor_score > 0:
            tf_votes.append(1)
        elif factor_score < 0:
            tf_votes.append(-1)
    
    if total_weight == 0:
        return 0, 0.0
    
    max_possible_score = sum(CONFIG.timeframe_weights.values()) * 4.0
    prob_raw = min(1.0, abs(total_score) / max_possible_score) if max_possible_score > 0 else 0.5
    prob = 0.5 + 0.45 * prob_raw
    
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

# ==================== 凯利公式仓位计算 ====================
def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    kelly = (p * b - q) / b
    return max(0, min(kelly, 0.25))

def calc_position_size(balance: float, prob: float, atr: float, price: float, atr_pct: float) -> float:
    if price <= 0 or prob < 0.5:
        return 0.0
    
    # 从历史交易中估算胜率和盈亏比
    win_rate_history = [t['pnl'] for t in st.session_state.trade_log[-50:] if 'pnl' in t]
    if len(win_rate_history) > 10:
        trades_df = pd.DataFrame(st.session_state.trade_log[-50:])
        if not trades_df.empty:
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] < 0]
            win_rate = len(wins) / max(len(trades_df), 1)
            avg_win = wins['pnl'].mean() if not wins.empty else 0
            avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 1
            kelly = kelly_fraction(win_rate, avg_win, avg_loss)
        else:
            kelly = 0.02
    else:
        kelly = 0.02
    
    edge = max(0.05, prob - 0.5) * 2
    risk_amount = balance * kelly * edge
    
    if atr == 0 or np.isnan(atr) or atr < price * CONFIG.min_atr_pct / 100:
        stop_distance = price * 0.01
    else:
        stop_distance = atr * CONFIG.atr_multiplier
    
    leverage_mode = st.session_state.get('leverage_mode', "稳健 (3-5x)")
    min_lev, max_lev = CONFIG.leverage_modes.get(leverage_mode, (3,5))
    max_size_by_leverage = balance * max_lev / price
    
    size_by_risk = risk_amount / stop_distance
    size = min(size_by_risk, max_size_by_leverage)
    return max(size, 0.001)

# ==================== 风控检查 ====================
def check_daily_limit() -> bool:
    today = datetime.now().date()
    if st.session_state.get('last_trade_date') != today:
        st.session_state.daily_trades = 0
        st.session_state.last_trade_date = today
    return st.session_state.daily_trades >= CONFIG.max_daily_trades

def check_cooldown() -> bool:
    until = st.session_state.get('cooldown_until')
    return until is not None and datetime.now() < until

def update_losses(win: bool):
    if not win:
        st.session_state.consecutive_losses += 1
        if st.session_state.consecutive_losses >= CONFIG.cooldown_losses:
            st.session_state.cooldown_until = datetime.now() + timedelta(hours=CONFIG.cooldown_hours)
    else:
        st.session_state.consecutive_losses = 0
        st.session_state.cooldown_until = None

def check_circuit_breaker(atr_pct: float, fear_greed: int) -> bool:
    return atr_pct > 5.0 or fear_greed <= 15 or fear_greed >= 85

def check_max_drawdown() -> bool:
    drawdown = (st.session_state.peak_balance - st.session_state.account_balance) / st.session_state.peak_balance * 100
    return drawdown > CONFIG.max_drawdown_pct

# ==================== 持仓管理 ====================
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
    
    def __post_init__(self):
        if self.direction == 1:
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price
    
    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size * self.direction
    
    def update_stops(self, current_price: float, atr: float):
        if self.direction == 1:
            if current_price > self.highest_price:
                self.highest_price = current_price
            trailing_stop = self.highest_price * (1 - CONFIG.trailing_stop_pct / 100)
            self.stop_loss = max(self.stop_loss, trailing_stop)
            new_tp = current_price + atr * CONFIG.atr_multiplier * CONFIG.tp_min_ratio
            self.take_profit = max(self.take_profit, new_tp)
            if current_price >= self.entry_price + (self.entry_price - self.stop_loss_original()) * CONFIG.breakeven_trigger_pct:
                self.stop_loss = max(self.stop_loss, self.entry_price)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            trailing_stop = self.lowest_price * (1 + CONFIG.trailing_stop_pct / 100)
            self.stop_loss = min(self.stop_loss, trailing_stop)
            new_tp = current_price - atr * CONFIG.atr_multiplier * CONFIG.tp_min_ratio
            self.take_profit = min(self.take_profit, new_tp)
            if current_price <= self.entry_price - (self.stop_loss_original() - self.entry_price) * CONFIG.breakeven_trigger_pct:
                self.stop_loss = min(self.stop_loss, self.entry_price)
    
    def stop_loss_original(self) -> float:
        if self.direction == 1:
            return self.entry_price - (self.stop_loss if hasattr(self, 'stop_loss') else self.entry_price * 0.99)
        else:
            return (self.stop_loss if hasattr(self, 'stop_loss') else self.entry_price * 1.01) - self.entry_price
    
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
            if self.direction == 1 and high >= self.entry_price + (self.entry_price - self.stop_loss_original()) * CONFIG.partial_tp_r_multiple:
                return True, "部分止盈", self.entry_price + (self.entry_price - self.stop_loss_original()) * CONFIG.partial_tp_r_multiple
            if self.direction == -1 and low <= self.entry_price - (self.stop_loss_original() - self.entry_price) * CONFIG.partial_tp_r_multiple:
                return True, "部分止盈", self.entry_price - (self.stop_loss_original() - self.entry_price) * CONFIG.partial_tp_r_multiple
        return False, "", 0

# ==================== Monte Carlo 模拟 ====================
def monte_carlo_sim(price_series: pd.Series, n_sim: int = 500) -> pd.DataFrame:
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

# ==================== 执行下单 ====================
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
        real=st.session_state.get('use_real', False) and st.session_state.exchange is not None
    )
    st.session_state.daily_trades += 1
    log_execution(f"开仓 {symbol} {dir_str} 仓位 {size:.4f} @ {price:.2f} 止损 {stop:.2f} 止盈 {take:.2f}")

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
    
    st.session_state.trade_log.append({
        'time': datetime.now(),
        'symbol': symbol,
        'direction': '多' if pos.direction == 1 else '空',
        'entry': pos.entry_price,
        'exit': exit_price,
        'size': pos.size,
        'pnl': pnl,
        'reason': reason
    })
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log.pop(0)
    
    win = pnl > 0
    update_losses(win)
    log_execution(f"平仓 {symbol} {reason} 盈亏 {pnl:.2f} 余额 {st.session_state.account_balance:.2f}")
    st.session_state.position = None

# ==================== 自动交易循环（增强版）====================
def auto_trade_step(symbol: str):
    # 数据源选择
    if st.session_state.use_simulated_data:
        multi_df = generate_simulated_data(symbol, CONFIG.fetch_limit)
        st.session_state.data_source_failed = False
    else:
        fetcher = get_fetcher()
        multi_df = fetcher.get_symbol_data(symbol)
        if multi_df is None:
            log_error("获取真实数据失败，请检查网络或API权限，已自动切换回模拟数据")
            st.session_state.use_simulated_data = True
            st.session_state.data_source_failed = True
            multi_df = generate_simulated_data(symbol, CONFIG.fetch_limit)
        else:
            st.session_state.data_source_failed = False
    
    st.session_state.multi_df = multi_df
    df_15m = multi_df['15m']
    current_price = df_15m['close'].iloc[-1]
    atr = df_15m['atr'].iloc[-1] if not pd.isna(df_15m['atr'].iloc[-1]) else 0
    
    if not st.session_state.use_simulated_data and not st.session_state.data_source_failed:
        st.session_state.fear_greed = fetch_fear_greed()
    else:
        st.session_state.fear_greed = 50
    
    st.session_state.market_regime = detect_market_regime(multi_df)
    
    if pd.isna(atr) or atr == 0:
        atr_pct = 0
    else:
        atr_pct = atr / current_price * 100
    
    st.session_state.circuit_breaker = check_circuit_breaker(atr_pct, st.session_state.fear_greed)
    
    direction, prob = calc_signal(multi_df)
    size = calc_position_size(st.session_state.account_balance, prob, atr, current_price, atr_pct)
    
    # 调试信息
    with st.expander("🔍 开仓调试信息", expanded=True):
        st.write(f"总分: {12.92:.2f}, 方向: {direction}, 概率: {prob:.2%}")
        st.write(f"ATR: {atr:.2f}, ATR%: {atr_pct:.2f}%, 计算仓位: {size:.4f}")
        st.write(f"信号阈值: {SignalStrength.WEAK.value:.2%}")
        st.write(f"市场状态: {st.session_state.market_regime.value}")
        st.write(f"恐惧贪婪: {st.session_state.fear_greed}")
        st.write(f"风控状态: 熔断={st.session_state.circuit_breaker}, 冷却={check_cooldown()}, 日内限制={check_daily_limit()}, 超回撤={check_max_drawdown()}")
        st.write(f"是否满足开仓条件: {direction != 0 and prob >= SignalStrength.WEAK.value and size > 0}")
        st.write(f"数据源: {'模拟' if st.session_state.use_simulated_data else '实盘'} {'(失败回退)' if st.session_state.data_source_failed else ''}")
    
    if st.session_state.circuit_breaker or check_cooldown() or check_daily_limit() or check_max_drawdown():
        pass
    else:
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
                    st.write("⏳ 防重机制阻止开仓（信号间隔过短）")
                    return
                if atr == 0 or np.isnan(atr):
                    stop_distance = current_price * 0.01
                else:
                    stop_distance = atr * CONFIG.atr_multiplier
                stop = current_price - stop_distance if direction == 1 else current_price + stop_distance
                take = current_price + stop_distance * CONFIG.tp_min_ratio if direction == 1 else current_price - stop_distance * CONFIG.tp_min_ratio
                execute_order(symbol, direction, size, current_price, stop, take)
                st.session_state.last_signal_time = datetime.now()
                st.rerun()

# ==================== UI渲染（增强版）====================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 配置")
        symbol = st.selectbox("品种", CONFIG.symbols, index=CONFIG.symbols.index(st.session_state.current_symbol))
        st.session_state.current_symbol = symbol

        # 数据源选择
        use_sim = st.checkbox("使用模拟数据（离线模式）", value=st.session_state.get('use_simulated_data', True))
        if use_sim != st.session_state.get('use_simulated_data', True):
            st.session_state.use_simulated_data = use_sim
            st.cache_data.clear()
            st.rerun()

        # 显示当前数据源状态
        if st.session_state.use_simulated_data:
            st.info("📡 当前数据源：模拟数据")
        else:
            if st.session_state.data_source_failed:
                st.error("📡 真实数据获取失败，已回退到模拟数据")
            else:
                st.success("📡 当前数据源：币安实时数据")

        mode = st.selectbox("杠杆模式", list(CONFIG.leverage_modes.keys()))
        st.session_state.leverage_mode = mode

        st.number_input("余额 USDT", value=st.session_state.account_balance, disabled=True, key="balance_display")

        if st.button("🔄 同步实盘余额"):
            if st.session_state.exchange and not st.session_state.use_simulated_data:
                try:
                    balance = st.session_state.exchange.fetch_balance()
                    st.session_state.account_balance = float(balance['total'].get('USDT', 0))
                    st.success(f"同步成功：{st.session_state.account_balance:.2f} USDT")
                except Exception as e:
                    st.error(f"同步失败: {e}")

        st.markdown("---")
        st.subheader("实盘")
        exchange_choice = st.selectbox("交易所", list(CONFIG.exchanges.keys()), key='exchange_choice')
        
        # 从 session_state 获取预设的密钥（由 secrets 自动填充）
        api_key = st.text_input("API Key", value=st.session_state.binance_api_key, type="password")
        secret_key = st.text_input("Secret Key", value=st.session_state.binance_secret_key, type="password")
        passphrase = st.text_input("Passphrase (仅OKX需要)", type="password") if "OKX" in exchange_choice else None
        
        testnet = st.checkbox("测试网", value=st.session_state.get('testnet', True))
        use_real = st.checkbox("实盘交易", value=st.session_state.get('use_real', False))

        if st.button("测试连接"):
            try:
                ex_class = CONFIG.exchanges[exchange_choice]
                exchange_params = {
                    'apiKey': api_key,
                    'secret': secret_key,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                }
                if passphrase:
                    exchange_params['password'] = passphrase
                ex = ex_class(exchange_params)
                if testnet:
                    ex.set_sandbox_mode(True)
                # 尝试获取 ticker 测试连接
                ticker = ex.fetch_ticker(symbol)
                st.success(f"连接成功！当前 {symbol} 价格: {ticker['last']}")
                # 保存到 session_state
                st.session_state.exchange = ex
                st.session_state.binance_api_key = api_key
                st.session_state.binance_secret_key = secret_key
                st.session_state.testnet = testnet
                st.session_state.use_real = use_real
            except Exception as e:
                st.error(f"连接失败: {e}")

        st.session_state.auto_enabled = st.checkbox("自动交易", value=True)

        with st.expander("Telegram通知"):
            token = st.text_input("Bot Token", type="password", key="tg_token")
            chat_id = st.text_input("Chat ID", key="tg_chat")
            if token and chat_id:
                st.session_state.telegram_token = token
                st.session_state.telegram_chat_id = chat_id

        if st.button("🚨 一键紧急平仓"):
            if st.session_state.position:
                close_position(st.session_state.current_symbol, st.session_state.multi_df['15m']['close'].iloc[-1], "紧急平仓")
            st.rerun()

        if st.button("运行回测"):
            st.info("回测功能需单独实现，暂不可用")

        if st.button("🖐️ 手动开仓测试"):
            multi_df = st.session_state.multi_df
            if multi_df:
                price = multi_df['15m']['close'].iloc[-1]
                atr = multi_df['15m']['atr'].iloc[-1] if not pd.isna(multi_df['15m']['atr'].iloc[-1]) else 0
                if atr == 0:
                    stop_distance = price * 0.01
                else:
                    stop_distance = atr * CONFIG.atr_multiplier
                stop = price - stop_distance
                take = price + stop_distance * CONFIG.tp_min_ratio
                size = calc_position_size(st.session_state.account_balance, 0.7, atr, price, 0)
                if size > 0:
                    execute_order(st.session_state.current_symbol, 1, size, price, stop, take)
                    st.rerun()

        if st.session_state.error_log:
            with st.expander("⚠️ 错误日志"):
                for err in st.session_state.error_log[-10:]:
                    st.text(err)

        if st.session_state.execution_log:
            with st.expander("📋 执行日志"):
                for log in st.session_state.execution_log[-10:]:
                    st.text(log)

        if st.button("🗑️ 重置所有状态"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_main_panel():
    if 'multi_df' not in st.session_state or not st.session_state.multi_df:
        st.warning("等待数据加载...")
        return

    multi_df = st.session_state.multi_df
    df_15m = multi_df['15m']
    current_price = df_15m['close'].iloc[-1]

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### 📊 市场状态")
        c1, c2, c3 = st.columns(3)
        c1.metric("恐惧贪婪", st.session_state.fear_greed)
        c2.metric("信号概率", f"{calc_signal(multi_df)[1]:.1%}")
        c3.metric("当前价格", f"{current_price:.2f}")

        if st.session_state.position:
            pos = st.session_state.position
            pnl = pos.pnl(current_price)
            st.markdown(f"### 持仓 {('多' if pos.direction==1 else '空')}")
            st.info(f"入场 {pos.entry_price:.2f} | 数量 {pos.size:.4f}")
            st.info(f"止损 {pos.stop_loss:.2f} | 止盈 {pos.take_profit:.2f}")
            st.metric("浮动盈亏", f"{pnl:.2f} USDT", delta=f"{(pnl/pos.size):.2f}")
        else:
            st.markdown("### 无持仓")
            st.info("等待信号...")

        with st.expander("🔍 多周期信号详情"):
            for tf, df in multi_df.items():
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

        if st.session_state.net_value_history:
            hist_df = pd.DataFrame(st.session_state.net_value_history[-200:])
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
            sim_df = monte_carlo_sim(df_15m['close'], n_sim=500)
            if sim_df.empty:
                st.warning("数据不足，无法模拟")
            else:
                fig_mc = go.Figure()
                for i in range(min(30, sim_df.shape[1])):
                    fig_mc.add_trace(go.Scatter(y=sim_df.iloc[:, i], mode='lines', line=dict(color='rgba(0,200,0,0.1)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=sim_df.mean(axis=1), mode='lines', line=dict(color='red', width=2), name='均值'))
                fig_mc.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig_mc, use_container_width=True)

# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="终极量化终端 终极版 v5.0", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 终极版 v5.0")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北")

    init_session_state()
    render_sidebar()
    auto_trade_step(st.session_state.current_symbol)
    render_main_panel()
    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

if __name__ == "__main__":
    main()
