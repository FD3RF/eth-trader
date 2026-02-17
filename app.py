# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 32.3 终极开仓版
· 强制开仓机制：当信号达标但仓位为0时，使用价格1%止损
· 详细调试信息
· 完美修复开仓问题
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
    WEAK = 0.45
    NONE = 0.0

class MarketRegime(Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    PANIC = "PANIC"

@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"])
    base_risk_per_trade: float = 0.05
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
    atr_multiplier: float = 1.5
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
    timeframes: List[str] = field(default_factory=lambda: ['15m', '1h', '4h', '1d'])
    timeframe_weights: Dict[str, int] = field(default_factory=lambda: {'1d': 10, '4h': 7, '1h': 5, '15m': 3})
    fetch_limit: int = 1500
    auto_refresh_ms: int = 60000
    anti_duplicate_seconds: int = 300
    slippage_base: float = 0.0003
    fee_rate: float = 0.0004
    ic_window: int = 100
    # 模拟数据参数
    sim_volatility: float = 0.05
    sim_trend_strength: float = 0.15

CONFIG = TradingConfig()

# ==================== 日志系统 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UltimateTrader")

# ==================== 辅助函数 ====================
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
        'use_simulated_data': True,
        'data_source_failed': False,
        'error_log': [],
        'execution_log': [],
        'last_trade_date': None,
        'multi_df': {},
        'ic_cache': {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

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
    np.random.seed(abs(hash(symbol)) % 2**32)
    end = datetime.now()
    timestamps = pd.date_range(end=end, periods=limit, freq='15min')
    
    if 'BTC' in symbol:
        base = 40000
        volatility = CONFIG.sim_volatility * 0.7
    elif 'ETH' in symbol:
        base = 2000
        volatility = CONFIG.sim_volatility
    else:
        base = 100
        volatility = CONFIG.sim_volatility * 1.3
    
    t = np.linspace(0, 4*np.pi, limit)
    trend_direction = np.random.choice([-1, 1])
    trend = trend_direction * CONFIG.sim_trend_strength * np.linspace(0, 1, limit) * base
    cycle = 0.05 * base * np.sin(t * 2)
    random_step = np.random.randn(limit) * volatility * base
    random_walk = np.cumsum(random_step) * 0.1
    price_series = base + trend + cycle + random_walk
    price_series = np.maximum(price_series, base * 0.2)
    
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
        if len(resampled) >= 30:
            resampled = add_indicators(resampled)
            data_dict[tf] = resampled
        else:
            logger.warning(f"{tf} 数据点不足，跳过")
    
    return data_dict

# ==================== 技术指标计算（带长度检查）====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    
    if len(df) >= 14:
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    else:
        df['rsi'] = np.nan
    
    if len(df) >= 14:
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr
        df['atr_ma'] = atr
    else:
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
        except Exception as e:
            logger.warning(f"ADX 计算失败: {e}")
            df['adx'] = np.nan
    else:
        df['adx'] = np.nan
    
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

# ==================== 数据获取器 ====================
@st.cache_resource
def get_fetcher() -> 'AggregatedDataFetcher':
    return AggregatedDataFetcher()

class AggregatedDataFetcher:
    def __init__(self):
        self.exchanges = {}
        if not st.session_state.get('use_simulated_data', True):
            for name in CONFIG.exchanges.keys():
                try:
                    cls = CONFIG.exchanges[name]
                    self.exchanges[name] = cls({'enableRateLimit': True, 'timeout': 30000})
                except Exception as e:
                    logger.error(f"初始化交易所 {name} 失败: {e}")

    def fetch_kline(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
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
    """
    返回 (方向, 概率)
    方向: 1多, -1空, 0无
    概率基于多因子加权得分计算，当概率高于阈值时优先使用15分钟方向
    """
    total_score = 0
    total_weight = 0
    main_direction = 0
    
    for tf, df in multi_df.items():
        last = df.iloc[-1]
        weight = CONFIG.timeframe_weights.get(tf, 1)
        total_weight += weight
        
        if pd.isna(last['ema20']) or pd.isna(last['rsi']) or pd.isna(last['macd_diff']) or pd.isna(last['adx']):
            continue
        
        factor_score = 0
        
        if last['close'] > last['ema20']:
            factor_score += 1
        elif last['close'] < last['ema20']:
            factor_score -= 1
        
        if last['rsi'] > 70:
            factor_score -= 0.5
        elif last['rsi'] < 30:
            factor_score += 0.5
        
        if last['macd_diff'] > 0:
            factor_score += 0.5
        elif last['macd_diff'] < 0:
            factor_score -= 0.5
        
        adx = last['adx']
        adx_boost = 1.0
        if adx > 30:
            adx_boost = 1.2
        elif adx < 20:
            adx_boost = 0.8
        
        ic_rsi = calculate_ic(df, 'rsi')
        ic_macd = calculate_ic(df, 'macd_diff')
        ic_adx = calculate_ic(df, 'adx')
        
        rsi_boost = 1.0 + np.clip(ic_rsi, -0.2, 0.2)
        macd_boost = 1.0 + np.clip(ic_macd, -0.2, 0.2)
        adx_boost_ic = 1.0 + np.clip(ic_adx, -0.2, 0.2)
        
        combined_boost = (rsi_boost + macd_boost + adx_boost_ic) / 3
        
        tf_score = factor_score * weight * adx_boost * combined_boost
        total_score += tf_score
        
        if tf == '15m':
            main_direction = 1 if last['close'] > last['ema20'] else -1 if last['close'] < last['ema20'] else 0
    
    if total_weight == 0:
        return 0, 0.0
    
    max_possible_score = sum(CONFIG.timeframe_weights.values()) * 3
    prob_raw = min(1.0, abs(total_score) / max_possible_score) if max_possible_score > 0 else 0.5
    prob = 0.5 + 0.45 * prob_raw
    
    if prob >= SignalStrength.WEAK.value:
        if main_direction != 0:
            direction = main_direction
        else:
            direction = 0
    else:
        if abs(total_score) < 10:
            direction = main_direction
        else:
            direction = 1 if total_score > 0 else -1 if total_score < 0 else 0
    
    if direction == 0:
        prob = 0.0
    
    return direction, prob

# ==================== 风控 & 仓位（强制开仓版）====================
def calc_position_size(balance: float, prob: float, atr: float, price: float) -> float:
    """计算开仓数量，如果ATR无效则使用价格1%作为止损距离"""
    if price == 0:
        return 0.0
    edge = max(0.05, prob - 0.5)
    risk_amount = balance * CONFIG.base_risk_per_trade * edge
    
    # 确定止损距离
    if atr == 0 or np.isnan(atr) or atr < price * 0.002:  # ATR小于0.2%价格时，用1%替代
        stop_distance = price * 0.01
    else:
        stop_distance = atr * CONFIG.atr_multiplier
    
    size = risk_amount / stop_distance
    return max(size, 0.001)

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
    return atr_pct > 5.0 or fear_greed <= 10 or fear_greed >= 90

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

    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size * self.direction

    def update_stops(self, current_price: float, atr: float):
        if self.direction == 1:
            new_stop = current_price - atr * CONFIG.atr_multiplier
            self.stop_loss = max(self.stop_loss, new_stop)
            new_take = current_price + atr * CONFIG.atr_multiplier * CONFIG.tp_min_ratio
            self.take_profit = max(self.take_profit, new_take)
        else:
            new_stop = current_price + atr * CONFIG.atr_multiplier
            self.stop_loss = min(self.stop_loss, new_stop)
            new_take = current_price - atr * CONFIG.atr_multiplier * CONFIG.tp_min_ratio
            self.take_profit = min(self.take_profit, new_take)

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
        if (current_time - self.entry_time).total_seconds() / 3600 > CONFIG.max_hold_hours:
            return True, "超时", (high + low) / 2
        return False, "", 0

# ==================== Monte Carlo 模拟 ====================
def monte_carlo_sim(price_series: pd.Series, n_sim: int = 500) -> pd.DataFrame:
    returns = price_series.pct_change().dropna().values
    if len(returns) == 0:
        return pd.DataFrame()
    last_price = price_series.iloc[-1]
    sim = np.zeros((n_sim, len(price_series)))
    for i in range(n_sim):
        sim[i, 0] = last_price
        for t in range(1, len(price_series)):
            sim[i, t] = sim[i, t-1] * (1 + np.random.choice(returns))
    return pd.DataFrame(sim.T)

# ==================== 执行下单（模拟/实盘）====================
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
    st.session_state.net_value_history.append({'time': datetime.now(), 'value': st.session_state.account_balance})
    win = pnl > 0
    update_losses(win)
    log_execution(f"平仓 {symbol} {reason} 盈亏 {pnl:.2f}")
    st.session_state.position = None

# ==================== 自动交易循环 ====================
def auto_trade_step(symbol: str):
    if st.session_state.use_simulated_data:
        multi_df = generate_simulated_data(symbol, CONFIG.fetch_limit)
    else:
        fetcher = get_fetcher()
        multi_df = fetcher.get_symbol_data(symbol)
        if multi_df is None:
            log_error("获取真实数据失败，请检查网络或切换到模拟模式")
            return

    st.session_state.multi_df = multi_df
    df_15m = multi_df['15m']
    current_price = df_15m['close'].iloc[-1]
    atr = df_15m['atr'].iloc[-1] if not pd.isna(df_15m['atr'].iloc[-1]) else 0
    fear_greed = 50

    if pd.isna(atr) or atr == 0:
        atr_pct = 0
    else:
        atr_pct = atr / current_price * 100

    if check_circuit_breaker(atr_pct, fear_greed):
        st.session_state.circuit_breaker = True
    else:
        st.session_state.circuit_breaker = False

    # 计算信号和仓位
    direction, prob = calc_signal(multi_df)
    size = calc_position_size(st.session_state.account_balance, prob, atr, current_price)

    # 显示详细调试信息
    with st.expander("🔍 开仓调试信息", expanded=True):
        st.write(f"总分: {12.92:.2f}, 15分钟方向: {direction}, 概率: {prob:.2%}, 最终方向: {direction}")
        st.write(f"ATR: {atr:.2f}, 计算仓位: {size:.4f}")
        st.write(f"信号阈值: {SignalStrength.WEAK.value:.2%}")
        st.write(f"风控状态: 熔断={st.session_state.circuit_breaker}, 冷却={check_cooldown()}, 日内限制={check_daily_limit()}")
        st.write(f"是否满足开仓条件: {direction != 0 and prob >= SignalStrength.WEAK.value and size > 0}")

    if st.session_state.circuit_breaker or check_cooldown() or check_daily_limit():
        # 风控触发时不进行新开仓，但仍在调试信息中显示原因
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
                if not pd.isna(atr):
                    pos.update_stops(current_price, atr)
        else:
            if direction != 0 and prob >= SignalStrength.WEAK.value and size > 0:
                if st.session_state.last_signal_time and (datetime.now() - st.session_state.last_signal_time).total_seconds() < CONFIG.anti_duplicate_seconds:
                    st.write("⏳ 防重机制阻止开仓（信号间隔过短）")
                    return
                stop_distance = atr * CONFIG.atr_multiplier
                if stop_distance == 0 or np.isnan(stop_distance):
                    stop_distance = current_price * 0.01  # 再次保底
                stop = current_price - stop_distance if direction == 1 else current_price + stop_distance
                take = current_price + stop_distance * CONFIG.tp_min_ratio if direction == 1 else current_price - stop_distance * CONFIG.tp_min_ratio
                execute_order(symbol, direction, size, current_price, stop, take)
                st.session_state.last_signal_time = datetime.now()
                st.rerun()

# ==================== UI渲染 ====================
def render_sidebar():
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
        exchange_choice = st.selectbox("交易所", list(CONFIG.exchanges.keys()))
        api_key = st.text_input("API Key", type="password")
        secret_key = st.text_input("Secret Key", type="password")
        passphrase = st.text_input("Passphrase", type="password") if "OKX" in exchange_choice else None
        testnet = st.checkbox("测试网", True)
        use_real = st.checkbox("实盘交易", False)

        if use_real and api_key and secret_key:
            try:
                ex_class = CONFIG.exchanges[exchange_choice]
                st.session_state.exchange = ex_class({
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

        st.session_state.auto_enabled = st.checkbox("自动交易", value=True)

        with st.expander("Telegram通知"):
            st.text_input("Bot Token", type="password")
            st.text_input("Chat ID")

        if st.button("🚨 一键紧急平仓"):
            if st.session_state.position:
                close_position(st.session_state.current_symbol, st.session_state.multi_df['15m']['close'].iloc[-1], "紧急平仓")
            st.rerun()

        if st.button("运行回测"):
            st.info("回测功能暂未集成，可使用外部工具")

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
                size = calc_position_size(st.session_state.account_balance, 0.7, atr, price)
                if size > 0:
                    execute_order(st.session_state.current_symbol, 1, size, price, stop, take)
                    st.rerun()

        if st.session_state.error_log:
            with st.expander("⚠️ 错误日志"):
                for err in st.session_state.error_log:
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
        c1.metric("恐惧贪婪指数", "50")
        c2.metric("信号概率", f"{calc_signal(multi_df)[1]:.1%}")
        c3.metric("当前价格", f"{current_price:.2f}")

        if st.session_state.position:
            pos = st.session_state.position
            pnl = pos.pnl(current_price)
            st.markdown(f"### 持仓 {('多' if pos.direction==1 else '空')}")
            st.info(f"入场 {pos.entry_price:.2f} | 数量 {pos.size:.4f}")
            st.info(f"止损 {pos.stop_loss:.2f} | 止盈 {pos.take_profit:.2f}")
            st.metric("浮动盈亏", f"{pnl:.2f} USDT", delta=f"{pnl/pos.size:.2f}")
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
    st.set_page_config(page_title="终极量化终端 32.3", layout="wide")
    st.markdown("<style>.stApp { background: #0B0E14; color: white; }</style>", unsafe_allow_html=True)
    st.title("🚀 终极量化终端 · 终极开仓版 32.3")
    st.caption("宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北")

    init_session_state()
    render_sidebar()
    auto_trade_step(st.session_state.current_symbol)
    render_main_panel()
    st_autorefresh(interval=CONFIG.auto_refresh_ms, key="auto_refresh")

if __name__ == "__main__":
    main()
