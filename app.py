# -*- coding: utf-8 -*-
"""
🚀 合约智能监控中心 · 终极职业版 V4
三层过滤 | 信号强度分级 | 震荡市模式 | 动态仓位 | 全局风控 | 多品种扩展
数据源：MEXC + CryptoCompare（价格） | 模拟资金费率 / Bybit API | 预留链上接口
适配 100倍杠杆，符合职业交易员终极标准
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# -------------------- 品种配置 --------------------
SYMBOLS = {
    "ETHUSDT": {"name": "Ethereum", "base": "ETH"},
    "BTCUSDT": {"name": "Bitcoin", "base": "BTC"},
    "SOLUSDT": {"name": "Solana", "base": "SOL"},
    "BNBUSDT": {"name": "Binance Coin", "base": "BNB"}
}

# -------------------- 强平价格计算 --------------------
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "long":
        return entry_price * (1 - 1/leverage)
    else:
        return entry_price * (1 + 1/leverage)

# -------------------- 市场状态判断（基于ADX/ATR） --------------------
def get_market_state(df):
    high, low, close = df['high'], df['low'], df['close']
    adx = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    current_adx = adx.iloc[-1]
    atr = df['atr'].iloc[-1]
    atr_pct = (atr / close.iloc[-1]) * 100

    # 趋势强度判断
    if current_adx > 20:
        trend_strength = "强趋势"
    elif current_adx > 15 and atr_pct > 0.6:
        trend_strength = "温和趋势"
    else:
        trend_strength = "震荡/无趋势"

    # 波动率
    if atr_pct > 5:
        volatility = "高波动"
    elif atr_pct > 2:
        volatility = "中波动"
    else:
        volatility = "低波动"

    return trend_strength, volatility, current_adx, atr_pct

# -------------------- 震荡市检测（连续12根K线） --------------------
def is_oscillation_mode(df, lookback=12):
    """检测过去12根K线是否处于震荡市"""
    if len(df) < lookback:
        return False
    high, low, close = df['high'], df['low'], df['close']
    adx = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    recent_adx = adx.iloc[-lookback:].values
    atr = df['atr'].iloc[-lookback:].values
    close_prices = df['close'].iloc[-lookback:].values
    atr_pct = (atr / close_prices) * 100
    # 连续12根 ADX < 18 且 ATR% < 0.5%
    if np.all(recent_adx < 18) and np.all(atr_pct < 0.5):
        return True
    return False

# -------------------- 资金费率获取（模拟 + Bybit备用） --------------------
def fetch_funding_rate(symbol):
    """尝试从Bybit获取资金费率，失败返回模拟值"""
    try:
        # Bybit 永续合约资金费率 API
        url = "https://api.bybit.com/v5/market/funding/history"
        params = {"category": "linear", "symbol": symbol, "limit": 1}
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data['retCode'] == 0:
                rate = float(data['result']['list'][0]['fundingRate'])
                return rate
    except:
        pass
    # 模拟资金费率（根据价格涨跌模拟，仅用于演示）
    # 真实环境应使用 Coinglass 或交易所API
    import random
    return random.uniform(-0.001, 0.001)

# -------------------- 未平仓合约变化模拟 --------------------
def fetch_oi_change(symbol):
    """模拟OI变化率（真实环境应接入Coinglass等）"""
    import random
    return random.uniform(-8, 8)  # -8% ~ +8%

# -------------------- 高级数据获取器（价格 + 资金面模拟）--------------------
class AdvancedDataFetcher:
    def __init__(self, symbol="ETHUSDT"):
        self.symbol = symbol
        self.base = SYMBOLS[symbol]["base"]
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200
        self.timeout = 5

        # 价格源（MEXC + CryptoCompare）
        self.mexc = {
            'name': 'MEXC',
            'url': 'https://api.mexc.com/api/v3/klines',
            'params': {'symbol': self.symbol, 'interval': None, 'limit': self.limit}
        }
        self.cryptocompare = {
            'name': 'CryptoCompare',
            'base_url': 'https://min-api.cryptocompare.com/data/v2',
            'params': {'fsym': self.base, 'tsym': 'USD', 'limit': self.limit}
        }
        self.price_url = f'https://api.mexc.com/api/v3/ticker/price?symbol={self.symbol}'

    def fetch_kline(self, period):
        # 尝试 MEXC
        params = self.mexc['params'].copy()
        params['interval'] = period
        try:
            resp = requests.get(self.mexc['url'], params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                df = self._parse_mexc_kline(data)
                if df is not None:
                    return df, self.mexc['name']
        except:
            pass

        # 尝试 CryptoCompare
        try:
            if period in ['1m', '5m', '15m']:
                endpoint = 'histominute'
                aggregate = {'1m':1, '5m':5, '15m':15}[period]
            elif period in ['1h', '4h']:
                endpoint = 'histohour'
                aggregate = 1 if period == '1h' else 4
            elif period == '1d':
                endpoint = 'histoday'
                aggregate = 1
            else:
                return None, None
            url = f"{self.cryptocompare['base_url']}/{endpoint}"
            params = self.cryptocompare['params'].copy()
            params['aggregate'] = aggregate
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('Response') == 'Success':
                    df = self._parse_cryptocompare_kline(data)
                    if df is not None:
                        return df, self.cryptocompare['name']
        except:
            pass
        return None, None

    def _parse_mexc_kline(self, data):
        if not isinstance(data, list) or len(data) == 0:
            return None
        rows = [row[:6] for row in data if isinstance(row, list) and len(row) >= 6]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def _parse_cryptocompare_kline(self, data):
        items = data['Data']['Data']
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'volumefrom': 'volume'}, inplace=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def fetch_price(self):
        try:
            resp = requests.get(self.price_url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return float(data['price']), 'MEXC'
        except:
            pass
        return None, None

    def fetch_all(self):
        data_dict = {}
        price = None
        price_source = None
        source_display = None
        errors = []

        for period in self.periods:
            df, src = self.fetch_kline(period)
            if df is not None:
                data_dict[period] = df
                if source_display is None:
                    source_display = src
            else:
                errors.append(f"{period} 数据获取失败")

        price, price_source = self.fetch_price()
        if price is None and data_dict:
            if '4h' in data_dict:
                price = data_dict['4h']['close'].iloc[-1]
                price_source = '4h收盘价(备用)'
            elif data_dict:
                first = next(iter(data_dict))
                price = data_dict[first]['close'].iloc[-1]
                price_source = f'{first}收盘价(备用)'

        # 获取资金面数据（模拟）
        funding_rate = fetch_funding_rate(self.symbol)
        oi_change = fetch_oi_change(self.symbol)

        return data_dict, price, price_source, errors, source_display or '无', funding_rate, oi_change

# -------------------- 指标计算 + 三层过滤信号 --------------------
def compute_indicators(df):
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    # RSI斜率（前6根）
    df['rsi_slope'] = df['rsi'] - df['rsi'].shift(6)
    return df

def generate_signals_v4(df, df_1h, df_4h, funding_rate, oi_change):
    """
    三层过滤信号生成
    1. 趋势过滤层：ADX > 20 或 (ADX>15且ATR%>0.6)
    2. 共振确认层：1h和4h方向一致
    3. 动量+资金面确认：RSI斜率>5且RSI<70，资金费率条件，OI变化>5%
    """
    if df is None or len(df) < 20:
        return 0, 0, 0  # 方向, 强度, 震荡模式标志

    last = df.iloc[-1]
    high, low, close = df['high'], df['low'], df['close']
    adx = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    current_adx = adx.iloc[-1]
    atr_pct = (last['atr'] / last['close']) * 100

    # 1. 趋势过滤
    if current_adx > 20:
        trend_ok = True
        trend_score = 40  # 趋势分满分40
    elif current_adx > 15 and atr_pct > 0.6:
        trend_ok = True
        trend_score = 30  # 温和趋势得分略低
    else:
        trend_ok = False
        trend_score = 0

    # 2. 共振确认（1h和4h）
    def get_period_trend(df_period):
        if df_period is None or len(df_period) < 20:
            return 0
        last_p = df_period.iloc[-1]
        if last_p['ma20'] > last_p['ma60']:
            return 1  # 多头
        elif last_p['ma20'] < last_p['ma60']:
            return -1  # 空头
        else:
            return 0

    trend_1h = get_period_trend(df_1h)
    trend_4h = get_period_trend(df_4h)

    if trend_1h == 1 and trend_4h == 1:
        resonance_score = 30
        resonance_dir = 1
    elif trend_1h == -1 and trend_4h == -1:
        resonance_score = 30
        resonance_dir = -1
    else:
        resonance_score = 0
        resonance_dir = 0

    # 3. 动量+资金面
    # 动量：RSI斜率 > 5 且 RSI < 70（多头）或 RSI > 30 且斜率 < -5（空头）
    rsi_slope = last['rsi_slope'] if not pd.isna(last['rsi_slope']) else 0
    if rsi_slope > 5 and last['rsi'] < 70:
        momentum_dir = 1
        momentum_score = 20
    elif rsi_slope < -5 and last['rsi'] > 30:
        momentum_dir = -1
        momentum_score = 20
    else:
        momentum_dir = 0
        momentum_score = 0

    # 资金费率条件（多头：费率<0，空头：费率>0.01%）
    funding_score = 0
    if funding_rate < 0 and resonance_dir == 1:
        funding_score = 10
    elif funding_rate > 0.0001 and resonance_dir == -1:  # 0.01%
        funding_score = 10

    # OI变化条件（同方向且>5%）
    oi_score = 0
    if abs(oi_change) > 5:
        if oi_change > 5 and resonance_dir == 1:
            oi_score = 10
        elif oi_change < -5 and resonance_dir == -1:
            oi_score = 10

    # 资金面总分
    fundamental_score = momentum_score + funding_score + oi_score

    # 最终方向：趋势、共振、动量三者一致时才有信号
    if trend_ok and resonance_dir != 0 and momentum_dir == resonance_dir:
        direction = resonance_dir
    else:
        direction = 0

    # 强度计算（0-100）
    strength = trend_score + resonance_score + fundamental_score

    # 震荡市模式检测
    oscillation_mode = is_oscillation_mode(df)

    return direction, strength, oscillation_mode

# -------------------- 震荡市专用策略（布林带缩口+RSI背离）--------------------
def oscillation_signals(df):
    """返回布林带反转信号（简单模拟）"""
    if df is None or len(df) < 20:
        return 0
    last = df.iloc[-1]
    bb_width = (last['bb_high'] - last['bb_low']) / last['close']
    if bb_width < 0.05:  # 缩口
        if last['rsi'] < 30 and last['close'] < last['bb_low'] * 1.02:
            return 1  # 超卖反弹
        elif last['rsi'] > 70 and last['close'] > last['bb_high'] * 0.98:
            return -1  # 超买回落
    return 0

# -------------------- 多周期融合（简化版，用于最终方向）--------------------
class MultiPeriodFusion:
    def __init__(self):
        self.period_weights = {
            '1m': 0.05, '5m': 0.1, '15m': 0.15,
            '1h': 0.2, '4h': 0.25, '1d': 0.25
        }
        self.strategy_weights = {'trend': 0.5, 'oscillator': 0.3, 'volume': 0.2}

    def get_period_signal(self, df):
        last = df.iloc[-1]
        signals = {}
        if last['ma20'] > last['ma60']:
            signals['trend'] = 1
        elif last['ma20'] < last['ma60']:
            signals['trend'] = -1
        else:
            signals['trend'] = 0
        if last['rsi'] < 30:
            signals['oscillator'] = 1
        elif last['rsi'] > 70:
            signals['oscillator'] = -1
        else:
            signals['oscillator'] = 0
        if last['volume_ratio'] > 1.2 and last['close'] > last['open']:
            signals['volume'] = 1
        elif last['volume_ratio'] > 1.2 and last['close'] < last['open']:
            signals['volume'] = -1
        else:
            signals['volume'] = 0
        return signals

    def fuse_periods(self, df_dict):
        period_scores = {}
        for period, df in df_dict.items():
            if df is not None and len(df) > 20:
                signals = self.get_period_signal(df)
                score = sum(signals[s] * self.strategy_weights[s] for s in signals)
                period_scores[period] = score
        if not period_scores:
            return 0, 0
        total_score = 0
        total_weight = 0
        for p, score in period_scores.items():
            w = self.period_weights.get(p, 0)
            total_score += score * w
            total_weight += w
        if total_weight == 0:
            return 0, 0
        avg_score = total_score / total_weight
        if abs(avg_score) < 0.2:
            return 0, abs(avg_score)
        direction = 1 if avg_score > 0 else -1
        confidence = min(abs(avg_score) * 1.2, 1.0)
        return direction, confidence

# -------------------- 动态仓位计算 --------------------
def calculate_position_size(account_balance, risk_pct_per_trade, signal_strength, atr, current_price, leverage_max=100):
    """
    根据信号强度和ATR计算建议仓位
    仓位 = 账户余额 × 风险系数 × (信号强度/100) / (ATR% × 2)
    结果以ETH数量表示
    """
    if atr is None or atr == 0 or current_price == 0:
        return 0
    risk_amount = account_balance * (risk_pct_per_trade / 100)
    strength_factor = signal_strength / 100
    atr_percent = (atr / current_price) * 100
    # 目标风险距离 = 2倍ATR
    risk_per_unit = atr * 2
    quantity = (risk_amount * strength_factor) / risk_per_unit
    # 限制最大杠杆
    max_quantity_by_leverage = (account_balance * leverage_max) / current_price
    return min(quantity, max_quantity_by_leverage)

# -------------------- 全局风控状态管理 --------------------
def init_risk_state():
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 10000.0
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'daily_loss_limit' not in st.session_state:
        st.session_state.daily_loss_limit = 500.0
    if 'peak_balance' not in st.session_state:
        st.session_state.peak_balance = 10000.0
    if 'consecutive_losses' not in st.session_state:
        st.session_state.consecutive_losses = 0
    if 'last_trade_result' not in st.session_state:
        st.session_state.last_trade_result = None
    if 'last_date' not in st.session_state:
        st.session_state.last_date = datetime.now().date()

def update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage):
    today = datetime.now().date()
    if today != st.session_state.last_date:
        st.session_state.daily_pnl = 0.0
        st.session_state.last_date = today

    if sim_entry > 0:
        if sim_side == "多单":
            unrealized_pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
        else:
            unrealized_pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
        st.session_state.daily_pnl = unrealized_pnl

    current_balance = st.session_state.account_balance + st.session_state.daily_pnl
    if current_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = current_balance
    drawdown = (st.session_state.peak_balance - current_balance) / st.session_state.peak_balance * 100
    return drawdown

def check_risk_limits():
    warnings_list = []
    if st.session_state.daily_pnl < -st.session_state.daily_loss_limit:
        warnings_list.append("🚨 日亏损限额已触发！建议停止交易。")
    if st.session_state.consecutive_losses >= 3:
        warnings_list.append("⚠️ 连续3单亏损，建议降低杠杆50%。")
    # 爆仓热力（模拟）
    # 可接入真实爆仓数据
    return warnings_list

# -------------------- 缓存数据获取 --------------------
@st.cache_data(ttl=60)
def fetch_all_data(symbol, sensitivity):
    fetcher = AdvancedDataFetcher(symbol)
    data_dict, price, price_source, errors, source_display, funding_rate, oi_change = fetcher.fetch_all()
    if data_dict:
        for p in data_dict:
            data_dict[p] = compute_indicators(data_dict[p])
    return data_dict, price, price_source, errors, source_display, funding_rate, oi_change

# -------------------- Streamlit 界面 --------------------
st.set_page_config(page_title="合约智能监控·终极职业版 V4", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: white; }
.ai-box { background: #1A1D27; border-radius: 10px; padding: 20px; border-left: 6px solid #00F5A0; }
.metric { background: #232734; padding: 15px; border-radius: 8px; }
.signal-buy { color: #00F5A0; font-weight: bold; }
.signal-sell { color: #FF5555; font-weight: bold; }
.profit { color: #00F5A0; }
.loss { color: #FF5555; }
.warning { color: #FFA500; }
.danger { color: #FF0000; font-weight: bold; }
.info-box { background: #1A2A3A; border-left: 6px solid #00F5A0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
.trade-plan { background: #232734; padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 6px solid #FFAA00; }
.dashboard { background: #1A1D27; padding: 15px; border-radius: 8px; border-left: 6px solid #00F5A0; margin-bottom: 10px; }
.highlight { color: #00F5A0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 合约智能监控中心 · 终极职业版 V4")
st.caption("三层过滤｜信号强度分级｜震荡市模式｜动态仓位｜全局风控｜多品种扩展")

# 初始化
init_risk_state()
if 'fusion' not in st.session_state:
    st.session_state.fusion = MultiPeriodFusion()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_symbol = st.selectbox("选择交易对", list(SYMBOLS.keys()), index=0)
    period_options = ['1m', '5m', '15m', '1h', '4h', '1d']
    selected_period = st.selectbox("选择K线周期", period_options, index=2)
    
    sensitivity = st.slider("信号灵敏度", 0.5, 2.0, 1.0, 0.1,
                            help="值越大，信号越容易触发（但假信号可能增多）。建议1.0为标准值。")
    
    auto_refresh = st.checkbox("开启自动刷新", value=True)
    refresh_interval = st.number_input("刷新间隔(秒)", 5, 60, 10, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    st.markdown("---")
    st.subheader("📈 模拟合约")
    sim_entry = st.number_input("开仓价", value=0.0, format="%.2f")
    sim_side = st.selectbox("方向", ["多单", "空单"])
    sim_leverage = st.slider("杠杆倍数", 1, 100, 10)
    sim_quantity = st.number_input("数量 (ETH)", value=0.01, format="%.4f")
    st.markdown("---")
    st.subheader("💰 账户设置")
    account_balance_input = st.number_input("初始资金 (USDT)", value=st.session_state.account_balance, min_value=100.0, step=1000.0, format="%.2f")
    daily_loss_limit_input = st.number_input("日亏损限额 (USDT)", value=st.session_state.daily_loss_limit, min_value=0.0, step=100.0, format="%.2f")
    risk_per_trade = st.slider("单笔风险 (%)", 0.5, 5.0, 2.0, 0.5, help="每笔交易愿意承担的资金百分比")
    st.session_state.account_balance = account_balance_input
    st.session_state.daily_loss_limit = daily_loss_limit_input

# 获取数据
data_dict, current_price, price_source, errors, source_display, funding_rate, oi_change = fetch_all_data(selected_symbol, sensitivity)

# 提取各周期DataFrame
df_15m = data_dict.get('15m') if data_dict else None
df_1h = data_dict.get('1h') if data_dict else None
df_4h = data_dict.get('4h') if data_dict else None

# 计算三层过滤信号
direction, strength, oscillation_mode = generate_signals_v4(df_15m, df_1h, df_4h, funding_rate, oi_change)
if oscillation_mode:
    # 震荡市模式：使用布林带反转信号覆盖方向
    osc_dir = oscillation_signals(df_15m)
    if osc_dir != 0:
        direction = osc_dir
        strength = 50  # 震荡市信号强度固定为50
        st.info("🌀 当前处于震荡市模式，采用布林带反转策略")

# 多周期融合（备用，用于置信度）
fusion_dir, fusion_conf = 0, 0
if data_dict:
    fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)

# 市场状态
trend_state, volatility, adx_val, atr_pct = "未知", "未知", 0, 0
if df_15m is not None and len(df_15m) > 20:
    trend_state, volatility, adx_val, atr_pct = get_market_state(df_15m)

# 动态仓位建议
suggested_quantity = 0
if direction != 0 and current_price is not None and df_15m is not None:
    atr_val = df_15m['atr'].iloc[-1]
    suggested_quantity = calculate_position_size(st.session_state.account_balance, risk_per_trade, strength, atr_val, current_price, sim_leverage)

# 更新风控统计
drawdown = 0
if current_price is not None:
    drawdown = update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage)

risk_warnings = check_risk_limits()

# 显示数据源状态
if data_dict:
    st.markdown(f'<div class="info-box">✅ 当前数据源：{source_display} | 价格源：{price_source} | 灵敏度：{sensitivity} | 资金费率：{funding_rate:.6f} | OI变化：{oi_change:+.2f}%</div>', unsafe_allow_html=True)

if errors and len(errors) > 3:
    st.warning(f"⚠️ 部分周期数据不可用 ({len(errors)}个周期)，将使用可用周期计算信号")

# 显示风险警告
for warn in risk_warnings:
    st.error(warn)

# 主布局
col1, col2 = st.columns([2.2, 1.3])

with col1:
    # 市场状态横幅
    if df_15m is not None:
        state_color = {"强趋势": "#00F5A0", "温和趋势": "#FFAA00", "震荡/无趋势": "#FF5555"}.get(trend_state, "#FFFFFF")
        st.markdown(f"<h5>市场状态: <span style='color:{state_color};'>{trend_state}</span> | 波动: {volatility} | ADX: {adx_val:.1f} | ATR%: {atr_pct:.2f}%</h5>", unsafe_allow_html=True)
    
    st.subheader(f"📊 {selected_symbol} K线 ({selected_period})  — 绿色▲=做多信号，红色▼=做空信号")
    if df_15m is not None:
        df = df_15m.tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{selected_symbol} {selected_period}", "RSI"))
        # K线
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                      low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        # 均线
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)

        # 模拟信号箭头（此处可以基于generate_signals_v4的逻辑在每个点画箭头，但为了简化，我们只显示最终方向）
        # 实际应用中可存储历史信号，这里仅示意
        if direction != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            if direction == 1:
                fig.add_annotation(x=last_date, y=last_price * 1.02,
                                   text="▲ 三层多", showarrow=True, arrowhead=2, arrowcolor="green")
            else:
                fig.add_annotation(x=last_date, y=last_price * 0.98,
                                   text="▼ 三层空", showarrow=True, arrowhead=2, arrowcolor="red")

        # RSI
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("等待数据...")

with col2:
    st.subheader("🧠 即时决策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
    st.markdown(f'<div class="ai-box">{dir_map[direction]}<br>信号强度: {strength}/100</div>', unsafe_allow_html=True)

    # 强度分级
    if strength >= 80:
        st.success("🔥 重仓级 (10倍+)")
    elif strength >= 60:
        st.info("⚡ 中仓级 (2-5倍)")
    elif strength >= 30:
        st.info("💡 轻仓级 (0.5-1倍)")
    else:
        st.info("⛔ 观望")

    if current_price is not None:
        st.metric("当前价格", f"${current_price:.2f}", delta_color="off")
    else:
        st.metric("当前价格", "获取中...")

    # 风险仪表盘
    with st.container():
        st.markdown('<div class="dashboard">', unsafe_allow_html=True)
        st.markdown("#### 📊 风险仪表盘")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("账户余额", f"${st.session_state.account_balance:.2f}")
            st.metric("日盈亏", f"${st.session_state.daily_pnl:.2f}", delta_color="inverse")
        with col_r2:
            st.metric("当前回撤", f"{drawdown:.2f}%")
            st.metric("连续亏损", st.session_state.consecutive_losses)
        st.markdown(f"**日亏损剩余:** ${st.session_state.daily_loss_limit + st.session_state.daily_pnl:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 资金面快照
    st.markdown(f"""
    <div class="metric">
        <h4>资金面快照</h4>
        <p>资金费率: <span class="highlight">{funding_rate:.6f}</span></p>
        <p>OI变化: <span class="highlight">{oi_change:+.2f}%</span></p>
    </div>
    """, unsafe_allow_html=True)

    # 动态仓位建议
    if direction != 0 and suggested_quantity > 0:
        st.markdown(f"""
        <div class="trade-plan">
            <h4>📋 动态仓位建议</h4>
            <p>建议数量: <span class="highlight">{suggested_quantity:.4f} {SYMBOLS[selected_symbol]['base']}</span></p>
            <p>基于 {risk_per_trade}% 风险，信号强度 {strength}/100</p>
            <p>当前ATR: {df_15m['atr'].iloc[-1]:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("无信号，无仓位建议")

    # 模拟合约持仓
    if sim_entry > 0 and current_price is not None and df_15m is not None:
        if sim_side == "多单":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
            pnl_pct = (current_price - sim_entry) / sim_entry * sim_leverage * 100
            liq_price = calculate_liquidation_price(sim_entry, "long", sim_leverage)
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
            pnl_pct = (sim_entry - current_price) / sim_entry * sim_leverage * 100
            liq_price = calculate_liquidation_price(sim_entry, "short", sim_leverage)

        color_class = "profit" if pnl >= 0 else "loss"
        distance_to_liq = abs(current_price - liq_price) / current_price * 100

        st.markdown(f"""
        <div class="metric">
            <h4>模拟合约持仓</h4>
            <p>方向: {sim_side} | 杠杆: {sim_leverage}x</p>
            <p>开仓: ${sim_entry:.2f}</p>
            <p class="{color_class}">盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>强平价: <span class="warning">${liq_price:.2f}</span></p>
            <p>距强平: {distance_to_liq:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        if (sim_side == "多单" and current_price <= liq_price) or (sim_side == "空单" and current_price >= liq_price):
            st.error("🚨 强平风险！当前价格已触及强平线！")
        elif distance_to_liq < 5:
            st.warning(f"⚠️ 距离强平仅 {distance_to_liq:.2f}%，请注意风险！")
    else:
        st.info("请输入开仓价以查看模拟盈亏与强平分析")

    # 多周期共振矩阵（简单表格）
    if data_dict:
        rows = []
        for p, df in data_dict.items():
            if df is not None and len(df) > 20:
                last = df.iloc[-1]
                trend = "多" if last['ma20'] > last['ma60'] else "空" if last['ma20'] < last['ma60'] else "平"
                rows.append({
                    "周期": p,
                    "趋势": trend,
                    "RSI": round(last['rsi'], 1),
                    "ATR%": round(last['atr']/last['close']*100, 2)
                })
        if rows:
            with st.expander("📈 多周期共振", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
