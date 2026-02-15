# -*- coding: utf-8 -*-
"""
🚀 机构级量化终端 · 原型 v1
环境 → 规则 → 信号 → 风险 → 资本 → 监控
不预测、不情绪化、不解释，只展示决策必要变量
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
import time
from streamlit_autorefresh import st_autorefresh
import warnings
import joblib
import os
from collections import Counter

warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT"]
BASE_RISK = 0.01
MAX_LEVERAGE_GLOBAL = 125.0
DAILY_LOSS_LIMIT = 300.0
MIN_ATR_PCT = 0.5

LEVERAGE_MODES = {
    "低倍试炼 (3-10x)": (3, 10),
    "中倍试炼 (20-50x)": (20, 50),
    "高倍神级 (50-125x)": (50, 125)
}

# ==================== 免费数据获取器 ====================
class FreeDataFetcherV5:
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = SYMBOLS
        self.symbols = symbols
        self.periods = ['15m', '1h', '4h', '1d']
        self.limit = 500
        self.timeout = 10
        self.exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
        self.fng_url = "https://api.alternative.me/fng/"
        self.chain_netflow = 5234
        self.chain_whale = 128

    def fetch_kline(self, symbol, timeframe):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df, "MEXC"
        except Exception as e:
            return None, None

    def fetch_fear_greed(self):
        try:
            resp = requests.get(self.fng_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return int(data['data'][0]['value'])
        except:
            pass
        return 50

    def fetch_all(self):
        all_data = {}
        fear_greed = self.fetch_fear_greed()
        for symbol in self.symbols:
            data_dict = {}
            price_sources = []
            data_ok = True
            for period in self.periods:
                df, src = self.fetch_kline(symbol, period)
                if df is not None:
                    data_dict[period] = self._add_indicators(df)
                    price_sources.append(src)
                else:
                    data_ok = False
            if data_ok and data_dict:
                all_data[symbol] = {
                    "data_dict": data_dict,
                    "current_price": data_dict['15m']['close'].iloc[-1] if '15m' in data_dict else None,
                    "source": price_sources[0] if price_sources else "MEXC",
                    "fear_greed": fear_greed,
                    "chain_netflow": self.chain_netflow,
                    "chain_whale": self.chain_whale,
                }
            else:
                all_data[symbol] = {
                    "data_dict": None,
                    "current_price": None,
                    "source": "不可用",
                    "fear_greed": fear_greed,
                    "chain_netflow": self.chain_netflow,
                    "chain_whale": self.chain_whale,
                }
        return all_data

    def _add_indicators(self, df):
        df = df.copy()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100.0
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        return df


# ==================== 策略配置 ====================
def get_mode_config(mode):
    if mode == "稳健":
        return {
            'min_five_score': 60,
            'fear_threshold': 20,
            'netflow_required': 5000,
            'whale_required': 100,
            'stop_atr': 1.8,
            'tp_min_ratio': 2.5,
            'position_pct': lambda fear: 0.6 if fear <= 10 else (0.3 if fear <= 20 else 0.0),
        }
    elif mode == "无敌":
        return {
            'min_five_score': 70,
            'fear_threshold': 15,
            'netflow_required': 6000,
            'whale_required': 120,
            'stop_atr': 2.0,
            'tp_min_ratio': 3.0,
            'position_pct': lambda fear: 1.0 if fear <= 10 else (0.5 if fear <= 20 else 0.0),
        }
    elif mode == "神级":
        return {
            'min_five_score': 80,
            'fear_threshold': 8,
            'netflow_required': 8000,
            'whale_required': 150,
            'stop_atr': 2.5,
            'tp_min_ratio': 4.0,
            'position_pct': lambda fear: 1.0 if fear <= 8 else (0.8 if fear <= 15 else 0.0),
        }
    else:
        return get_mode_config("稳健")


def evaluate_market(df_dict):
    if df_dict is None or '15m' not in df_dict:
        return "数据不足", 0.0, 0.0
    df = df_dict['15m']
    if df.empty:
        return "数据不足", 0.0, 0.0
    last = df.iloc[-1]

    ema20 = last['ema20']
    ema50 = last['ema50']
    adx = last['adx']
    atr_pct = last['atr_pct']

    body = abs(last['close'] - last['open'])
    if body > 3 * last['atr']:
        return "异常波动", atr_pct, adx

    if ema20 > ema50 and adx > 20:
        return "趋势", atr_pct, adx
    elif adx < 25:
        return "震荡", atr_pct, adx
    else:
        return "不明朗", atr_pct, adx


def five_layer_score(df_dict, fear_greed, chain_netflow, chain_whale):
    if df_dict is None or any(period not in df_dict for period in ['15m', '1h', '4h', '1d']):
        return 0, 0, {}

    df_15m = df_dict['15m']
    df_1h = df_dict['1h']
    df_4h = df_dict['4h']
    df_1d = df_dict['1d']

    if any(df.empty for df in [df_15m, df_1h, df_4h, df_1d]):
        return 0, 0, {}

    last_15m = df_15m.iloc[-1]
    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]
    last_1d = df_1d.iloc[-1]

    # 趋势因子
    trend_score = 0
    trend_dir = 0
    adx = last_15m['adx']
    if adx > 25:
        trend_score = 20
        trend_dir = 1 if last_15m['ema20'] > last_15m['ema50'] else -1
    elif adx > 20:
        trend_score = 10
        trend_dir = 1 if last_15m['ema20'] > last_15m['ema50'] else -1

    # 多周期因子
    multi_score = 0
    multi_dir = 0
    dir_15m = 1 if last_15m['ema20'] > last_15m['ema50'] else -1
    dir_1h = 1 if last_1h['ema20'] > last_1h['ema50'] else -1
    dir_4h = 1 if last_4h['ema20'] > last_4h['ema50'] else -1
    dir_1d = 1 if last_1d['ema20'] > last_1d['ema50'] else -1

    if dir_15m == dir_1h == dir_4h == dir_1d:
        multi_score = 20
        multi_dir = dir_15m
    elif dir_15m == dir_1h == dir_4h:
        multi_score = 15
        multi_dir = dir_15m
    elif dir_15m == dir_1h:
        multi_score = 10
        multi_dir = dir_15m

    # 资金因子（模拟）
    fund_score = 0
    fund_dir = 0

    # 链上情绪因子
    chain_score = 0
    chain_dir = 0
    if chain_netflow > 5000 and chain_whale > 100:
        chain_score = 20
        chain_dir = 1
    elif fear_greed < 30:
        chain_score = 15
        chain_dir = 1
    elif fear_greed > 70:
        chain_score = 15
        chain_dir = -1

    # 动量因子
    momentum_score = 0
    momentum_dir = 0
    rsi = last_15m['rsi']
    macd_diff = last_15m['macd'] - last_15m['macd_signal']
    if rsi > 55 and macd_diff > 0:
        momentum_score = 20
        momentum_dir = 1
    elif rsi < 45 and macd_diff < 0:
        momentum_score = 20
        momentum_dir = -1
    elif rsi > 50:
        momentum_score = 10
        momentum_dir = 1
    elif rsi < 50:
        momentum_score = 10
        momentum_dir = -1

    total_score = trend_score + multi_score + fund_score + chain_score + momentum_score

    dirs = [trend_dir, multi_dir, fund_dir, chain_dir, momentum_dir]
    dirs = [d for d in dirs if d != 0]
    if len(dirs) >= 3:
        count = Counter(dirs)
        final_dir = count.most_common(1)[0][0]
    else:
        final_dir = 0

    layer_scores = {
        "趋势": trend_score,
        "多周期": multi_score,
        "资金": fund_score,
        "链上": chain_score,
        "动量": momentum_score
    }
    return final_dir, total_score, layer_scores


def generate_entry_signal(five_dir, five_total, fear_greed, netflow, whale_tx, config):
    if five_total < config['min_five_score']:
        return 0
    if fear_greed > config['fear_threshold']:
        return 0
    if netflow < config['netflow_required']:
        return 0
    if whale_tx < config['whale_required']:
        return 0
    if five_dir == 0:
        return 0
    return five_dir


def calculate_stops(entry_price, side, atr_value, stop_atr, tp_min_ratio):
    stop_distance = stop_atr * atr_value
    take_distance = stop_distance * tp_min_ratio
    if side == 1:
        stop = entry_price - stop_distance
        take = entry_price + take_distance
    else:
        stop = entry_price + stop_distance
        take = entry_price - take_distance
    return stop, take, take_distance/stop_distance


def calculate_position_size(balance, entry_price, stop_price, leverage, position_pct):
    risk_amount = balance * position_pct
    nominal = risk_amount * leverage
    quantity = nominal / entry_price
    return round(quantity, 3)


def liquidation_price(entry_price, side, leverage):
    if side == 1:
        return entry_price * (1 - 1.0/leverage)
    else:
        return entry_price * (1 + 1.0/leverage)


def init_risk_state():
    if 'consecutive_losses' not in st.session_state:
        st.session_state.consecutive_losses = 0
    if 'peak_balance' not in st.session_state:
        st.session_state.peak_balance = 10000.0
    if 'daily_loss_triggered' not in st.session_state:
        st.session_state.daily_loss_triggered = False
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 10000.0
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'last_date' not in st.session_state:
        st.session_state.last_date = datetime.now().date()
    if 'balance_history' not in st.session_state:
        st.session_state.balance_history = []
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'auto_enabled' not in st.session_state:
        st.session_state.auto_enabled = False
    if 'auto_position' not in st.session_state:
        st.session_state.auto_position = None
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []


def update_risk_state(trade_result, current_balance, daily_pnl):
    if trade_result < 0:
        st.session_state.consecutive_losses += 1
    else:
        st.session_state.consecutive_losses = 0
    if current_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = current_balance
    drawdown = (st.session_state.peak_balance - current_balance) / st.session_state.peak_balance * 100.0
    if daily_pnl < -DAILY_LOSS_LIMIT:
        st.session_state.daily_loss_triggered = True
    return drawdown


def can_trade():
    return not st.session_state.daily_loss_triggered


# ==================== 主界面 ====================
st.set_page_config(page_title="机构量化终端 · 原型v1", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: white; font-size: 0.85rem; }
.card { background: #1A1D27; border-radius: 4px; padding: 10px; margin-bottom: 8px; border-left: 4px solid #00F5A0; }
.card-header { font-size: 0.9rem; color: #8A8F9C; margin-bottom: 6px; }
.metric-row { display: flex; justify-content: space-between; }
.metric-item { text-align: left; }
.metric-label { font-size: 0.75rem; color: #8A8F9C; }
.metric-value { font-size: 1.1rem; font-weight: bold; }
.risk-factor { display: flex; justify-content: space-between; font-size: 0.9rem; padding: 2px 0; }
.risk-line { border-top: 1px solid #333; margin: 6px 0; }
.factor-name { color: #8A8F9C; }
.factor-value { font-weight: bold; }
.eligibility-blocked { color: #FF5555; font-weight: bold; }
.eligibility-active { color: #00F5A0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ 机构量化终端 · 原型 v1")
st.caption("环境 → 规则 → 信号 → 风险 → 资本 → 监控")

init_risk_state()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 市场设置")
    selected_symbol = st.selectbox("交易品种", SYMBOLS, index=0, key="selected_symbol")
    main_period = st.selectbox("分析周期", ["15m", "1h", "4h", "1d"], index=0)
    auto_refresh = st.checkbox("自动刷新", value=True)
    refresh_interval = st.number_input("刷新间隔(秒)", min_value=5, max_value=300, value=60, step=1, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

    st.markdown("---")
    st.subheader("🧬 策略模式")
    manual_mode = st.selectbox("手动选择", ["稳健", "无敌", "神级"], index=0)
    auto_mode = st.checkbox("自动模式切换 (AI 推荐)", value=False)
    st.markdown("---")

    st.subheader("🔥 高倍试炼")
    leverage_mode = st.selectbox("杠杆模式", list(LEVERAGE_MODES.keys()), index=0)
    min_lev, max_lev = LEVERAGE_MODES[leverage_mode]
    st.info(f"当前试炼范围: {min_lev}x - {max_lev}x")

    st.markdown("---")
    st.subheader("📊 风险参数")
    account_balance = st.number_input("账户余额 (USDT)", value=st.session_state.account_balance, step=1000.0, format="%.2f")
    st.session_state.account_balance = account_balance
    daily_loss_limit = st.number_input("日亏损限额 (USDT)", value=DAILY_LOSS_LIMIT, step=50.0, format="%.2f")
    st.session_state.daily_loss_limit = daily_loss_limit

# ==================== 获取数据 ====================
with st.spinner("获取市场数据..."):
    fetcher = FreeDataFetcherV5(symbols=SYMBOLS)
    all_data = fetcher.fetch_all()

# ==================== 多品种卡片（顶部快捷栏）====================
# 改为表格形式放在底部，此处移除，直接在底部实现

# ==================== 当前品种数据处理 ====================
if selected_symbol not in all_data or all_data[selected_symbol]["data_dict"] is None:
    st.error(f"❌ 品种 {selected_symbol} 数据不可用，请稍后重试")
    st.stop()

data = all_data[selected_symbol]
data_dict = data["data_dict"]
current_price = data["current_price"]
fear_greed = data["fear_greed"]
source_display = data["source"]
netflow = data["chain_netflow"]
whale = data["chain_whale"]

# 多因子强度
five_dir, five_total, layer_scores = five_layer_score(data_dict, fear_greed, netflow, whale)

# 市场环境
market_mode, atr_pct, adx = evaluate_market(data_dict)

# 自动模式选择
if auto_mode:
    if five_total >= 80 and fear_greed <= 10 and atr_pct <= 2.5:
        mode = "神级"
    elif five_total >= 70 and fear_greed <= 15 and atr_pct <= 3.0:
        mode = "无敌"
    else:
        mode = "稳健"
else:
    mode = manual_mode

config = get_mode_config(mode)
entry_signal = generate_entry_signal(five_dir, five_total, fear_greed, netflow, whale, config)

atr_value = data_dict['15m']['atr'].iloc[-1] if '15m' in data_dict else 0.0
position_pct = config['position_pct'](fear_greed)
suggested_leverage = (min_lev + max_lev) / 2

stop_loss = take_profit = risk_reward = None
position_size = 0.0
if entry_signal != 0 and atr_value > 0:
    stop_loss, take_profit, risk_reward = calculate_stops(current_price, entry_signal, atr_value, config['stop_atr'], config['tp_min_ratio'])
    position_size = calculate_position_size(
        st.session_state.account_balance,
        current_price,
        stop_loss,
        suggested_leverage,
        position_pct
    )

# 计算风险因子
F_quality = five_total / 100.0
F_volatility = 1.0 if atr_pct > 0.8 else 0.5
drawdown = update_risk_state(0.0, st.session_state.account_balance + st.session_state.daily_pnl, st.session_state.daily_pnl)
F_drawdown = 1.0 if drawdown < 10 else 0.5
F_loss_streak = 1.0 if st.session_state.consecutive_losses < 3 else 0.5
R_final = BASE_RISK * F_quality * F_volatility * F_drawdown * F_loss_streak
R_final = max(0.001, min(0.02, R_final))

capital_at_risk = st.session_state.account_balance * R_final

# 强平价格计算
if entry_signal == 1:
    liq_price = liquidation_price(current_price, 1, suggested_leverage)
    distance_to_liq = (current_price - liq_price) / current_price * 100
elif entry_signal == -1:
    liq_price = liquidation_price(current_price, -1, suggested_leverage)
    distance_to_liq = (liq_price - current_price) / current_price * 100
else:
    liq_price = None
    distance_to_liq = None

can_trade_flag = can_trade()
eligibility = "ACTIVE" if can_trade_flag and entry_signal != 0 else "BLOCKED"

# ==================== 主布局：左侧信息面板，右侧图表 ====================
col_left, col_right = st.columns([1.4, 1.6])

with col_left:
    # ① 全球宏观面板
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">① GLOBAL REGIME PANEL</div>', unsafe_allow_html=True)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"<div class='metric-label'>Market Regime</div><div class='metric-value'>{market_mode}</div>", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"<div class='metric-label'>Volatility (ATR)</div><div class='metric-value'>{atr_pct:.2f}%</div>", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"<div class='metric-label'>Trend Strength</div><div class='metric-value'>{adx:.1f}</div>", unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"<div class='metric-label'>Fear Index</div><div class='metric-value'>{fear_greed}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:4px;'>Data Source: {source_display}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ② 策略概况
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">② STRATEGY PROFILE</div>', unsafe_allow_html=True)
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(f"<div class='metric-label'>Strategy Mode</div><div class='metric-value'>{mode}</div>", unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"<div class='metric-label'>Leverage Band</div><div class='metric-value'>{min_lev:.0f}x–{max_lev:.0f}x</div>", unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"<div class='metric-label'>AI Profile</div><div class='metric-value'>{mode}</div>", unsafe_allow_html=True)
    with col_s4:
        st.markdown(f"<div class='metric-label'>Daily Loss Limit</div><div class='metric-value'>{DAILY_LOSS_LIMIT:.0f} USDT</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ③ 信号引擎
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">③ SIGNAL ENGINE</div>', unsafe_allow_html=True)
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    with col_i1:
        st.markdown(f"<div class='metric-label'>Instrument</div><div class='metric-value'>{selected_symbol}</div>", unsafe_allow_html=True)
    with col_i2:
        st.markdown(f"<div class='metric-label'>Timeframe</div><div class='metric-value'>{main_period}</div>", unsafe_allow_html=True)
    with col_i3:
        status = "WAIT" if entry_signal == 0 else ("LONG" if entry_signal == 1 else "SHORT")
        st.markdown(f"<div class='metric-label'>Signal Status</div><div class='metric-value'>{status}</div>", unsafe_allow_html=True)
    with col_i4:
        st.markdown(f"<div class='metric-label'>Strength</div><div class='metric-value'>{five_total}/100</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:6px;'><span class='metric-label'>Execution Eligibility:</span> <span class='eligibility-{'active' if eligibility=='ACTIVE' else 'blocked'}'>{eligibility}</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ④ 风险引擎（核心）
    st.markdown('<div class="card" style="border-left-color: #FFAA00;">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">④ RISK ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-factor"><span class="factor-name">F_quality</span><span class="factor-value">{:.2f}</span></div>'.format(F_quality), unsafe_allow_html=True)
    st.markdown('<div class="risk-factor"><span class="factor-name">F_volatility</span><span class="factor-value">{:.2f}</span></div>'.format(F_volatility), unsafe_allow_html=True)
    st.markdown('<div class="risk-factor"><span class="factor-name">F_drawdown</span><span class="factor-value">{:.2f}</span></div>'.format(F_drawdown), unsafe_allow_html=True)
    st.markdown('<div class="risk-factor"><span class="factor-name">F_loss_streak</span><span class="factor-value">{:.2f}</span></div>'.format(F_loss_streak), unsafe_allow_html=True)
    st.markdown('<div class="risk-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-factor"><span class="factor-name">R_final</span><span class="factor-value">{:.2f}%</span></div>'.format(R_final*100), unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; margin-top:8px;">'
                f'<div><span class="metric-label">Capital at Risk</span><br><span class="metric-value">{capital_at_risk:.1f} USDT</span></div>'
                f'<div><span class="metric-label">Suggested Leverage</span><br><span class="metric-value">{suggested_leverage:.1f}x</span></div>'
                f'<div><span class="metric-label">Position Allocation</span><br><span class="metric-value">Dynamic</span></div>'
                '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ⑤ 资本状态
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⑤ CAPITAL STATE</div>', unsafe_allow_html=True)
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.markdown(f"<div class='metric-label'>Account Balance</div><div class='metric-value'>{st.session_state.account_balance:.0f} USDT</div>", unsafe_allow_html=True)
    with col_c2:
        st.markdown(f"<div class='metric-label'>Daily PnL</div><div class='metric-value'>{st.session_state.daily_pnl:.1f}</div>", unsafe_allow_html=True)
    with col_c3:
        st.markdown(f"<div class='metric-label'>Current Drawdown</div><div class='metric-value'>{drawdown:.2f}%</div>", unsafe_allow_html=True)
    with col_c4:
        st.markdown(f"<div class='metric-label'>Loss Streak</div><div class='metric-value'>{st.session_state.consecutive_losses}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ⑥ 市场监控（底部表格）
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⑥ MARKET MONITOR</div>', unsafe_allow_html=True)
    monitor_data = []
    for sym in SYMBOLS:
        if sym in all_data and all_data[sym]["data_dict"] is not None:
            df_dict = all_data[sym]["data_dict"]
            f = all_data[sym]["fear_greed"]
            n = all_data[sym]["chain_netflow"]
            w = all_data[sym]["chain_whale"]
            dir_, total, _ = five_layer_score(df_dict, f, n, w)
            status = "ACTIVE" if total >= 60 else "NEUTRAL"
            monitor_data.append([sym, total, status])
        else:
            monitor_data.append([sym, "—", "UNAVAILABLE"])
    df_monitor = pd.DataFrame(monitor_data, columns=["Instrument", "Strength", "Status"])
    st.table(df_monitor)
    st.markdown('</div>', unsafe_allow_html=True)

    # ⑦ 执行日志（折叠）
    with st.expander("⑦ EXECUTION LOG"):
        tab1, tab2 = st.tabs(["Trade Log", "Signal History"])
        with tab1:
            if st.session_state.trade_log:
                st.dataframe(pd.DataFrame(st.session_state.trade_log), use_container_width=True, height=150)
            else:
                st.info("暂无交易记录")
        with tab2:
            if st.session_state.signal_history:
                st.dataframe(pd.DataFrame(st.session_state.signal_history), use_container_width=True, height=150)
            else:
                st.info("暂无历史信号")

with col_right:
    # 图表：K线 + 成交量 + RSI
    st.subheader(f"📈 {selected_symbol} K线 ({main_period})")
    if main_period in data_dict:
        df = data_dict[main_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.2, 0.2],
                           subplot_titles=("", "", ""))
        # K线
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K线", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema20'], name="EMA20", line=dict(color="orange", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema50'], name="EMA50", line=dict(color="blue", width=1), showlegend=False), row=1, col=1)
        # 当前价格水平线
        fig.add_hline(y=current_price, line_dash="dot", line_color="white", annotation_text=f"现价 {current_price:.2f}", row=1, col=1)

        if entry_signal != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            arrow_text = "▲" if entry_signal == 1 else "▼"
            arrow_color = "green" if entry_signal == 1 else "red"
            fig.add_annotation(x=last_date, y=last_price * (1.02 if entry_signal==1 else 0.98),
                               text=arrow_text, showarrow=True, arrowhead=2, arrowcolor=arrow_color, font=dict(size=12))

        # RSI
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple", width=1), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        # 成交量
        colors_vol = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['日期'], y=df['volume'], name="成交量", marker_color=colors_vol, showlegend=False), row=3, col=1)

        fig.update_layout(hovermode='x unified', template="plotly_dark", xaxis_rangeslider_visible=False, height=600, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # 显示最新MACD值
        latest_macd = df['macd'].iloc[-1]
        latest_signal = df['macd_signal'].iloc[-1]
        st.markdown(f"<span style='font-size:0.8rem;'>MACD: {latest_macd:.2f} | Signal: {latest_signal:.2f}</span>", unsafe_allow_html=True)
    else:
        st.warning("K线数据不可用")

# ==================== 自动交易逻辑（保留原样，但不在主界面显示，仅用于日志）====================
# 保持原有自动交易逻辑，但不影响UI，仅当启用时记录日志
now = datetime.now()
if st.session_state.get('auto_enabled', False) and can_trade_flag and entry_signal != 0:
    if st.session_state.auto_position is None:
        st.session_state.auto_position = {
            'side': 'long' if entry_signal == 1 else 'short',
            'entry': current_price,
            'time': now,
            'leverage': suggested_leverage,
            'stop': stop_loss,
            'take': take_profit,
            'size': position_size
        }
        # 记录信号历史
        st.session_state.signal_history.append({
            '时间': now.strftime("%H:%M"),
            '方向': '多' if entry_signal == 1 else '空',
            '市场': market_mode,
            '多因子强度': five_total
        })
        # 发送通知等（略）
    else:
        pos = st.session_state.auto_position
        if (pos['side'] == 'long' and (current_price <= pos['stop'] or current_price >= pos['take'])) or \
           (pos['side'] == 'short' and (current_price >= pos['stop'] or current_price <= pos['take'])) or \
           (entry_signal == -1 and pos['side'] == 'long') or \
           (entry_signal == 1 and pos['side'] == 'short'):
            if pos['side'] == 'long':
                pnl = (current_price - pos['entry']) * pos['size']
            else:
                pnl = (pos['entry'] - current_price) * pos['size']
            pnl_pct = pnl / (pos['entry'] * pos['size']) * 100.0
            update_risk_state(pnl, st.session_state.account_balance + st.session_state.daily_pnl, st.session_state.daily_pnl)
            st.session_state.trade_log.append({
                '开仓时间': pos['time'].strftime('%H:%M'),
                '方向': pos['side'],
                '开仓价': f"{pos['entry']:.2f}",
                '平仓时间': now.strftime('%H:%M'),
                '平仓价': f"{current_price:.2f}",
                '盈亏': f"{pnl:.2f}",
                '盈亏%': f"{pnl_pct:.1f}%"
            })
            st.session_state.balance_history.append(st.session_state.account_balance + st.session_state.daily_pnl)
            st.session_state.auto_position = None
