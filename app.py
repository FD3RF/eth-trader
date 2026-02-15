# -*- coding: utf-8 -*-
"""
🚀 合约量化终端 · 紧凑神境版（多空双向）
恐惧卡片｜强平预警｜成交量｜市场状态｜多因子强度｜资本监控｜自动策略切换
支持做多/做空双向信号
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
        self.chain_netflow = 5234  # 模拟值
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


# ==================== 入场信号（双向） ====================
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
    return five_dir  # 返回实际方向（1多/-1空）


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
st.set_page_config(page_title="合约量化终端 · 多空双向", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: white; font-size: 0.9rem; }
.ai-box { background: #1A1D27; border-radius: 6px; padding: 12px; border-left: 4px solid #00F5A0; margin-bottom: 8px; }
.metric { background: #232734; padding: 8px; border-radius: 4px; }
.signal-buy { color: #00F5A0; font-weight: bold; }
.signal-sell { color: #FF5555; font-weight: bold; }
.profit { color: #00F5A0; }
.loss { color: #FF5555; }
.warning { color: #FFA500; }
.danger { color: #FF0000; font-weight: bold; }
.info-box { background: #1A2A3A; border-left: 4px solid #00F5A0; padding: 6px; border-radius: 4px; margin-bottom: 6px; font-size:0.85rem; }
.trade-plan { background: #232734; padding: 10px; border-radius: 4px; margin-top: 6px; border-left: 4px solid #FFAA00; font-size:0.9rem; }
.dashboard { background: #1A1D27; padding: 10px; border-radius: 4px; border-left: 4px solid #00F5A0; margin-bottom: 6px; }
.card { background: #1A1D27; border-radius: 4px; padding: 6px; text-align: center; cursor: pointer; font-size:0.85rem; }
.card:hover { background: #2A2D37; }
.fear-card { background: #8B0000; color: white; padding: 8px; border-radius: 6px; text-align: center; animation: blink 1s infinite; font-size:1rem; margin-bottom:8px; }
@keyframes blink { 50% { background-color: #B22222; } }
</style>
""", unsafe_allow_html=True)

st.title("📊 合约量化终端 · 多空双向版")
st.caption("恐惧卡片｜强平预警｜成交量｜多空信号｜紧凑布局")

init_risk_state()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 市场设置")
    selected_symbol = st.selectbox("交易品种", SYMBOLS, index=0, key="selected_symbol")
    main_period = st.selectbox("分析周期", ["15m", "1h", "4h", "1d"], index=0)
    auto_refresh = st.checkbox("自动刷新", value=True)
    refresh_interval = st.number_input("刷新间隔(秒)", min_value=5, max_value=60, value=10, step=1, disabled=not auto_refresh)
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

# ==================== 多品种卡片 ====================
cols = st.columns(len(SYMBOLS))
for i, sym in enumerate(SYMBOLS):
    if sym in all_data and all_data[sym]["data_dict"] is not None:
        df_dict = all_data[sym]["data_dict"]
        fear = all_data[sym]["fear_greed"]
        netflow = all_data[sym]["chain_netflow"]
        whale = all_data[sym]["chain_whale"]
        five_dir, five_total, _ = five_layer_score(df_dict, fear, netflow, whale)
        dir_icon = {1: "🟢", -1: "🔴", 0: "⚪"}[five_dir]
        with cols[i]:
            st.markdown(f"<div class='card'><b>{sym}</b><br>强度 {five_total}<br>{dir_icon}</div>", unsafe_allow_html=True)
    else:
        with cols[i]:
            st.markdown(f"<div class='card'><b>{sym}</b><br>数据不可用</div>", unsafe_allow_html=True)

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
    st.sidebar.info(f"🤖 AI 推荐模式: {mode}")
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

current_balance = st.session_state.account_balance + st.session_state.daily_pnl
drawdown = update_risk_state(0.0, current_balance, st.session_state.daily_pnl)
can_trade_flag = can_trade()

# ==================== 恐惧贪婪卡片 ====================
fear_card_style = "fear-card" if fear_greed <= 10 else "info-box"
st.markdown(f"""
<div class="{fear_card_style}" style="margin-bottom:6px;">
    <span style="font-size:1.2rem;">😨 恐惧贪婪指数: {fear_greed}</span>
    <span style="margin-left:15px;">{'🚨 极度恐惧 — 神底信号' if fear_greed <= 10 else '😐 正常范围'}</span>
</div>
""", unsafe_allow_html=True)

# ==================== 顶部状态信息 ====================
st.markdown(f"""
<div class="info-box">
    ✅ 数据源：{source_display} | 市场环境：{market_mode} | 多因子强度：{five_total}
    <br>⚠️ 链上数据为模拟值 | { '🔴 交易暂停' if not can_trade_flag else '' }
</div>
""", unsafe_allow_html=True)

# ==================== 主布局：两列 ====================
col_left, col_right = st.columns([1.7, 1.3])

with col_left:
    # 市场状态卡片（三指标）
    col_state1, col_state2, col_state3 = st.columns(3)
    with col_state1:
        st.metric("市场状态", market_mode, delta=None)
    with col_state2:
        st.metric("波动率(ATR%)", f"{atr_pct:.2f}%")
    with col_state3:
        st.metric("趋势强度(ADX)", f"{adx:.1f}")

    # 多因子强度热力图
    st.markdown("#### 📊 多因子强度")
    cols = st.columns(5)
    layer_names = list(layer_scores.keys())
    layer_values = list(layer_scores.values())
    colors = ['#00F5A0', '#00F5A0', '#FFAA00', '#FF5555', '#FFAA00']
    for i, col in enumerate(cols):
        with col:
            val = layer_values[i]
            bg_color = colors[i] if val > 10 else '#555'
            st.markdown(f"""
            <div style="background:{bg_color}22; border-left:2px solid {bg_color}; padding:4px; border-radius:3px; text-align:center; font-size:0.8rem;">
                <div>{layer_names[i]}</div>
                <div style="font-size:1.2rem;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # K线图 + 成交量
    st.markdown(f"#### 📉 {selected_symbol} K线 ({main_period}) + 成交量")
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
        if entry_signal != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            arrow_text = "▲ 多" if entry_signal == 1 else "▼ 空"
            arrow_color = "green" if entry_signal == 1 else "red"
            fig.add_annotation(x=last_date, y=last_price * (1.02 if entry_signal==1 else 0.98),
                               text=arrow_text, showarrow=True, arrowhead=2, arrowcolor=arrow_color, font=dict(size=10))
        # RSI
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple", width=1), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        # 成交量
        colors_vol = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['日期'], y=df['volume'], name="成交量", marker_color=colors_vol, showlegend=False), row=3, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("K线数据不可用")

with col_right:
    # 交易信号卡片
    st.markdown("#### 📡 交易信号")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
    st.markdown(f'<div class="ai-box">{dir_map[entry_signal]}<br>强度 {five_total}/100</div>', unsafe_allow_html=True)

    # 入场条件
    st.markdown("#### 🔍 入场条件")
    cond1 = "✅" if five_total >= config['min_five_score'] else "❌"
    cond2 = "✅" if fear_greed <= config['fear_threshold'] else "❌"
    cond3 = "✅" if netflow >= config['netflow_required'] else "❌"
    cond4 = "✅" if whale >= config['whale_required'] else "❌"
    dir_icon = "✅" if five_dir != 0 else "❌"
    st.markdown(f"""
    <div style="font-size:0.85rem; line-height:1.4;">
        {cond1} 强度 ≥ {config['min_five_score']}<br>
        {cond2} 恐惧 ≤ {config['fear_threshold']}<br>
        {cond3} 净流入 ≥ {config['netflow_required']} ETH<br>
        {cond4} 大额转账 ≥ {config['whale_required']} 笔<br>
        {dir_icon} 方向明确 ({'多' if five_dir==1 else '空' if five_dir==-1 else '无'})
    </div>
    """, unsafe_allow_html=True)

    # 风险因子卡片
    st.markdown("#### ⚖️ 风险敞口")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("质量因子", f"{five_total/100:.2f}")
        st.metric("波动因子", f"{1.0 if atr_pct>0.8 else 0.5:.2f}")
    with col_r2:
        st.metric("回撤因子", f"{1.0 if drawdown<10 else 0.5:.2f}")
        st.metric("连亏因子", f"{1.0 if st.session_state.consecutive_losses<3 else 0.5:.2f}")
    st.metric("建议杠杆", f"{suggested_leverage:.1f}x")

    # 强平预警
    if entry_signal != 0 and liq_price is not None:
        liq_color = "danger" if distance_to_liq < 5 else "warning" if distance_to_liq < 10 else "normal"
        st.markdown(f"""
        <div style="background:#232734; padding:8px; border-radius:4px; margin-top:8px;">
            <span style="font-weight:bold;">⚠️ 强平预警</span><br>
            强平价: <span style="color:#FF5555;">${liq_price:.2f}</span><br>
            价差: <span style="color:{'red' if distance_to_liq<5 else 'orange' if distance_to_liq<10 else 'white'};">{distance_to_liq:.2f}%</span>
            { '🚨 极度危险！' if distance_to_liq < 5 else '⚠️ 注意风险' if distance_to_liq < 10 else '✅ 安全' }
        </div>
        """, unsafe_allow_html=True)

    # 当前价格
    st.metric("当前价格", f"${current_price:.2f}" if current_price else "N/A")

    # 资本监控
    with st.container():
        st.markdown('<div class="dashboard">', unsafe_allow_html=True)
        st.markdown("#### 💼 资本监控")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("账户余额", f"${st.session_state.account_balance:.2f}")
            st.metric("日盈亏", f"${st.session_state.daily_pnl:.2f}", delta_color="inverse")
        with col_c2:
            st.metric("当前回撤", f"{drawdown:.2f}%")
            st.metric("连续亏损", st.session_state.consecutive_losses)
        st.markdown('</div>', unsafe_allow_html=True)

    # 链上情绪
    with st.expander("🔗 链上情绪", expanded=False):
        st.write(f"交易所净流入: **{netflow:+.0f} {selected_symbol.split('/')[0]}** (模拟)")
        st.write(f"大额转账: **{whale}** 笔 (模拟)")
        st.write(f"恐惧贪婪指数: **{fear_greed}**")

    # 自动化
    st.markdown("---")
    st.subheader("🤖 自动化")
    auto_enabled = st.checkbox("启用模拟自动跟随", value=st.session_state.auto_enabled)
    st.session_state.auto_enabled = auto_enabled

    now = datetime.now()
    # 自动交易逻辑（与原版相同，已支持双向）
    if auto_enabled and can_trade_flag and entry_signal != 0:
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
            st.success(f"✅ 自动开{st.session_state.auto_position['side']}仓 @ {current_price:.2f}")
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
                st.info(f"📉 平仓 {pos['side']}，盈亏: ${pnl:.2f}")
                st.session_state.auto_position = None

    if st.session_state.auto_position:
        pos = st.session_state.auto_position
        pnl = (current_price - pos['entry']) * (1.0 if pos['side']=='long' else -1.0) * pos['size']
        pnl_pct = (current_price - pos['entry']) / pos['entry'] * 100.0 * (1.0 if pos['side']=='long' else -1.0)
        liq_price_auto = liquidation_price(pos['entry'], 1 if pos['side']=='long' else -1, pos['leverage'])
        distance_auto = abs(current_price - liq_price_auto) / current_price * 100.0
        color_class = "profit" if pnl >= 0 else "loss"
        st.markdown(f"""
        <div class="metric" style="padding:8px;">
            <h4 style="font-size:1rem;">自动模拟持仓</h4>
            <p style="font-size:0.9rem;">方向: {'多' if pos['side']=='long' else '空'} | 杠杆: {pos['leverage']:.1f}x</p>
            <p style="font-size:0.9rem;">开仓: ${pos['entry']:.2f} ({pos['time'].strftime('%H:%M')})</p>
            <p style="font-size:1rem;" class="{color_class}">盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p style="font-size:0.9rem;">强平价: <span class="warning">${liq_price_auto:.2f}</span> (距 {distance_auto:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("手动平仓", key="auto_close"):
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
            st.success(f"平仓，盈亏: ${pnl:.2f}")
            st.session_state.auto_position = None
            st.rerun()
    else:
        if auto_enabled:
            if can_trade_flag:
                st.info("等待信号开仓")
            else:
                st.warning("交易暂停中")

    # 交易日誌
    with st.expander("📋 交易日誌", expanded=False):
        if st.session_state.trade_log:
            st.dataframe(pd.DataFrame(st.session_state.trade_log), use_container_width=True, height=150)
        else:
            st.info("暂无交易记录")

    # 信号历史
    if entry_signal != 0:
        current_dir = "多" if entry_signal == 1 else "空"
        if not st.session_state.signal_history or st.session_state.signal_history[-1]['方向'] != current_dir:
            st.session_state.signal_history.append({
                '时间': now.strftime("%H:%M"),
                '方向': current_dir,
                '市场': market_mode,
                '多因子强度': five_total
            })
            st.session_state.signal_history = st.session_state.signal_history[-20:]

    with st.expander("📜 信号历史", expanded=False):
        if st.session_state.signal_history:
            st.dataframe(pd.DataFrame(st.session_state.signal_history), use_container_width=True, height=150)
        else:
            st.info("暂无历史信号")
