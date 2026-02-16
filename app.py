# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 神境完美版 v4.0
顺势交易｜风险回报≥1:2｜动态仓位｜多重过滤｜移动止损｜一键平仓｜强度评分
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
BASE_RISK = 0.02                     # 单笔风险2%
MAX_LEVERAGE = 10.0                  # 最大杠杆10倍
DAILY_LOSS_LIMIT = 300.0
MIN_ATR_PCT = 0.8                     # 最小波动率要求
TP_MIN_RATIO = 2.0                    # 最小盈亏比 1:2

LEVERAGE_MODES = {
    "低倍试炼 (3-5x)": (3, 5),
    "中倍试炼 (5-8x)": (5, 8),
    "高倍神级 (8-10x)": (8, 10)
}

# 重要事件日期（示例）
EVENT_DATES = ["2026-02-20", "2026-03-15"]

# ==================== 数据获取器 ====================
class DataFetcher:
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = SYMBOLS
        self.symbols = symbols
        self.periods = ['15m', '1h', '4h', '1d']
        self.limit = 500
        self.timeout = 10
        self.exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
        self.fng_url = "https://api.alternative.me/fng/"
        self.chain_netflow = 5234   # 模拟值
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
        df['ema200'] = df['close'].ewm(span=200).mean()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100.0
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()

        # 成交量均量
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_surge'] = df['volume'] > df['volume_ma20'] * 1.2

        # 买卖信号（用于历史标注）
        df['buy_signal'] = (df['rsi'] < 30) & (df['close'] > df['ma20']) | (df['rsi'].shift(1) < 30) & (df['close'] > df['ma20'])
        df['sell_signal'] = (df['rsi'] > 70) & (df['close'] < df['ma60']) | (df['rsi'].shift(1) > 70) & (df['close'] < df['ma60'])
        return df


# ==================== 多周期趋势判断 ====================
def check_multiframe_trend(data_dict):
    """
    检查15m、1h、4h趋势是否一致
    返回：1（多头一致），-1（空头一致），0（不一致）
    """
    trends = []
    for tf in ['15m', '1h', '4h']:
        if tf not in data_dict:
            continue
        df = data_dict[tf]
        if df.empty or len(df) < 20:
            continue
        last = df.iloc[-1]
        if last['close'] > last['ema50'] > last['ema200'] and last['adx'] > 20:
            trends.append(1)
        elif last['close'] < last['ema50'] < last['ema200'] and last['adx'] > 20:
            trends.append(-1)
        else:
            trends.append(0)
    if all(t == 1 for t in trends):
        return 1
    if all(t == -1 for t in trends):
        return -1
    return 0


def is_trend_up(df):
    """严格上升趋势定义：价格 > EMA200 且 MACD 在零轴上且金叉"""
    last = df.iloc[-1]
    return last['close'] > last['ema200'] and last['macd'] > last['macd_signal'] and last['macd'] > 0

def is_trend_down(df):
    """严格下降趋势定义：价格 < EMA200 且 MACD 在零轴下且死叉"""
    last = df.iloc[-1]
    return last['close'] < last['ema200'] and last['macd'] < last['macd_signal'] and last['macd'] < 0


def evaluate_market(df_dict):
    if df_dict is None or '15m' not in df_dict:
        return "数据不足", 0.0, 0.0
    df = df_dict['15m']
    if df.empty:
        return "数据不足", 0.0, 0.0
    last = df.iloc[-1]

    if is_trend_up(df):
        return "上升趋势", last['atr_pct'], last['adx']
    elif is_trend_down(df):
        return "下降趋势", last['atr_pct'], last['adx']
    elif last['adx'] < 25:
        return "震荡", last['atr_pct'], last['adx']
    else:
        return "不明朗", last['atr_pct'], last['adx']


def get_mode_config(mode):
    if mode == "稳健":
        return {
            'min_five_score': 60,
            'fear_threshold': 20,
            'netflow_required': 5000,
            'whale_required': 100,
            'stop_atr': 1.8,
            'tp_min_ratio': 2.5,
            'position_pct': lambda fear: 0.5 if fear <= 10 else (0.3 if fear <= 20 else 0.0),
        }
    elif mode == "无敌":
        return {
            'min_five_score': 70,
            'fear_threshold': 15,
            'netflow_required': 6000,
            'whale_required': 120,
            'stop_atr': 2.0,
            'tp_min_ratio': 3.0,
            'position_pct': lambda fear: 0.8 if fear <= 10 else (0.5 if fear <= 20 else 0.0),
        }
    elif mode == "神级":
        return {
            'min_five_score': 80,
            'fear_threshold': 8,
            'netflow_required': 8000,
            'whale_required': 150,
            'stop_atr': 2.2,
            'tp_min_ratio': 4.0,
            'position_pct': lambda fear: 1.0 if fear <= 8 else (0.6 if fear <= 15 else 0.0),
        }
    else:
        return get_mode_config("稳健")


def five_layer_score(df_dict, fear_greed, chain_netflow, chain_whale):
    # 简化多因子评分（仅供参考）
    if df_dict is None or '15m' not in df_dict:
        return 0, 0, {}
    df_15m = df_dict['15m']
    last = df_15m.iloc[-1]
    trend_score = 20 if is_trend_up(df_15m) else 0
    multi_score = 20 if check_multiframe_trend(df_dict) != 0 else 0
    fund_score = 20 if chain_netflow > 5000 else 0
    chain_score = 15 if fear_greed < 30 else 0
    momentum_score = 15 if last['macd_diff'] > 0 else 0
    total = trend_score + multi_score + fund_score + chain_score + momentum_score
    final_dir = 1 if total >= 60 else -1 if total <= -60 else 0
    layer_scores = {"趋势": trend_score, "多周期": multi_score, "资金": fund_score, "链上": chain_score, "动量": momentum_score}
    return final_dir, total, layer_scores


def generate_entry_signal(data_dict, config, btc_trend=None):
    """
    严格的多因子过滤信号，返回 (方向, 强度评分)
    强度评分 = 满足条件数 / 总条件数 * 100，只有 >=70 才允许开仓
    """
    if data_dict is None or '15m' not in data_dict:
        return 0, 0

    df_15m = data_dict['15m']
    last = df_15m.iloc[-1]
    conditions_met = 0
    total_conditions = 0

    # 1. 趋势过滤（硬性）
    if is_trend_up(df_15m):
        trend_dir = 1
    elif is_trend_down(df_15m):
        trend_dir = -1
    else:
        trend_dir = 0

    if trend_dir == 0:
        return 0, 0
    conditions_met += 1
    total_conditions += 1

    # 2. 多周期趋势一致
    mf_trend = check_multiframe_trend(data_dict)
    if mf_trend == trend_dir:
        conditions_met += 1
    total_conditions += 1

    # 3. 波动率足够
    if last['atr_pct'] >= MIN_ATR_PCT:
        conditions_met += 1
    total_conditions += 1

    # 4. 成交量放量
    if last['volume_surge']:
        conditions_met += 1
    total_conditions += 1

    # 5. RSI过滤
    if trend_dir == 1 and last['rsi'] > 50:
        conditions_met += 1
    elif trend_dir == -1 and last['rsi'] < 50:
        conditions_met += 1
    total_conditions += 1

    # 6. 大盘BTC同步（如果提供）
    if btc_trend is not None:
        total_conditions += 1
        if btc_trend == trend_dir:
            conditions_met += 1

    # 计算强度评分
    strength = int(conditions_met / total_conditions * 100)

    # 只有强度 >=70 才发信号
    if strength >= 70:
        return trend_dir, strength
    else:
        return 0, strength


def calculate_stops(entry_price, side, atr_value, stop_atr):
    """
    计算止损止盈，确保盈亏比 ≥ TP_MIN_RATIO
    """
    stop_distance = stop_atr * atr_value
    take_distance = stop_distance * TP_MIN_RATIO
    if side == 1:
        stop = entry_price - stop_distance
        take = entry_price + take_distance
    else:
        stop = entry_price + stop_distance
        take = entry_price - take_distance
    return stop, take, take_distance / stop_distance


def calculate_position_size(balance, entry_price, stop_price, leverage):
    """
    动态仓位：风险金额 = 账户余额 × BASE_RISK
    """
    risk_amount = balance * BASE_RISK
    stop_distance_pct = abs(entry_price - stop_price) / entry_price
    if stop_distance_pct == 0:
        return 0.0
    position_value = risk_amount / stop_distance_pct
    max_position = balance * leverage
    position_value = min(position_value, max_position)
    quantity = position_value / entry_price
    return round(quantity, 3)


def liquidation_price(entry_price, side, leverage):
    if side == 1:
        return entry_price * (1 - 1.0 / leverage)
    else:
        return entry_price * (1 + 1.0 / leverage)


def update_trailing_stop(position, current_price):
    """
    移动止损与保本机制
    - 盈利 ≥1% 时止损移至入场价（保本）
    - 之后每盈利 0.8%，止损上移 0.5%
    """
    if position is None:
        return position
    entry = position['entry']
    side = position['side']
    current_stop = position['stop']
    if side == 'long':
        pnl_pct = (current_price - entry) / entry * 100
        if pnl_pct >= 1.0 and current_stop < entry:
            position['stop'] = entry
        elif pnl_pct > 1.0:
            trailing_step = int((pnl_pct - 1.0) / 0.8) * 0.5
            new_stop = entry + (pnl_pct - 1.0 - trailing_step) / 100 * entry
            if new_stop > current_stop:
                position['stop'] = new_stop
    else:  # short
        pnl_pct = (entry - current_price) / entry * 100
        if pnl_pct >= 1.0 and current_stop > entry:
            position['stop'] = entry
        elif pnl_pct > 1.0:
            trailing_step = int((pnl_pct - 1.0) / 0.8) * 0.5
            new_stop = entry - (pnl_pct - 1.0 - trailing_step) / 100 * entry
            if new_stop < current_stop:
                position['stop'] = new_stop
    return position


def is_event_day():
    today = datetime.now().strftime("%Y-%m-%d")
    return today in EVENT_DATES


def send_telegram_message(message):
    token = st.session_state.get("telegram_token", "")
    chat_id = st.session_state.get("telegram_chat_id", "")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        try:
            requests.post(url, json=data, timeout=5)
        except:
            pass


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
        st.session_state.auto_enabled = True
    if 'auto_position' not in st.session_state:
        st.session_state.auto_position = None
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []
    if 'pause_until' not in st.session_state:
        st.session_state.pause_until = None  # 紧急平仓后暂停


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
    if st.session_state.daily_loss_triggered:
        return False
    if st.session_state.pause_until and datetime.now() < st.session_state.pause_until:
        return False
    return True


# ==================== 主界面 ====================
st.set_page_config(page_title="终极量化终端 · 神境完美版 v4.0", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: white; font-size: 0.85rem; }
.card { background: #1A1D27; border-radius: 4px; padding: 10px; margin-bottom: 8px; border-left: 4px solid #00F5A0; }
.card-header { font-size: 0.9rem; color: #8A8F9C; margin-bottom: 6px; }
.metric-label { font-size: 0.75rem; color: #8A8F9C; }
.metric-value { font-size: 1.1rem; font-weight: bold; }
.risk-factor { display: flex; justify-content: space-between; font-size: 0.9rem; padding: 2px 0; }
.risk-line { border-top: 1px solid #333; margin: 6px 0; }
.eligibility-blocked { color: #FF5555; font-weight: bold; }
.eligibility-active { color: #00F5A0; font-weight: bold; }
.trade-plan { background: #232734; padding: 8px; border-radius: 4px; margin-top: 8px; border-left: 4px solid #FFAA00; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 终极量化终端 · 神境完美版 v4.0")
st.caption("顺势交易｜风险回报≥1:2｜动态仓位｜多重过滤｜移动止损｜一键平仓｜强度评分")

init_risk_state()

with st.sidebar:
    st.header("⚙️ 市场设置")
    selected_symbol = st.selectbox("交易品种", SYMBOLS, index=0)
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

    st.markdown("---")
    st.subheader("📲 Telegram通知")
    use_telegram = st.checkbox("启用Telegram通知", value=False)
    if use_telegram:
        bot_token = st.text_input("Bot Token", type="password")
        chat_id = st.text_input("Chat ID")
        if bot_token and chat_id:
            st.session_state.telegram_token = bot_token
            st.session_state.telegram_chat_id = chat_id
        else:
            st.warning("请输入Token和Chat ID")
    else:
        st.session_state.telegram_token = ""
        st.session_state.telegram_chat_id = ""

    st.markdown("---")
    st.subheader("🤖 自动交易")
    auto_enabled = st.checkbox("启用自动跟随", value=st.session_state.auto_enabled)
    st.session_state.auto_enabled = auto_enabled

    # 一键紧急平仓（红色醒目）
    if st.button("🚨 一键紧急平仓", type="primary", use_container_width=True):
        if st.session_state.auto_position:
            st.session_state.auto_position = None
            st.session_state.pause_until = datetime.now() + timedelta(hours=2)
            st.success("已平仓，策略暂停2小时")
            st.rerun()
        else:
            st.warning("当前无持仓")

# 获取数据
with st.spinner("获取市场数据..."):
    fetcher = DataFetcher(symbols=SYMBOLS)
    all_data = fetcher.fetch_all()

if selected_symbol not in all_data or all_data[selected_symbol]["data_dict"] is None:
    st.error(f"❌ 品种 {selected_symbol} 数据不可用")
    st.stop()

data = all_data[selected_symbol]
data_dict = data["data_dict"]
current_price = data["current_price"]
fear_greed = data["fear_greed"]
source_display = data["source"]
netflow = data["chain_netflow"]
whale = data["chain_whale"]

# 获取BTC趋势（用于联动）
btc_data = all_data.get("BTC/USDT")
btc_trend = None
if btc_data and btc_data["data_dict"] is not None:
    btc_df = btc_data["data_dict"]['15m']
    if not btc_df.empty:
        if is_trend_up(btc_df):
            btc_trend = 1
        elif is_trend_down(btc_df):
            btc_trend = -1
        else:
            btc_trend = 0

# 多因子评分
five_dir, five_total, layer_scores = five_layer_score(data_dict, fear_greed, netflow, whale)
market_mode, atr_pct, adx = evaluate_market(data_dict)

# 自动模式
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
entry_signal, signal_strength = generate_entry_signal(data_dict, config, btc_trend)

atr_value = data_dict['15m']['atr'].iloc[-1] if '15m' in data_dict else 0.0
position_pct = config['position_pct'](fear_greed)
suggested_leverage = (min_lev + max_lev) / 2

stop_loss = take_profit = risk_reward = None
position_size = 0.0
if entry_signal != 0 and atr_value > 0:
    stop_loss, take_profit, risk_reward = calculate_stops(current_price, entry_signal, atr_value, config['stop_atr'])
    # 确保盈亏比至少为 TP_MIN_RATIO（已在 calculate_stops 中保证）
    position_size = calculate_position_size(
        st.session_state.account_balance,
        current_price,
        stop_loss,
        suggested_leverage
    )

# 风险因子（简化版）
F_quality = five_total / 100.0 if five_total else 0.0
F_volatility = 1.0 if atr_pct > 0.8 else 0.5 if atr_pct else 0.5
drawdown = update_risk_state(0.0, st.session_state.account_balance + st.session_state.daily_pnl, st.session_state.daily_pnl)
F_drawdown = 1.0 if drawdown < 10 else 0.5 if drawdown else 1.0
F_loss_streak = 1.0 if st.session_state.consecutive_losses < 3 else 0.5
R_final = BASE_RISK * F_quality * F_volatility * F_drawdown * F_loss_streak
R_final = max(0.001, min(0.02, R_final))
capital_at_risk = st.session_state.account_balance * R_final

# 强平价格
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
eligibility = "活跃" if can_trade_flag and entry_signal != 0 else "禁止"

# 事件过滤
if is_event_day():
    eligibility = "事件暂停"
    entry_signal = 0

# 主布局
col_left, col_right = st.columns([1.4, 1.6])

with col_left:
    # ① 全球宏观
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">① 全球宏观</div>', unsafe_allow_html=True)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: st.markdown(f"<div class='metric-label'>市场状态</div><div class='metric-value'>{market_mode}</div>", unsafe_allow_html=True)
    with col_m2: st.markdown(f"<div class='metric-label'>波动率(ATR)</div><div class='metric-value'>{atr_pct:.2f}%</div>", unsafe_allow_html=True)
    with col_m3: st.markdown(f"<div class='metric-label'>趋势强度</div><div class='metric-value'>{adx:.1f}</div>", unsafe_allow_html=True)
    with col_m4: st.markdown(f"<div class='metric-label'>恐惧指数</div><div class='metric-value'>{fear_greed}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:4px;'>数据源: {source_display} | 大盘BTC趋势: {'↑' if btc_trend==1 else '↓' if btc_trend==-1 else '↔'}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ② 策略概况
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">② 策略概况</div>', unsafe_allow_html=True)
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1: st.markdown(f"<div class='metric-label'>策略模式</div><div class='metric-value'>{mode}</div>", unsafe_allow_html=True)
    with col_s2: st.markdown(f"<div class='metric-label'>杠杆范围</div><div class='metric-value'>{min_lev:.0f}x–{max_lev:.0f}x</div>", unsafe_allow_html=True)
    with col_s3: st.markdown(f"<div class='metric-label'>盈亏比</div><div class='metric-value'>1:{TP_MIN_RATIO}</div>", unsafe_allow_html=True)
    with col_s4: st.markdown(f"<div class='metric-label'>日亏损限额</div><div class='metric-value'>{DAILY_LOSS_LIMIT:.0f} USDT</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ③ 信号引擎 + 入场条件 + 强度评分
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">③ 信号引擎</div>', unsafe_allow_html=True)
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    with col_i1: st.markdown(f"<div class='metric-label'>品种</div><div class='metric-value'>{selected_symbol}</div>", unsafe_allow_html=True)
    with col_i2: st.markdown(f"<div class='metric-label'>周期</div><div class='metric-value'>{main_period}</div>", unsafe_allow_html=True)
    with col_i3:
        status = "等待" if entry_signal == 0 else ("做多" if entry_signal == 1 else "做空")
        st.markdown(f"<div class='metric-label'>信号状态</div><div class='metric-value'>{status}</div>", unsafe_allow_html=True)
    with col_i4: st.markdown(f"<div class='metric-label'>强度</div><div class='metric-value'>{signal_strength}/100</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:6px;'><span class='metric-label'>执行资格:</span> <span class='eligibility-{'active' if eligibility=='活跃' else 'blocked'}'>{eligibility}</span></div>", unsafe_allow_html=True)

    # 入场条件明细
    st.markdown("#### 入场条件检查")
    cond1 = "✅" if is_trend_up(data_dict['15m']) or is_trend_down(data_dict['15m']) else "❌"
    cond2 = "✅" if check_multiframe_trend(data_dict) != 0 else "❌"
    cond3 = "✅" if atr_pct >= MIN_ATR_PCT else "❌"
    cond4 = "✅" if data_dict['15m'].iloc[-1]['volume_surge'] else "❌"
    cond5 = "✅" if (entry_signal == 1 and data_dict['15m'].iloc[-1]['rsi'] > 50) or (entry_signal == -1 and data_dict['15m'].iloc[-1]['rsi'] < 50) else "❌"
    cond6 = "✅" if btc_trend == entry_signal else "❌" if btc_trend is not None else "⚪未启用"
    st.markdown(f"""
    <div style="font-size:0.8rem; line-height:1.4;">
        {cond1} 严格趋势过滤<br>
        {cond2} 多周期趋势一致<br>
        {cond3} 波动率 ≥ {MIN_ATR_PCT}%<br>
        {cond4} 成交量放量<br>
        {cond5} RSI方向匹配<br>
        {cond6} 大盘BTC同步
    </div>
    """, unsafe_allow_html=True)

    # 交易计划
    if entry_signal != 0 and stop_loss and take_profit:
        st.markdown("#### 📝 交易计划")
        st.markdown(f"""
        <div class="trade-plan">
            <p>入场价: <span style="color:#00F5A0;">${current_price:.2f}</span></p>
            <p>止损价: <span style="color:#FF5555;">${stop_loss:.2f}</span> (亏损 {abs(current_price-stop_loss)/current_price*100:.2f}%)</p>
            <p>止盈价: <span style="color:#00F5A0;">${take_profit:.2f}</span> (盈亏比 {risk_reward:.2f})</p>
            <p>建议仓位: {position_size} {selected_symbol.split('/')[0]}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ④ 风险引擎
    st.markdown('<div class="card" style="border-left-color: #FFAA00;">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">④ 风险引擎</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-factor"><span class="factor-name">F_quality</span><span class="factor-value">{F_quality:.2f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-factor"><span class="factor-name">F_volatility</span><span class="factor-value">{F_volatility:.2f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-factor"><span class="factor-name">F_drawdown</span><span class="factor-value">{F_drawdown:.2f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-factor"><span class="factor-name">F_loss_streak</span><span class="factor-value">{F_loss_streak:.2f}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-line"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-factor"><span class="factor-name">R_final</span><span class="factor-value">{R_final*100:.2f}%</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="display:flex; justify-content:space-between; margin-top:8px;">'
                f'<div><span class="metric-label">资本风险</span><br><span class="metric-value">{capital_at_risk:.1f} USDT</span></div>'
                f'<div><span class="metric-label">建议杠杆</span><br><span class="metric-value">{suggested_leverage:.1f}x</span></div>'
                f'<div><span class="metric-label">爆仓价</span><br><span class="metric-value">{"${:.2f}".format(liq_price) if liq_price else "—"}</span></div>'
                '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ⑤ 资本状态
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⑤ 资本状态</div>', unsafe_allow_html=True)
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1: st.markdown(f"<div class='metric-label'>账户余额</div><div class='metric-value'>{st.session_state.account_balance:.0f} USDT</div>", unsafe_allow_html=True)
    with col_c2: st.markdown(f"<div class='metric-label'>日盈亏</div><div class='metric-value'>{st.session_state.daily_pnl:.1f}</div>", unsafe_allow_html=True)
    with col_c3: st.markdown(f"<div class='metric-label'>当前回撤</div><div class='metric-value'>{drawdown:.2f}%</div>", unsafe_allow_html=True)
    with col_c4: st.markdown(f"<div class='metric-label'>连亏次数</div><div class='metric-value'>{st.session_state.consecutive_losses}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ⑥ 链上情绪
    with st.expander("🔗 链上情绪", expanded=False):
        st.write(f"交易所净流入: **{netflow:+.0f} {selected_symbol.split('/')[0]}** (模拟)")
        st.write(f"大额转账: **{whale}** 笔 (模拟)")
        st.write(f"恐惧贪婪指数: **{fear_greed}**")

    # ⑦ 市场监控
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⑦ 市场监控</div>', unsafe_allow_html=True)
    monitor_data = []
    for sym in SYMBOLS:
        if sym in all_data and all_data[sym]["data_dict"] is not None:
            d = all_data[sym]["data_dict"]
            f = all_data[sym]["fear_greed"]
            n = all_data[sym]["chain_netflow"]
            w = all_data[sym]["chain_whale"]
            _, total, _ = five_layer_score(d, f, n, w)
            status = "活跃" if total >= 60 else "中性"
            monitor_data.append([sym, total, status])
        else:
            monitor_data.append([sym, "—", "不可用"])
    st.table(pd.DataFrame(monitor_data, columns=["品种", "强度", "状态"]))
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
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
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema50'], line=dict(color="orange", width=1), name="EMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema200'], line=dict(color="blue", width=1), name="EMA200"), row=1, col=1)
        fig.add_hline(y=current_price, line_dash="dot", line_color="white", annotation_text=f"现价 {current_price:.2f}", row=1, col=1)

        # 止损止盈线
        if entry_signal != 0 and stop_loss and take_profit:
            fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"止损 {stop_loss:.2f}", row=1, col=1)
            fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text=f"止盈 {take_profit:.2f}", row=1, col=1)

        # 历史买卖信号标注
        buy_signals = df[df['buy_signal'] == True]
        for idx, row in buy_signals.iterrows():
            fig.add_annotation(x=row['日期'], y=row['low'] * 0.99, text="▲", showarrow=False, font=dict(size=12, color="#00F5A0"), row=1, col=1)
        sell_signals = df[df['sell_signal'] == True]
        for idx, row in sell_signals.iterrows():
            fig.add_annotation(x=row['日期'], y=row['high'] * 1.01, text="▼", showarrow=False, font=dict(size=12, color="#FF5555"), row=1, col=1)

        # 当前信号箭头
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
        latest_rsi = df['rsi'].iloc[-1]
        fig.add_annotation(x=df['日期'].iloc[-1], y=latest_rsi, text=f"RSI: {latest_rsi:.1f}", showarrow=False, xanchor='left', row=2, col=1, font=dict(size=9, color="white"))

        # 成交量
        colors_vol = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['日期'], y=df['volume'], name="成交量", marker_color=colors_vol, showlegend=False), row=3, col=1)

        fig.update_layout(hovermode='x unified', template="plotly_dark", xaxis_rangeslider_visible=False, height=600, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        latest_macd = df['macd'].iloc[-1]
        latest_signal = df['macd_signal'].iloc[-1]
        st.markdown(f"<span style='font-size:0.8rem;'>MACD: {latest_macd:.2f} | Signal: {latest_signal:.2f}</span>", unsafe_allow_html=True)
    else:
        st.warning("K线数据不可用")

    # 执行日志
    with st.expander("📋 执行日志"):
        tab1, tab2 = st.tabs(["交易记录", "信号历史"])
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

# ==================== 自动交易逻辑 ====================
now = datetime.now()
if can_trade_flag and st.session_state.get('auto_enabled', False) and entry_signal != 0 and eligibility == "活跃":
    # 更新移动止损（如果已有持仓）
    if st.session_state.auto_position is not None:
        st.session_state.auto_position = update_trailing_stop(st.session_state.auto_position, current_price)

    # 检查现有持仓是否需要平仓
    if st.session_state.auto_position is not None:
        pos = st.session_state.auto_position
        # 检查是否触及止损或止盈
        if (pos['side'] == 'long' and current_price <= pos['stop']) or \
           (pos['side'] == 'short' and current_price >= pos['stop']):
            # 止损
            pnl = (current_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - current_price) * pos['size']
            pnl_pct = pnl / (pos['entry'] * pos['size']) * 100.0
            update_risk_state(pnl, st.session_state.account_balance + st.session_state.daily_pnl, st.session_state.daily_pnl)
            st.session_state.trade_log.append({
                '开仓时间': pos['time'].strftime('%H:%M'),
                '方向': pos['side'],
                '开仓价': f"{pos['entry']:.2f}",
                '平仓时间': now.strftime('%H:%M'),
                '平仓价': f"{current_price:.2f}",
                '盈亏': f"{pnl:.2f}",
                '盈亏%': f"{pnl_pct:.1f}%",
                '类型': '止损'
            })
            st.session_state.balance_history.append(st.session_state.account_balance + st.session_state.daily_pnl)
            st.session_state.auto_position = None
            if use_telegram:
                send_telegram_message(f"🔴 止损平仓，盈亏: ${pnl:.2f} ({pnl_pct:.1f}%)")
        elif (pos['side'] == 'long' and current_price >= pos['take']) or \
             (pos['side'] == 'short' and current_price <= pos['take']):
            # 止盈
            pnl = (current_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - current_price) * pos['size']
            pnl_pct = pnl / (pos['entry'] * pos['size']) * 100.0
            update_risk_state(pnl, st.session_state.account_balance + st.session_state.daily_pnl, st.session_state.daily_pnl)
            st.session_state.trade_log.append({
                '开仓时间': pos['time'].strftime('%H:%M'),
                '方向': pos['side'],
                '开仓价': f"{pos['entry']:.2f}",
                '平仓时间': now.strftime('%H:%M'),
                '平仓价': f"{current_price:.2f}",
                '盈亏': f"{pnl:.2f}",
                '盈亏%': f"{pnl_pct:.1f}%",
                '类型': '止盈'
            })
            st.session_state.balance_history.append(st.session_state.account_balance + st.session_state.daily_pnl)
            st.session_state.auto_position = None
            if use_telegram:
                send_telegram_message(f"🟢 止盈平仓，盈亏: ${pnl:.2f} ({pnl_pct:.1f}%)")
        elif (pos['side'] == 'long' and entry_signal == -1) or (pos['side'] == 'short' and entry_signal == 1):
            # 反向信号平仓
            pnl = (current_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - current_price) * pos['size']
            pnl_pct = pnl / (pos['entry'] * pos['size']) * 100.0
            update_risk_state(pnl, st.session_state.account_balance + st.session_state.daily_pnl, st.session_state.daily_pnl)
            st.session_state.trade_log.append({
                '开仓时间': pos['time'].strftime('%H:%M'),
                '方向': pos['side'],
                '开仓价': f"{pos['entry']:.2f}",
                '平仓时间': now.strftime('%H:%M'),
                '平仓价': f"{current_price:.2f}",
                '盈亏': f"{pnl:.2f}",
                '盈亏%': f"{pnl_pct:.1f}%",
                '类型': '反向'
            })
            st.session_state.balance_history.append(st.session_state.account_balance + st.session_state.daily_pnl)
            st.session_state.auto_position = None
            if use_telegram:
                send_telegram_message(f"↩️ 反向平仓，盈亏: ${pnl:.2f} ({pnl_pct:.1f}%)")

    # 开新仓
    if st.session_state.auto_position is None and entry_signal != 0:
        st.session_state.auto_position = {
            'side': 'long' if entry_signal == 1 else 'short',
            'entry': current_price,
            'time': now,
            'leverage': suggested_leverage,
            'stop': stop_loss,
            'take': take_profit,
            'size': position_size
        }
        st.session_state.signal_history.append({
            '时间': now.strftime("%H:%M"),
            '方向': '多' if entry_signal == 1 else '空',
            '市场': market_mode,
            '多因子强度': five_total,
            '强度评分': signal_strength
        })
        if use_telegram and st.session_state.telegram_token:
            msg = f"🚀 <b>开仓信号</b>\n品种: {selected_symbol}\n方向: {'多' if entry_signal==1 else '空'}\n价格: ${current_price:.2f}\n杠杆: {suggested_leverage:.1f}x"
            send_telegram_message(msg)
