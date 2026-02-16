# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 最终完美版
智能AI分析｜严格趋势过滤｜动态仓位｜移动止损｜一键平仓｜Telegram通知
数据源：MEXC + Alternative.me + 模拟链上（可替换真实API）
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
MIN_ATR_PCT = 0.8                    # 最小波动率要求
TP_MIN_RATIO = 2.0                   # 最小盈亏比 1:2

LEVERAGE_MODES = {
    "低倍试炼 (3-5x)": (3, 5),
    "中倍试炼 (5-8x)": (5, 8),
    "高倍神级 (8-10x)": (8, 10)
}

# 尝试加载AI模型（XGBoost）
AI_MODEL = None
if os.path.exists('eth_ai_model.pkl'):
    try:
        AI_MODEL = joblib.load('eth_ai_model.pkl')
    except Exception as e:
        st.sidebar.warning(f"AI模型加载失败: {e}")

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
            if not ohlcv:
                return None, None
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
                if df is not None and not df.empty:
                    data_dict[period] = self._add_indicators(df)
                    price_sources.append(src)
                else:
                    data_ok = False
            if data_ok and data_dict:
                all_data[symbol] = {
                    "data_dict": data_dict,
                    "current_price": float(data_dict['15m']['close'].iloc[-1]) if '15m' in data_dict else None,
                    "source": price_sources[0] if price_sources else "MEXC",
                    "fear_greed": fear_greed,
                    "chain_netflow": self.chain_netflow,
                    "chain_whale": self.chain_whale,
                }
            else:
                all_data[symbol] = None
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
        df['buy_signal'] = (df['rsi'] < 30) & (df['close'] > df['ma20'])
        df['sell_signal'] = (df['rsi'] > 70) & (df['close'] < df['ma60'])
        return df


# ==================== 趋势判断 ====================
def is_uptrend(df):
    """上升趋势：价格 > EMA200 且 MACD 在零轴上且金叉"""
    if df is None or df.empty:
        return False
    last = df.iloc[-1]
    try:
        return last['close'] > last['ema200'] and last['macd'] > last['macd_signal'] and last['macd'] > 0
    except:
        return False

def is_downtrend(df):
    """下降趋势：价格 < EMA200 且 MACD 在零轴下且死叉"""
    if df is None or df.empty:
        return False
    last = df.iloc[-1]
    try:
        return last['close'] < last['ema200'] and last['macd'] < last['macd_signal'] and last['macd'] < 0
    except:
        return False

def evaluate_market(df):
    if df is None or df.empty:
        return "数据不足", 0.0, 0.0
    last = df.iloc[-1]
    atr_pct = last['atr_pct'] if not pd.isna(last['atr_pct']) else 0.0
    adx = last['adx'] if not pd.isna(last['adx']) else 0.0
    if is_uptrend(df):
        return "上升趋势", atr_pct, adx
    if is_downtrend(df):
        return "下降趋势", atr_pct, adx
    if adx < 25:
        return "震荡", atr_pct, adx
    return "不明朗", atr_pct, adx


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
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        try:
            if last['close'] > last['ema50'] > last['ema200'] and last['adx'] > 20:
                trends.append(1)
            elif last['close'] < last['ema50'] < last['ema200'] and last['adx'] > 20:
                trends.append(-1)
            else:
                trends.append(0)
        except:
            trends.append(0)
    if len(trends) < 3:
        return 0
    if all(t == 1 for t in trends):
        return 1
    if all(t == -1 for t in trends):
        return -1
    return 0


def get_ai_prediction(df):
    """使用加载的AI模型预测未来方向概率"""
    if AI_MODEL is None or df is None or df.empty:
        return None
    try:
        last = df.iloc[-1]
        features = [
            last['rsi'],
            last['ma20'],
            last['ma60'],
            last['macd'],
            last['macd_signal'],
            last['atr_pct'],
            last['adx']
        ]
        prob = AI_MODEL.predict_proba([features])[0][1] * 100
        return prob
    except:
        return None


def generate_signal(data_dict, btc_trend=None):
    """
    严格信号：趋势明确 + 多周期一致 + 波动率足够 + 成交量放量 + RSI方向匹配 + (可选) BTC同步
    """
    if data_dict is None or '15m' not in data_dict:
        return 0

    df_15m = data_dict['15m']
    if df_15m is None or df_15m.empty:
        return 0

    last = df_15m.iloc[-1]
    # 1. 趋势方向
    if is_uptrend(df_15m):
        trend_dir = 1
    elif is_downtrend(df_15m):
        trend_dir = -1
    else:
        return 0

    # 2. 多周期一致
    mf_trend = check_multiframe_trend(data_dict)
    if mf_trend != trend_dir:
        return 0

    # 3. 波动率
    atr_pct = last['atr_pct']
    if pd.isna(atr_pct) or atr_pct < MIN_ATR_PCT:
        return 0

    # 4. 成交量
    if not last['volume_surge']:
        return 0

    # 5. RSI
    rsi = last['rsi']
    if pd.isna(rsi):
        return 0
    if trend_dir == 1 and rsi <= 50:
        return 0
    if trend_dir == -1 and rsi >= 50:
        return 0

    # 6. BTC同步（如果提供）
    if btc_trend is not None and btc_trend != trend_dir:
        return 0

    return trend_dir


def calculate_stops(entry_price, side, atr_value):
    stop_distance = 2.0 * atr_value          # 固定2倍ATR止损
    take_distance = stop_distance * TP_MIN_RATIO
    if side == 1:
        stop = entry_price - stop_distance
        take = entry_price + take_distance
    else:
        stop = entry_price + stop_distance
        take = entry_price - take_distance
    return stop, take, take_distance / stop_distance


def calculate_position_size(balance, entry_price, stop_price, leverage):
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


def init_state():
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 10000.0
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'peak_balance' not in st.session_state:
        st.session_state.peak_balance = 10000.0
    if 'consecutive_losses' not in st.session_state:
        st.session_state.consecutive_losses = 0
    if 'last_date' not in st.session_state:
        st.session_state.last_date = datetime.now().date()
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'auto_position' not in st.session_state:
        st.session_state.auto_position = None
    if 'auto_enabled' not in st.session_state:
        st.session_state.auto_enabled = True
    if 'pause_until' not in st.session_state:
        st.session_state.pause_until = None
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []


def update_daily_pnl(current_price, pos):
    if pos is None:
        return
    if pos['side'] == 'long':
        st.session_state.daily_pnl = (current_price - pos['entry']) * pos['size']
    else:
        st.session_state.daily_pnl = (pos['entry'] - current_price) * pos['size']


def can_trade():
    if st.session_state.pause_until and datetime.now() < st.session_state.pause_until:
        return False
    if st.session_state.daily_pnl < -DAILY_LOSS_LIMIT:
        return False
    return True


def update_trailing_stop(position, current_price):
    """
    移动止损：盈利≥1%时止损移至入场价，之后每盈利0.8%上移0.5%
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
            steps = int((pnl_pct - 1.0) / 0.8) * 0.5
            new_stop = entry + (pnl_pct - 1.0 - steps) / 100 * entry
            if new_stop > current_stop:
                position['stop'] = new_stop
    else:
        pnl_pct = (entry - current_price) / entry * 100
        if pnl_pct >= 1.0 and current_stop > entry:
            position['stop'] = entry
        elif pnl_pct > 1.0:
            steps = int((pnl_pct - 1.0) / 0.8) * 0.5
            new_stop = entry - (pnl_pct - 1.0 - steps) / 100 * entry
            if new_stop < current_stop:
                position['stop'] = new_stop
    return position


# ==================== 主界面 ====================
st.set_page_config(page_title="终极量化终端 · 最终完美版", layout="wide")
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

st.title("🏆 终极量化终端 · 最终完美版")
st.caption("智能AI分析｜严格趋势｜动态仓位｜移动止损｜一键平仓｜Telegram通知")

init_state()

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
    # 杠杆范围简化为与模式关联
    if manual_mode == "稳健":
        min_lev, max_lev = 3, 5
    elif manual_mode == "无敌":
        min_lev, max_lev = 5, 8
    else:
        min_lev, max_lev = 8, 10
    st.info(f"当前杠杆范围: {min_lev}x – {max_lev}x")

    st.markdown("---")
    st.subheader("📊 风险参数")
    account_balance = st.number_input("账户余额 (USDT)", value=st.session_state.account_balance, step=1000.0, format="%.2f")
    st.session_state.account_balance = account_balance
    st.number_input("日亏损限额 (USDT)", value=DAILY_LOSS_LIMIT, step=50.0, format="%.2f", disabled=True)

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
        st.session_state.telegram_token = ""

    st.markdown("---")
    st.subheader("🤖 自动交易")
    auto_enabled = st.checkbox("启用自动跟随", value=st.session_state.auto_enabled)
    st.session_state.auto_enabled = auto_enabled

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
    fetcher = DataFetcher()
    all_data = fetcher.fetch_all()

# 当前品种数据
data = all_data.get(selected_symbol)
if data is None or data["data_dict"] is None:
    st.error("❌ 数据不可用，请稍后重试")
    st.stop()

data_dict = data["data_dict"]
df_15m = data_dict.get('15m')
if df_15m is None or df_15m.empty:
    st.error("❌ 15分钟数据缺失")
    st.stop()

current_price = data["current_price"]
fear_greed = data["fear_greed"]
source = data["source"]
netflow = data["chain_netflow"]
whale = data["chain_whale"]

# 市场状态
market_mode, atr_pct, adx = evaluate_market(df_15m)

# BTC趋势（用于联动）
btc_data = all_data.get("BTC/USDT")
btc_trend = None
if btc_data and btc_data["data_dict"]:
    btc_df = btc_data["data_dict"].get('15m')
    if btc_df is not None and not btc_df.empty:
        if is_uptrend(btc_df):
            btc_trend = 1
        elif is_downtrend(btc_df):
            btc_trend = -1
        else:
            btc_trend = 0

# 生成信号
entry_signal = generate_signal(data_dict, btc_trend)

# AI预测
ai_prob = get_ai_prediction(df_15m)

# 止损止盈计算
atr_value = df_15m['atr'].iloc[-1] if not pd.isna(df_15m['atr'].iloc[-1]) else 0.0
suggested_leverage = (min_lev + max_lev) / 2
stop_loss = take_profit = risk_reward = None
position_size = 0.0
if entry_signal != 0 and atr_value > 0:
    stop_loss, take_profit, risk_reward = calculate_stops(current_price, entry_signal, atr_value)
    position_size = calculate_position_size(
        st.session_state.account_balance,
        current_price,
        stop_loss,
        suggested_leverage
    )

# 强平价格
if entry_signal == 1:
    liq_price = liquidation_price(current_price, 1, suggested_leverage)
elif entry_signal == -1:
    liq_price = liquidation_price(current_price, -1, suggested_leverage)
else:
    liq_price = None

# 日盈亏
if st.session_state.auto_position:
    update_daily_pnl(current_price, st.session_state.auto_position)
else:
    st.session_state.daily_pnl = 0.0

# 回撤
current_total = st.session_state.account_balance + st.session_state.daily_pnl
if current_total > st.session_state.peak_balance:
    st.session_state.peak_balance = current_total
drawdown = (st.session_state.peak_balance - current_total) / st.session_state.peak_balance * 100.0

can_trade_flag = can_trade()
eligibility = "活跃" if can_trade_flag and entry_signal != 0 else "禁止"

# 更新移动止损
if st.session_state.auto_position:
    st.session_state.auto_position = update_trailing_stop(st.session_state.auto_position, current_price)

# ==================== 主布局 ====================
col_left, col_right = st.columns([1.4, 1.6])

with col_left:
    # ① 全球宏观
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">① 全球宏观</div>', unsafe_allow_html=True)
        cm1, cm2, cm3, cm4 = st.columns(4)
        with cm1: st.metric("市场状态", market_mode)
        with cm2: st.metric("波动率(ATR%)", f"{atr_pct:.2f}%")
        with cm3: st.metric("趋势强度(ADX)", f"{adx:.1f}")
        with cm4: st.metric("恐惧指数", f"{fear_greed}")
        st.markdown(f"<div>数据源: {source}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ② 策略概况
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">② 策略概况</div>', unsafe_allow_html=True)
        cs1, cs2, cs3, cs4 = st.columns(4)
        with cs1: st.metric("策略模式", manual_mode)
        with cs2: st.metric("杠杆范围", f"{min_lev:.0f}x–{max_lev:.0f}x")
        with cs3: st.metric("盈亏比", f"1:{TP_MIN_RATIO}")
        with cs4: st.metric("日亏损限额", f"{DAILY_LOSS_LIMIT:.0f} USDT")
        st.markdown('</div>', unsafe_allow_html=True)

    # ③ 信号引擎 + 入场条件
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">③ 信号引擎</div>', unsafe_allow_html=True)
        ci1, ci2, ci3, ci4 = st.columns(4)
        with ci1: st.metric("品种", selected_symbol)
        with ci2: st.metric("周期", main_period)
        with ci3:
            status = "等待" if entry_signal == 0 else ("做多" if entry_signal == 1 else "做空")
            st.metric("信号状态", status)
        with ci4: st.metric("杠杆建议", f"{suggested_leverage:.1f}x")
        st.markdown(f"<div>执行资格: <span class='eligibility-{'active' if eligibility=='活跃' else 'blocked'}'>{eligibility}</span></div>", unsafe_allow_html=True)

        if ai_prob is not None:
            st.markdown(f"<div>AI预测胜率: <span style='color:#FFD700;'>{ai_prob:.1f}%</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div>AI预测: 未启用</div>", unsafe_allow_html=True)

        st.markdown("#### 入场条件")
        c1 = "✅" if is_uptrend(df_15m) or is_downtrend(df_15m) else "❌"
        c2 = "✅" if check_multiframe_trend(data_dict) != 0 else "❌"
        c3 = "✅" if atr_pct >= MIN_ATR_PCT else "❌"
        c4 = "✅" if df_15m.iloc[-1]['volume_surge'] else "❌"
        rsi = df_15m.iloc[-1]['rsi']
        c5 = "✅" if (entry_signal == 1 and rsi > 50) or (entry_signal == -1 and rsi < 50) else "❌"
        c6 = "✅" if btc_trend == entry_signal else "❌" if btc_trend is not None else "⚪未启用"
        st.markdown(f"""
        <div style="font-size:0.8rem;">
            {c1} 严格趋势<br>
            {c2} 多周期一致<br>
            {c3} 波动率 ≥ {MIN_ATR_PCT}%<br>
            {c4} 成交量放量<br>
            {c5} RSI方向匹配<br>
            {c6} 大盘BTC同步
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
    with st.container():
        st.markdown('<div class="card" style="border-left-color: #FFAA00;">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">④ 风险引擎</div>', unsafe_allow_html=True)
        # 风险因子简化
        r1 = (entry_signal != 0) * 0.5 + 0.5
        r2 = 1.0 if atr_pct > 0.8 else 0.5
        r3 = 1.0 if drawdown < 10 else 0.5
        r4 = 1.0 if st.session_state.consecutive_losses < 3 else 0.5
        st.markdown(f'<div class="risk-factor"><span>质量因子</span><span>{r1:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-factor"><span>波动因子</span><span>{r2:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-factor"><span>回撤因子</span><span>{r3:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="risk-factor"><span>连亏因子</span><span>{r4:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="risk-line"></div>', unsafe_allow_html=True)
        r_final = BASE_RISK * r1 * r2 * r3 * r4
        r_final = max(0.001, min(0.02, r_final))
        st.markdown(f'<div class="risk-factor"><span>R_final</span><span>{r_final*100:.2f}%</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="display:flex; justify-content:space-between; margin-top:8px;">'
                    f'<div>资本风险<br><span class="metric-value">{(st.session_state.account_balance * r_final):.1f} USDT</span></div>'
                    f'<div>爆仓价<br><span class="metric-value">{"${:.2f}".format(liq_price) if liq_price else "—"}</span></div>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ⑤ 资本状态
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⑤ 资本状态</div>', unsafe_allow_html=True)
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: st.metric("账户余额", f"{st.session_state.account_balance:.0f} USDT")
        with cc2: st.metric("日盈亏", f"{st.session_state.daily_pnl:.1f}")
        with cc3: st.metric("当前回撤", f"{drawdown:.2f}%")
        with cc4: st.metric("连亏次数", st.session_state.consecutive_losses)
        st.markdown('</div>', unsafe_allow_html=True)

    # ⑥ 链上情绪（折叠）
    with st.expander("🔗 链上情绪"):
        st.write(f"交易所净流入: **{netflow:+.0f} {selected_symbol.split('/')[0]}** (模拟)")
        st.write(f"大额转账: **{whale}** 笔 (模拟)")
        st.write(f"恐惧贪婪指数: **{fear_greed}**")

with col_right:
    st.subheader(f"📈 {selected_symbol} K线 ({main_period})")
    if main_period in data_dict and not data_dict[main_period].empty:
        df = data_dict[main_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.2, 0.2])
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

        # 历史信号标注
        buy = df[df['buy_signal'] == True]
        for _, r in buy.iterrows():
            fig.add_annotation(x=r['日期'], y=r['low']*0.99, text="▲", showarrow=False, font=dict(size=12, color="#00F5A0"), row=1, col=1)
        sell = df[df['sell_signal'] == True]
        for _, r in sell.iterrows():
            fig.add_annotation(x=r['日期'], y=r['high']*1.01, text="▼", showarrow=False, font=dict(size=12, color="#FF5555"), row=1, col=1)

        # 当前信号箭头
        if entry_signal != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            arrow_text = "▲ 多" if entry_signal == 1 else "▼ 空"
            arrow_color = "green" if entry_signal == 1 else "red"
            fig.add_annotation(x=last_date, y=last_price*(1.02 if entry_signal==1 else 0.98),
                               text=arrow_text, showarrow=True, arrowhead=2, arrowcolor=arrow_color, font=dict(size=10))

        # RSI
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], line=dict(color="purple", width=1), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        # 成交量
        colors_vol = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['日期'], y=df['volume'], name="成交量", marker_color=colors_vol, showlegend=False), row=3, col=1)

        fig.update_layout(hovermode='x unified', template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
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
if can_trade_flag and st.session_state.get('auto_enabled', False) and entry_signal != 0:
    # 开新仓
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
        st.session_state.signal_history.append({
            '时间': now.strftime("%H:%M"),
            '方向': '多' if entry_signal == 1 else '空',
            '市场': market_mode,
            '价格': current_price,
            'AI胜率': f"{ai_prob:.1f}%" if ai_prob else "—"
        })
        if use_telegram and st.session_state.telegram_token:
            msg = f"🚀 开仓 {selected_symbol} {'多' if entry_signal==1 else '空'} @ {current_price:.2f}"
            send_telegram_message(msg)
    else:
        # 检查止损止盈
        pos = st.session_state.auto_position
        if (pos['side'] == 'long' and current_price <= pos['stop']) or \
           (pos['side'] == 'short' and current_price >= pos['stop']):
            pnl = (current_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - current_price) * pos['size']
            pnl_pct = pnl / (pos['entry'] * pos['size']) * 100.0
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
            st.session_state.auto_position = None
            if use_telegram:
                send_telegram_message(f"🔴 止损平仓 {pnl:.2f} USDT")
        elif (pos['side'] == 'long' and current_price >= pos['take']) or \
             (pos['side'] == 'short' and current_price <= pos['take']):
            pnl = (current_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - current_price) * pos['size']
            pnl_pct = pnl / (pos['entry'] * pos['size']) * 100.0
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
            st.session_state.auto_position = None
            if use_telegram:
                send_telegram_message(f"🟢 止盈平仓 {pnl:.2f} USDT")
        elif (pos['side'] == 'long' and entry_signal == -1) or (pos['side'] == 'short' and entry_signal == 1):
            pnl = (current_price - pos['entry']) * pos['size'] if pos['side'] == 'long' else (pos['entry'] - current_price) * pos['size']
            pnl_pct = pnl / (pos['entry'] * pos['size']) * 100.0
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
            st.session_state.auto_position = None
            if use_telegram:
                send_telegram_message(f"↩️ 反向平仓 {pnl:.2f} USDT")
