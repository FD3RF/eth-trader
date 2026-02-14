# -*- coding: utf-8 -*-
"""
🚀 合约智能监控中心 · 终极职业版 V3
数据源：MEXC + CryptoCompare | 多周期融合 | 职业级风险引擎 | 动态仓位 | 回撤监控 | 共振矩阵
适配 100倍杠杆，符合职业交易员标准
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

# -------------------- 强平价格计算 --------------------
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "long":
        return entry_price * (1 - 1/leverage)
    else:
        return entry_price * (1 + 1/leverage)

# -------------------- 市场状态判断 --------------------
def get_market_state(df, period='15m'):
    """
    判断市场状态：趋势/震荡/高波动
    基于 ADX 和 ATR%
    """
    high, low, close = df['high'], df['low'], df['close']
    adx = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    current_adx = adx.iloc[-1]
    atr = df['atr'].iloc[-1]
    atr_pct = (atr / close.iloc[-1]) * 100
    
    if current_adx > 25:
        state = "趋势"
    else:
        state = "震荡"
    
    if atr_pct > 5:
        volatility = "高波动"
    elif atr_pct > 2:
        volatility = "中波动"
    else:
        volatility = "低波动"
    
    return state, volatility, current_adx, atr_pct

# -------------------- 信号强度评分 (0-100) --------------------
def signal_strength_score(df, fusion_dir, fusion_conf):
    """
    综合多个因子给出信号强度评分
    """
    if fusion_dir == 0:
        return 0
    last = df.iloc[-1]
    score = 0
    # 趋势因子
    if last['ma20'] > last['ma60']:
        score += 20
    # RSI 因子
    if fusion_dir == 1 and last['rsi'] < 40:
        score += 20
    elif fusion_dir == -1 and last['rsi'] > 60:
        score += 20
    # 成交量因子
    if last['volume_ratio'] > 1.2:
        score += 15
    # 布林带因子
    if fusion_dir == 1 and last['close'] < last['bb_low'] * 1.02:
        score += 15
    elif fusion_dir == -1 and last['close'] > last['bb_high'] * 0.98:
        score += 15
    # 置信度因子
    score += fusion_conf * 30
    return min(int(score), 100)

# -------------------- 高级信号生成（多指标融合，可调灵敏度）--------------------
def generate_signals(df, sensitivity=1.0):
    """
    基于多个指标生成买卖信号
    sensitivity: 灵敏度系数（0.5~2.0），值越大信号越容易触发
    """
    df = df.copy()
    # 基础指标
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
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # 根据灵敏度调整阈值
    rsi_oversold = 30 / sensitivity
    rsi_overbought = 70 * sensitivity
    volume_threshold = 1.2 / sensitivity

    # 多头信号条件
    buy_cond1 = (df['rsi'] < rsi_oversold) & (df['close'] > df['ma20'])
    buy_cond2 = (df['macd_diff'] > 0) & (df['macd_diff'].shift(1) <= 0)
    buy_cond3 = (df['close'] > df['ma20']) & (df['ma20'] > df['ma60']) & (df['volume_ratio'] > volume_threshold)
    buy_cond4 = (df['close'] < df['bb_low']) & (df['rsi'] < 50)
    df['buy_signal'] = buy_cond1 | buy_cond2 | buy_cond3 | buy_cond4

    # 空头信号条件
    sell_cond1 = (df['rsi'] > rsi_overbought) & (df['close'] < df['ma60'])
    sell_cond2 = (df['macd_diff'] < 0) & (df['macd_diff'].shift(1) >= 0)
    sell_cond3 = (df['close'] < df['ma20']) & (df['ma20'] < df['ma60']) & (df['volume_ratio'] > volume_threshold)
    sell_cond4 = (df['close'] > df['bb_high']) & (df['rsi'] > 50)
    df['sell_signal'] = sell_cond1 | sell_cond2 | sell_cond3 | sell_cond4

    return df

# -------------------- 极简数据获取器（仅 MEXC + CryptoCompare）--------------------
class SimpleDataFetcher:
    def __init__(self):
        self.symbol = "ETHUSDT"
        self.periods = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.limit = 200
        self.timeout = 5

        self.mexc = {
            'name': 'MEXC',
            'url': 'https://api.mexc.com/api/v3/klines',
            'params': {'symbol': self.symbol, 'interval': None, 'limit': self.limit}
        }
        self.cryptocompare = {
            'name': 'CryptoCompare',
            'base_url': 'https://min-api.cryptocompare.com/data/v2',
            'params': {'fsym': 'ETH', 'tsym': 'USD', 'limit': self.limit}
        }
        self.price_url = 'https://api.mexc.com/api/v3/ticker/price'
        self.price_params = {'symbol': self.symbol}

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
            resp = requests.get(self.price_url, params=self.price_params, timeout=self.timeout)
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

        return data_dict, price, price_source, errors, source_display or '无'

# -------------------- AI预测（多周期融合） --------------------
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
        # 趋势信号
        if last['ma20'] > last['ma60']:
            signals['trend'] = 1
        elif last['ma20'] < last['ma60']:
            signals['trend'] = -1
        else:
            signals['trend'] = 0
        # 震荡信号（RSI）
        if last['rsi'] < 30:
            signals['oscillator'] = 1
        elif last['rsi'] > 70:
            signals['oscillator'] = -1
        else:
            signals['oscillator'] = 0
        # 成交量信号
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

# -------------------- 多周期共振矩阵 --------------------
def resonance_matrix(df_dict):
    """
    生成各周期信号强度表格
    """
    rows = []
    for period, df in df_dict.items():
        if df is not None and len(df) > 20:
            last = df.iloc[-1]
            # 简单信号：1=多，-1=空，0=观望
            if last['rsi'] < 30 and last['close'] > last['ma20']:
                sig = 1
            elif last['rsi'] > 70 and last['close'] < last['ma60']:
                sig = -1
            else:
                sig = 0
            rows.append({
                '周期': period,
                '信号': sig,
                'RSI': round(last['rsi'], 1),
                'MA20': round(last['ma20'], 1),
                'MA60': round(last['ma60'], 1),
                'ATR%': round(last['atr']/last['close']*100, 2)
            })
    df_matrix = pd.DataFrame(rows)
    return df_matrix

# -------------------- 动态止损止盈计算 --------------------
def dynamic_stops(entry_price, side, atr, leverage, risk_reward=2.0):
    """基于ATR和杠杆计算止损止盈"""
    if side == 1:  # 做多
        stop_loss = entry_price - 1.5 * atr
        take_profit = entry_price + 1.5 * atr * risk_reward
    else:  # 做空
        stop_loss = entry_price + 1.5 * atr
        take_profit = entry_price - 1.5 * atr * risk_reward
    return stop_loss, take_profit

# -------------------- 风险暴露与仓位计算 --------------------
def calculate_risk_exposure(entry_price, stop_loss, quantity, leverage, account_balance):
    """
    计算单笔风险暴露百分比
    """
    if stop_loss is None or entry_price == 0:
        return 0
    risk_per_unit = abs(entry_price - stop_loss)
    total_risk = risk_per_unit * quantity * leverage
    risk_pct = (total_risk / account_balance) * 100 if account_balance > 0 else 0
    return risk_pct

def kelly_fraction(win_rate, avg_win, avg_loss):
    """
    Kelly公式计算最优仓位比例
    """
    if avg_loss == 0:
        return 0
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    kelly = (p * b - q) / b
    return max(0, min(kelly, 0.25))  # 限制最大25%

# -------------------- 缓存数据获取 --------------------
@st.cache_data(ttl=60)
def fetch_all_data(sensitivity):
    fetcher = SimpleDataFetcher()
    data_dict, price, price_source, errors, source_display = fetcher.fetch_all()
    if data_dict:
        for p in data_dict:
            data_dict[p] = generate_signals(data_dict[p], sensitivity)
    return data_dict, price, price_source, errors, source_display

# -------------------- 会话状态初始化 --------------------
def init_session_state():
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 10000.0  # 初始资金
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'daily_loss_limit' not in st.session_state:
        st.session_state.daily_loss_limit = 500.0  # 日亏损限额
    if 'peak_balance' not in st.session_state:
        st.session_state.peak_balance = 10000.0
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []  # 用于计算胜率等
    if 'last_date' not in st.session_state:
        st.session_state.last_date = datetime.now().date()

# -------------------- 更新每日盈亏与回撤 --------------------
def update_daily_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage):
    today = datetime.now().date()
    if today != st.session_state.last_date:
        # 新的一天，重置日盈亏
        st.session_state.daily_pnl = 0.0
        st.session_state.last_date = today

    if sim_entry > 0:
        if sim_side == "多单":
            unrealized_pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
        else:
            unrealized_pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
        st.session_state.daily_pnl = unrealized_pnl  # 简化，实际应基于已平仓盈亏

    # 更新峰值
    current_balance = st.session_state.account_balance + st.session_state.daily_pnl
    if current_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = current_balance

    current_drawdown = (st.session_state.peak_balance - current_balance) / st.session_state.peak_balance * 100
    return current_drawdown

# -------------------- Streamlit 界面 --------------------
st.set_page_config(page_title="合约智能监控·终极职业版", layout="wide")
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
</style>
""", unsafe_allow_html=True)

st.title("🧠 合约智能监控中心 · 终极职业版 V3")
st.caption("数据源：MEXC + CryptoCompare｜职业级风险引擎｜多周期共振｜动态仓位｜回撤监控｜100倍杠杆适配")

# 初始化会话状态
init_session_state()

# 初始化融合模块
if 'fusion' not in st.session_state:
    st.session_state.fusion = MultiPeriodFusion()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
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
    account_balance = st.number_input("初始资金 (USDT)", value=st.session_state.account_balance, min_value=100.0, step=1000.0, format="%.2f")
    daily_loss_limit = st.number_input("日亏损限额 (USDT)", value=st.session_state.daily_loss_limit, min_value=0.0, step=100.0, format="%.2f")
    # 更新 session_state
    st.session_state.account_balance = account_balance
    st.session_state.daily_loss_limit = daily_loss_limit

# 获取数据
data_dict, current_price, price_source, errors, source_display = fetch_all_data(sensitivity)

# 显示数据源状态
if data_dict:
    st.markdown(f'<div class="info-box">✅ 当前数据源：{source_display} | 价格源：{price_source} | 灵敏度：{sensitivity}</div>', unsafe_allow_html=True)

if errors and len(errors) > 3:
    st.warning(f"⚠️ 部分周期数据不可用 ({len(errors)}个周期)，将使用可用周期计算信号")

# 计算多周期融合信号
if data_dict:
    fusion_dir, fusion_conf = st.session_state.fusion.fuse_periods(data_dict)
else:
    fusion_dir, fusion_conf = 0, 0

# 获取当前周期的数据和ATR
selected_df = data_dict.get(selected_period) if data_dict else None
atr_value = None
market_state = "未知"
volatility = "未知"
adx = 0
atr_pct = 0
if selected_df is not None and len(selected_df) > 20:
    atr_series = selected_df['atr']
    if not atr_series.empty:
        atr_value = atr_series.iloc[-1]
        state, vol, adx, atr_pct = get_market_state(selected_df)
        market_state = state
        volatility = vol

# 生成动态止损止盈
stop_loss = None
take_profit = None
if fusion_dir != 0 and current_price is not None and atr_value is not None and atr_value > 0:
    stop_loss, take_profit = dynamic_stops(current_price, fusion_dir, atr_value, sim_leverage)

# 计算风险暴露
risk_pct = 0
if sim_entry > 0 and stop_loss is not None:
    risk_pct = calculate_risk_exposure(sim_entry, stop_loss, sim_quantity, sim_leverage, st.session_state.account_balance)

# 更新每日盈亏与回撤
current_drawdown = 0
if selected_df is not None and current_price is not None:
    current_drawdown = update_daily_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage)

# 信号强度评分
signal_score = 0
if selected_df is not None:
    signal_score = signal_strength_score(selected_df, fusion_dir, fusion_conf)

# 检查日亏损限制
daily_loss_exceeded = st.session_state.daily_pnl < -st.session_state.daily_loss_limit
if daily_loss_exceeded:
    st.error("🚨 日亏损限额已触发！建议停止交易。")

# 多周期共振矩阵
if data_dict:
    matrix_df = resonance_matrix(data_dict)
    # 将信号映射为图标
    matrix_df['信号'] = matrix_df['信号'].map({1: "▲ 多", -1: "▼ 空", 0: "⚪ 观"})
else:
    matrix_df = pd.DataFrame()

# 主布局
col1, col2 = st.columns([2.2, 1.3])

with col1:
    # 市场状态标识
    if selected_df is not None:
        state_color = {"趋势": "#00F5A0", "震荡": "#FFAA00", "高波动": "#FF5555"}.get(market_state, "#FFFFFF")
        st.markdown(f"<h5>市场状态: <span style='color:{state_color};'>{market_state} / {volatility}</span> (ADX: {adx:.1f}, ATR%: {atr_pct:.2f}%)</h5>", unsafe_allow_html=True)
    
    st.subheader(f"📊 合约K线 ({selected_period})  — 绿色▲=做多信号，红色▼=做空信号")
    if selected_df is not None:
        df = selected_df.tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"ETHUSDT {selected_period}", "RSI"))
        # K线
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                      low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        # 均线
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma20'], name="MA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ma60'], name="MA60", line=dict(color="blue")), row=1, col=1)

        # 添加买卖信号标注
        buy_signals = df[df['buy_signal'] == True]
        for idx, row in buy_signals.iterrows():
            fig.add_annotation(
                x=row['日期'], y=row['low'] * 0.98,
                text="▲", showarrow=False, font=dict(size=16, color="#00F5A0"),
                row=1, col=1
            )
        sell_signals = df[df['sell_signal'] == True]
        for idx, row in sell_signals.iterrows():
            fig.add_annotation(
                x=row['日期'], y=row['high'] * 1.02,
                text="▼", showarrow=False, font=dict(size=16, color="#FF5555"),
                row=1, col=1
            )

        # 当前融合信号箭头
        if fusion_dir != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            if fusion_dir == 1:
                fig.add_annotation(x=last_date, y=last_price * 1.02,
                                   text="▲ 融合多", showarrow=True, arrowhead=2, arrowcolor="green")
            else:
                fig.add_annotation(x=last_date, y=last_price * 0.98,
                                   text="▼ 融合空", showarrow=True, arrowhead=2, arrowcolor="red")

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
    st.markdown(f'<div class="ai-box">{dir_map[fusion_dir]}<br>置信度: {fusion_conf:.1%}</div>', unsafe_allow_html=True)

    # 信号强度条形图
    if signal_score > 0:
        st.progress(signal_score/100, text=f"信号强度: {signal_score}/100")
    else:
        st.info("信号强度: 0/100")

    if current_price is not None:
        st.metric("当前价格", f"${current_price:.2f}", delta_color="off")
    else:
        st.metric("当前价格", "获取中...")

    # 风险仪表盘
    with st.container():
        st.markdown('<div class="dashboard">', unsafe_allow_html=True)
        st.markdown("#### 📊 风险仪表盘")
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            st.metric("账户余额", f"${st.session_state.account_balance:.2f}")
            st.metric("日盈亏", f"${st.session_state.daily_pnl:.2f}", delta_color="inverse")
        with col_risk2:
            st.metric("当前回撤", f"{current_drawdown:.2f}%")
            st.metric("风险暴露", f"{risk_pct:.2f}%")
        # 日亏损状态
        if daily_loss_exceeded:
            st.error("⚠️ 日亏损超限")
        else:
            remaining = st.session_state.daily_loss_limit + st.session_state.daily_pnl
            st.info(f"日亏损剩余: ${remaining:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 动态交易计划
    if fusion_dir != 0 and stop_loss is not None and take_profit is not None:
        risk_pct_plan = abs(current_price - stop_loss) / current_price * 100
        reward_pct_plan = abs(take_profit - current_price) / current_price * 100
        st.markdown(f"""
        <div class="trade-plan">
            <h4>📋 动态交易计划</h4>
            <p>进场: <span style="color:#00F5A0">${current_price:.2f}</span></p>
            <p>止损: <span style="color:#FF5555">${stop_loss:.2f}</span> (风险 {risk_pct_plan:.2f}%)</p>
            <p>止盈: <span style="color:#00F5A0">${take_profit:.2f}</span> (盈亏比 {reward_pct_plan/risk_pct_plan:.2f})</p>
            <p>ATR(14): {atr_value:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("当前无明确信号，无交易计划")

    # 多周期共振矩阵
    if not matrix_df.empty:
        with st.expander("📈 多周期共振矩阵", expanded=False):
            st.dataframe(matrix_df, use_container_width=True)

    # 模拟合约持仓显示 + 强平预警
    if sim_entry > 0 and current_price is not None and selected_df is not None:
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
            st.error("🚨🚨🚨 强平风险！当前价格已触及强平线！")
        elif distance_to_liq < 5:
            st.warning(f"⚠️⚠️ 距离强平仅 {distance_to_liq:.2f}%，请注意风险！")
        elif distance_to_liq < 10:
            st.info(f"距离强平 {distance_to_liq:.2f}%，请保持关注。")
    else:
        st.info("请输入开仓价以查看模拟盈亏与强平分析")
