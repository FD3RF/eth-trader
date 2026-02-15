# -*- coding: utf-8 -*-
"""
🚀 合约智能监控中心 · 终极生存版（五层架构）
市场环境层 | 入场信号层 | 风险控制层 | 资金管理层 | 生存保护层
多币种卡片｜资金曲线｜简易回测｜交易日志｜风险预警
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

warnings.filterwarnings('ignore')

# ==================== 全局配置（统一使用浮点数）====================
SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT"]
RISK_PCT = 0.01                     # 单笔风险 1%
MAX_LEVERAGE = 20.0                 # 最大杠杆限制
STOP_ATR = 1.5                      # 止损倍数
TAKE_ATR = 3.0                      # 止盈倍数
CONSECUTIVE_LOSS_LIMIT = 3          # 连亏刹车阈值
CONSECUTIVE_STOP_HOURS = 24         # 连亏暂停小时数
MAX_DRAWDOWN = 20.0                  # 最大回撤警戒线（%）
DAILY_LOSS_LIMIT = 300.0             # 日亏损限额（USDT）

# ==================== 免费数据获取器（支持多币种）====================
class FreeDataFetcherV5:
    """支持多币种的免费数据获取器"""
    
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = SYMBOLS
        self.symbols = symbols
        self.periods = ['15m', '1h', '4h', '1d']
        self.limit = 500
        self.timeout = 10
        
        # MEXC交易所实例
        self.exchange = ccxt.mexc({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # 恐惧贪婪指数
        self.fng_url = "https://api.alternative.me/fng/"
        
        # 模拟链上数据（标注模拟）
        self.chain_netflow = 5234
        self.chain_whale = 128
    
    def fetch_kline(self, symbol, timeframe):
        """获取单个币种K线"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df, "MEXC"
        except Exception as e:
            st.warning(f"{symbol} {timeframe} 获取失败: {e}")
            return None, None
    
    def fetch_fear_greed(self):
        """获取恐惧贪婪指数"""
        try:
            resp = requests.get(self.fng_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return int(data['data'][0]['value'])
        except:
            pass
        return 50
    
    def fetch_all(self):
        """获取所有币种所有周期的数据"""
        all_data = {}
        fear_greed = self.fetch_fear_greed()
        
        for symbol in self.symbols:
            data_dict = {}
            price_sources = []
            for period in self.periods:
                df, src = self.fetch_kline(symbol, period)
                if df is not None:
                    data_dict[period] = self._add_indicators(df)
                    price_sources.append(src)
            
            if data_dict:
                all_data[symbol] = {
                    "data_dict": data_dict,
                    "current_price": data_dict['15m']['close'].iloc[-1] if '15m' in data_dict else None,
                    "source": price_sources[0] if price_sources else "MEXC",
                    "fear_greed": fear_greed,
                    "chain_netflow": self.chain_netflow,
                    "chain_whale": self.chain_whale,
                }
        
        return all_data
    
    def _add_indicators(self, df):
        """添加技术指标"""
        df = df.copy()
        # 均线
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        # 布林带
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        return df


# ==================== 市场环境层 ====================
def evaluate_market(df_dict):
    """判断市场状态：趋势/震荡/禁止交易"""
    if '15m' not in df_dict:
        return "禁止交易", 0.0, 0.0
    df = df_dict['15m']
    last = df.iloc[-1]
    
    ema20 = last['ema20']
    ema50 = last['ema50']
    adx = last['adx']
    atr_pct = last['atr_pct']
    
    # 异常波动检测
    body = abs(last['close'] - last['open'])
    if body > 3 * last['atr']:
        return "禁止交易", atr_pct, adx
    
    # 波动率不足
    if atr_pct < 0.5:
        return "禁止交易", atr_pct, adx
    
    # 趋势模式
    if ema20 > ema50 and adx > 20:
        return "趋势", atr_pct, adx
    # 震荡模式
    elif adx < 25:
        return "震荡", atr_pct, adx
    else:
        return "禁止交易", atr_pct, adx


# ==================== 入场信号层 ====================
def generate_entry_signal(df_dict, mode):
    """根据市场模式生成入场信号"""
    if '15m' not in df_dict:
        return 0
    df = df_dict['15m']
    last = df.iloc[-1]
    
    if mode == "趋势":
        ema20 = last['ema20']
        ema50 = last['ema50']
        # 趋势多：EMA20 > EMA50 且 价格回踩EMA20 且 RSI未过热
        if (ema20 > ema50 and 
            last['close'] >= ema20 * 0.99 and 
            last['rsi'] < 70 and last['rsi'] > 40):
            return 1
        # 趋势空：EMA20 < EMA50 且 价格反弹至EMA20 且 RSI未超卖
        elif (ema20 < ema50 and 
              last['close'] <= ema20 * 1.01 and 
              last['rsi'] > 30 and last['rsi'] < 60):
            return -1
        else:
            return 0
    elif mode == "震荡":
        bb_upper = last['bb_high']
        bb_lower = last['bb_low']
        # 下轨买
        if last['close'] <= bb_lower * 1.01 and last['rsi'] < 30:
            return 1
        # 上轨卖
        elif last['close'] >= bb_upper * 0.99 and last['rsi'] > 70:
            return -1
        else:
            return 0
    else:
        return 0


# ==================== 风险控制层 ====================
def calculate_stops(entry_price, side, atr_value):
    """计算止损止盈价"""
    stop_distance = STOP_ATR * atr_value
    take_distance = TAKE_ATR * atr_value
    if side == 1:  # 多
        stop = entry_price - stop_distance
        take = entry_price + take_distance
    else:  # 空
        stop = entry_price + stop_distance
        take = entry_price - take_distance
    return stop, take, take_distance/stop_distance


# ==================== 资金管理层 ====================
def calculate_position_size(balance, entry_price, stop_price, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE):
    """计算仓位大小（基于风险金额）"""
    risk_amount = balance * risk_pct
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return 0.0
    # 理论仓位价值
    position_value = risk_amount / stop_distance * entry_price
    # 根据杠杆限制
    max_position = balance * max_leverage
    position_value = min(position_value, max_position)
    quantity = position_value / entry_price
    return round(quantity, 3)


# ==================== 生存保护层 ====================
class SurvivalProtection:
    """生存保护机制（单例，使用session_state持久化）"""
    
    def __init__(self):
        self.consecutive_losses = 0
        self.peak_balance = 10000.0
        self.mode_switch_time = None
        self.trading_paused_until = None
        self.daily_loss_triggered = False
        self.last_mode = None
        self.daily_pnl = 0.0
        
    def update(self, trade_result, current_balance, current_mode, last_kline_time, daily_pnl):
        # 更新连续亏损
        if trade_result < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # 更新回撤
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100.0
        
        # 模式切换冷却
        if self.last_mode is not None and current_mode != self.last_mode:
            self.mode_switch_time = last_kline_time
        self.last_mode = current_mode
        
        # 日亏损检测
        if daily_pnl < -DAILY_LOSS_LIMIT:
            self.daily_loss_triggered = True
        
        # 检查是否暂停交易
        paused = False
        if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            paused = True
            if self.trading_paused_until is None:
                self.trading_paused_until = last_kline_time + timedelta(hours=CONSECUTIVE_STOP_HOURS)
        if drawdown > MAX_DRAWDOWN:
            paused = True
            if self.trading_paused_until is None:
                self.trading_paused_until = last_kline_time + timedelta(hours=24)
        if self.daily_loss_triggered:
            paused = True
        
        return paused, drawdown
    
    def can_trade(self, current_time):
        if self.trading_paused_until and current_time < self.trading_paused_until:
            return False
        # 日亏损触发全天禁止
        if self.daily_loss_triggered:
            return False
        return True


# ==================== 强平价格计算 ====================
def calculate_liquidation_price(entry_price, side, leverage):
    if side == "多单":
        return entry_price * (1 - 1.0/leverage)
    else:
        return entry_price * (1 + 1.0/leverage)


# ==================== 简易回测模块 ====================
def run_backtest(df_dict, mode_func, signal_func, initial_balance=10000.0, lookback_days=30):
    """
    简单回测：根据历史K线模拟交易
    返回：胜率、总收益、最大回撤、盈亏比、交易次数
    """
    df = df_dict['15m'].copy()
    lookback = lookback_days * 96  # 每天96根15m K线
    df = df.iloc[-lookback:] if len(df) > lookback else df
    
    balance = initial_balance
    peak = balance
    trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    max_drawdown = 0.0
    
    position = None
    entry_price = 0.0
    entry_side = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        # 构造一个临时的data_dict供环境判断（只有当前周期）
        temp_dict = {'15m': df.iloc[:i+1]}
        
        mode, _, _ = mode_func(temp_dict)  # 市场环境
        signal = signal_func(temp_dict, mode)  # 入场信号
        
        if mode == "禁止交易":
            continue
        
        # 交易逻辑
        if position is None:
            if signal == 1:
                position = 'long'
                entry_price = row['close']
                entry_side = 1
            elif signal == -1:
                position = 'short'
                entry_price = row['close']
                entry_side = -1
        else:
            # 平仓条件：信号消失或反向
            if (position == 'long' and signal <= 0) or (position == 'short' and signal >= 0):
                exit_price = row['close']
                if position == 'long':
                    pnl = (exit_price - entry_price) / entry_price * 100.0
                else:
                    pnl = (entry_price - exit_price) / entry_price * 100.0
                trades += 1
                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)
                balance *= (1.0 + pnl/100.0)
                if balance > peak:
                    peak = balance
                else:
                    dd = (peak - balance) / peak * 100.0
                    if dd > max_drawdown:
                        max_drawdown = dd
                position = None
        
        if balance > peak:
            peak = balance
    
    win_rate = wins / trades if trades > 0 else 0.0
    total_return = (balance - initial_balance) / initial_balance * 100.0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    
    return {
        '胜率': f"{win_rate*100:.1f}%",
        '总收益': f"{total_return:.1f}%",
        '最大回撤': f"{max_drawdown:.1f}%",
        '盈亏比': f"{profit_factor:.2f}",
        '交易次数': trades
    }


# ==================== 初始化session_state ====================
def init_session_state():
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 10000.0
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'peak_balance' not in st.session_state:
        st.session_state.peak_balance = 10000.0
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
    if 'protection' not in st.session_state:
        st.session_state.protection = SurvivalProtection()


# ==================== 更新风控统计 ====================
def update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage):
    today = datetime.now().date()
    if today != st.session_state.last_date:
        st.session_state.daily_pnl = 0.0
        st.session_state.last_date = today
        # 重置保护层的日亏损标记
        st.session_state.protection.daily_loss_triggered = False
    
    if sim_entry > 0 and current_price:
        if sim_side == "多单":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
        st.session_state.daily_pnl = pnl
        # 更新保护层日亏损
        st.session_state.protection.daily_pnl = pnl
    
    current_balance = st.session_state.account_balance + st.session_state.daily_pnl
    if current_balance > st.session_state.peak_balance:
        st.session_state.peak_balance = current_balance
    drawdown = (st.session_state.peak_balance - current_balance) / st.session_state.peak_balance * 100.0
    return drawdown


# ==================== 主界面 ====================
st.set_page_config(page_title="合约智能监控·终极生存版", layout="wide")
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
.card { background: #1A1D27; border-radius: 5px; padding: 10px; text-align: center; cursor: pointer; }
.card:hover { background: #2A2D37; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 合约智能监控中心 · 终极生存版（五层架构）")
st.caption("市场环境｜入场信号｜风险控制｜资金管理｜生存保护｜多币种卡片｜资金曲线｜简易回测｜交易日志｜风险预警")

# 初始化
init_session_state()
ai_model = None  # 如需AI模型可加载，此处简化

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_symbol = st.selectbox("主交易对", SYMBOLS, index=0, key="selected_symbol")
    main_period = st.selectbox("主图周期", ["15m", "1h", "4h", "1d"], index=0)
    auto_refresh = st.checkbox("开启自动刷新", value=True)
    refresh_interval = st.number_input("刷新间隔(秒)", min_value=5, max_value=60, value=10, step=1, disabled=not auto_refresh)
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    
    st.markdown("---")
    st.subheader("📈 模拟合约")
    sim_entry = st.number_input("开仓价", value=0.0, format="%.2f", step=0.01)
    sim_side = st.selectbox("方向", ["多单", "空单"])
    sim_leverage = st.slider("杠杆倍数", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    sim_quantity = st.number_input("数量", value=0.01, format="%.4f", step=0.001)
    
    st.markdown("---")
    st.subheader("💰 风控设置")
    account_balance = st.number_input("初始资金 (USDT)", value=st.session_state.account_balance, step=1000.0, format="%.2f")
    daily_loss_limit = st.number_input("日亏损限额 (USDT)", value=DAILY_LOSS_LIMIT, step=50.0, format="%.2f")
    risk_per_trade = st.slider("单笔风险 (%)", min_value=0.5, max_value=3.0, value=RISK_PCT*100, step=0.5) / 100.0
    st.session_state.account_balance = account_balance
    
    st.markdown("---")
    st.subheader("📊 简易回测")
    backtest_days = st.slider("回测天数", min_value=7, max_value=90, value=30, step=1)
    if st.button("运行回测"):
        with st.spinner("回测中..."):
            fetcher = FreeDataFetcherV5(symbols=[selected_symbol])
            backtest_data = fetcher.fetch_all()
            if backtest_data and selected_symbol in backtest_data:
                bt_result = run_backtest(
                    backtest_data[selected_symbol]["data_dict"],
                    evaluate_market,
                    generate_entry_signal,
                    initial_balance=account_balance,
                    lookback_days=backtest_days
                )
                st.success("回测完成")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("胜率", bt_result['胜率'])
                    st.metric("总收益", bt_result['总收益'])
                with col2:
                    st.metric("最大回撤", bt_result['最大回撤'])
                    st.metric("盈亏比", bt_result['盈亏比'])
                with col3:
                    st.metric("交易次数", bt_result['交易次数'])

# 获取数据
with st.spinner("获取全市场数据..."):
    fetcher = FreeDataFetcherV5(symbols=SYMBOLS)
    all_data = fetcher.fetch_all()

# 多币种卡片
st.markdown("### 🔥 多币种实时信号")
cols = st.columns(len(SYMBOLS))
all_scores = {}
all_modes = {}
all_signals = {}
for i, sym in enumerate(SYMBOLS):
    if sym in all_data:
        df_dict = all_data[sym]["data_dict"]
        mode, atr_pct, adx = evaluate_market(df_dict)
        signal = generate_entry_signal(df_dict, mode)
        all_modes[sym] = mode
        all_signals[sym] = signal
        score_display = {1: "多", -1: "空", 0: "观"}[signal]
        color = {1: "🟢", -1: "🔴", 0: "⚪"}[signal]
        with cols[i]:
            if st.button(f"{sym}\n{color} {score_display}\n{mode}", key=f"card_{sym}"):
                st.session_state.selected_symbol = sym
                st.rerun()

# 当前选中的币种数据
if selected_symbol not in all_data:
    selected_symbol = SYMBOLS[0]
data = all_data[selected_symbol]
data_dict = data["data_dict"]
current_price = data["current_price"]
fear_greed = data["fear_greed"]
source_display = data["source"]
chain_netflow = data["chain_netflow"]
chain_whale = data["chain_whale"]

# 市场环境评估
market_mode, atr_pct, adx = evaluate_market(data_dict)
entry_signal = generate_entry_signal(data_dict, market_mode)

# 计算ATR值
atr_value = 0.0
if '15m' in data_dict:
    atr_value = data_dict['15m']['atr'].iloc[-1]

# 生成交易计划（如果有信号）
stop_loss = take_profit = risk_reward = None
position_size = 0.0
if entry_signal != 0 and atr_value > 0:
    stop_loss, take_profit, risk_reward = calculate_stops(current_price, entry_signal, atr_value)
    # 计算仓位
    position_size = calculate_position_size(
        st.session_state.account_balance,
        current_price,
        stop_loss,
        risk_pct=risk_per_trade,
        max_leverage=MAX_LEVERAGE
    )

# 更新风控统计
drawdown = update_risk_stats(current_price, sim_entry, sim_side, sim_quantity, sim_leverage)

# 生存保护层检查
protection = st.session_state.protection
now = datetime.now()
# 模拟上次交易结果（这里假设自动交易会更新，暂时设为0）
trade_result = 0.0
paused, drawdown_protect = protection.update(trade_result, 
                                              st.session_state.account_balance + st.session_state.daily_pnl,
                                              market_mode, now, st.session_state.daily_pnl)
can_trade = protection.can_trade(now)

# 显示数据源状态
st.markdown(f"""
<div class="info-box">
    ✅ 价格源：{source_display} | 恐惧贪婪：{fear_greed} | 市场状态：{market_mode}
    <br>⚠️ 链上数据为模拟值（可替换为Dune免费API）
    { '🔴 交易暂停中' if not can_trade else '' }
</div>
""", unsafe_allow_html=True)

# 风险预警
if not can_trade:
    reason = []
    if protection.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
        reason.append(f"连续{protection.consecutive_losses}笔亏损")
    if protection.daily_loss_triggered:
        reason.append("日亏损超限")
    if drawdown_protect > MAX_DRAWDOWN:
        reason.append("回撤超过20%")
    st.error(f"🚨 交易暂停: {', '.join(reason)}")

# 主布局
col_left, col_right = st.columns([2.2, 1.3])

with col_left:
    # 市场状态和五层评分（简化版）
    st.markdown(f"<h5>市场状态: <span style='color:green;'>{market_mode}</span> | ADX: {adx:.1f} | ATR%: {atr_pct:.2f}%</h5>", unsafe_allow_html=True)
    
    # 简化五层热力图（展示各层得分）
    layer_scores = {
        "趋势": 30 if market_mode == "趋势" else 0,
        "震荡": 30 if market_mode == "震荡" else 0,
        "资金面": 0,
        "链上": 15 if chain_netflow > 5000 else 0,
        "动量": 15 if entry_signal != 0 else 0
    }
    st.subheader("🔥 五层状态")
    cols = st.columns(5)
    for i, (name, val) in enumerate(layer_scores.items()):
        with cols[i]:
            st.markdown(f"<div style='background:#1A1D27; padding:10px; border-radius:5px; text-align:center;'><h4>{name}</h4><h2>{val}</h2></div>", unsafe_allow_html=True)

    # K线图
    st.subheader(f"📊 {selected_symbol} K线 ({main_period})")
    if main_period in data_dict:
        df = data_dict[main_period].tail(100).copy()
        df['日期'] = df['timestamp']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{selected_symbol} {main_period}", "RSI"))
        fig.add_trace(go.Candlestick(x=df['日期'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema20'], name="EMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['ema50'], name="EMA50", line=dict(color="blue")), row=1, col=1)
        # 信号箭头
        if entry_signal != 0:
            last_date = df['日期'].iloc[-1]
            last_price = df['close'].iloc[-1]
            arrow_text = "▲ 多" if entry_signal == 1 else "▼ 空"
            arrow_color = "green" if entry_signal == 1 else "red"
            fig.add_annotation(x=last_date, y=last_price * (1.02 if entry_signal==1 else 0.98),
                               text=arrow_text, showarrow=True, arrowhead=2, arrowcolor=arrow_color)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['rsi'], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("K线数据不可用")

with col_right:
    st.subheader("🧠 即时决策")
    dir_map = {1: "🔴 做多", -1: "🔵 做空", 0: "⚪ 观望"}
    st.markdown(f'<div class="ai-box">{dir_map[entry_signal]}<br>市场模式: {market_mode}</div>', unsafe_allow_html=True)

    if entry_signal != 0 and stop_loss and take_profit:
        st.markdown(f"""
        <div class="trade-plan">
            <h4>📋 交易计划</h4>
            <p>入场价: <span style="color:#00F5A0">${current_price:.2f}</span></p>
            <p>止损价: <span style="color:#FF5555">${stop_loss:.2f}</span> (亏损 {abs(current_price-stop_loss)/current_price*100:.2f}%)</p>
            <p>止盈价: <span style="color:#00F5A0">${take_profit:.2f}</span> (盈亏比 {risk_reward:.2f})</p>
            <p>建议仓位: {position_size} {selected_symbol.split('/')[0]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.metric("当前价格", f"${current_price:.2f}" if current_price else "N/A")

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
            st.metric("日亏损剩余", f"${daily_loss_limit + st.session_state.daily_pnl:.2f}")
        # 资金曲线
        if st.session_state.balance_history:
            st.line_chart(st.session_state.balance_history)
        st.markdown('</div>', unsafe_allow_html=True)

    # 资金面快照
    with st.expander("💰 资金面快照", expanded=True):
        st.write("资金费率: **暂缺（模拟）**")
        st.write("OI变化: **暂缺（模拟）**")
        st.write("多空比: **暂缺（模拟）**")

    # 链上&情绪
    with st.expander("🔗 链上&情绪", expanded=False):
        st.write(f"交易所净流入: **{chain_netflow:+.0f} {selected_symbol.split('/')[0]}** (模拟)")
        st.write(f"大额转账: **{chain_whale}** 笔 (模拟)")
        st.write(f"恐惧贪婪指数: **{fear_greed}**")

    # 模拟合约持仓
    if sim_entry > 0 and current_price:
        if sim_side == "多单":
            pnl = (current_price - sim_entry) * sim_quantity * sim_leverage
            pnl_pct = (current_price - sim_entry) / sim_entry * sim_leverage * 100.0
            liq_price = calculate_liquidation_price(sim_entry, "多单", sim_leverage)
        else:
            pnl = (sim_entry - current_price) * sim_quantity * sim_leverage
            pnl_pct = (sim_entry - current_price) / sim_entry * sim_leverage * 100.0
            liq_price = calculate_liquidation_price(sim_entry, "空单", sim_leverage)
        color_class = "profit" if pnl >= 0 else "loss"
        distance = abs(current_price - liq_price) / current_price * 100.0
        st.markdown(f"""
        <div class="metric">
            <h4>模拟持仓</h4>
            <p>{sim_side} | {sim_leverage:.1f}x</p>
            <p>开仓: ${sim_entry:.2f}</p>
            <p class="{color_class}">盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>强平价: <span class="warning">${liq_price:.2f}</span> (距 {distance:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        if distance < 5:
            st.warning("⚠️ 接近强平线！")
    else:
        st.info("输入开仓价查看模拟")

    # 策略自动测试
    st.markdown("---")
    st.subheader("🧪 策略自动测试")
    auto_enabled = st.checkbox("启用自动跟随信号（模拟）", value=st.session_state.auto_enabled)
    st.session_state.auto_enabled = auto_enabled

    # 自动交易逻辑（简化）
    if auto_enabled and can_trade and entry_signal != 0:
        # 开仓逻辑
        if st.session_state.auto_position is None:
            st.session_state.auto_position = {
                'side': 'long' if entry_signal == 1 else 'short',
                'entry': current_price,
                'time': datetime.now(),
                'leverage': MAX_LEVERAGE,
                'stop': stop_loss,
                'take': take_profit,
                'size': position_size
            }
            st.success(f"✅ 自动开{st.session_state.auto_position['side']}仓 @ {current_price:.2f}")
        else:
            # 检查是否应该平仓（反向信号或止损止盈触发）
            pos = st.session_state.auto_position
            if (pos['side'] == 'long' and (current_price <= pos['stop'] or current_price >= pos['take'])) or \
               (pos['side'] == 'short' and (current_price >= pos['stop'] or current_price <= pos['take'])) or \
               (entry_signal == -1 and pos['side'] == 'long') or \
               (entry_signal == 1 and pos['side'] == 'short'):
                # 平仓
                if pos['side'] == 'long':
                    pnl = (current_price - pos['entry']) * pos['leverage']
                else:
                    pnl = (pos['entry'] - current_price) * pos['leverage']
                pnl_pct = pnl / pos['entry'] * 100.0
                # 更新保护层（传入交易结果）
                protection.update(pnl, st.session_state.account_balance + st.session_state.daily_pnl,
                                   market_mode, now, st.session_state.daily_pnl)
                # 记录交易日志
                st.session_state.trade_log.append({
                    '开仓时间': pos['time'].strftime('%H:%M'),
                    '方向': pos['side'],
                    '开仓价': f"{pos['entry']:.2f}",
                    '平仓时间': datetime.now().strftime('%H:%M'),
                    '平仓价': f"{current_price:.2f}",
                    '盈亏': f"{pnl:.2f}",
                    '盈亏%': f"{pnl_pct:.1f}%"
                })
                # 更新余额历史
                st.session_state.balance_history.append(st.session_state.account_balance + st.session_state.daily_pnl)
                st.info(f"📉 平仓 {pos['side']}，盈亏: ${pnl:.2f}")
                st.session_state.auto_position = None

    # 显示当前自动持仓
    if st.session_state.auto_position:
        pos = st.session_state.auto_position
        pnl = (current_price - pos['entry']) * (1.0 if pos['side']=='long' else -1.0) * pos['leverage']
        pnl_pct = (current_price - pos['entry']) / pos['entry'] * pos['leverage'] * 100.0 * (1.0 if pos['side']=='long' else -1.0)
        liq_price = calculate_liquidation_price(pos['entry'], "多单" if pos['side']=='long' else "空单", pos['leverage'])
        distance = abs(current_price - liq_price) / current_price * 100.0
        color_class = "profit" if pnl >= 0 else "loss"
        st.markdown(f"""
        <div class="metric">
            <h4>自动模拟持仓</h4>
            <p>方向: {'多' if pos['side']=='long' else '空'} | 杠杆: {pos['leverage']:.1f}x</p>
            <p>开仓: ${pos['entry']:.2f} ({pos['time'].strftime('%H:%M')})</p>
            <p class="{color_class}">盈亏: ${pnl:.2f} ({pnl_pct:.2f}%)</p>
            <p>强平价: <span class="warning">${liq_price:.2f}</span> (距 {distance:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("手动平仓", key="auto_close"):
            if pos['side'] == 'long':
                pnl = (current_price - pos['entry']) * pos['leverage']
            else:
                pnl = (pos['entry'] - current_price) * pos['leverage']
            pnl_pct = pnl / pos['entry'] * 100.0
            protection.update(pnl, st.session_state.account_balance + st.session_state.daily_pnl,
                              market_mode, now, st.session_state.daily_pnl)
            st.session_state.trade_log.append({
                '开仓时间': pos['time'].strftime('%H:%M'),
                '方向': pos['side'],
                '开仓价': f"{pos['entry']:.2f}",
                '平仓时间': datetime.now().strftime('%H:%M'),
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
            if can_trade:
                st.info("等待信号开仓")
            else:
                st.warning("交易暂停中")

    # 交易日志
    with st.expander("📋 交易日志"):
        if st.session_state.trade_log:
            st.dataframe(pd.DataFrame(st.session_state.trade_log), use_container_width=True)
        else:
            st.info("暂无交易记录")

    # 历史信号记录
    if entry_signal != 0:
        current_dir = "多" if entry_signal == 1 else "空"
        if not st.session_state.signal_history or st.session_state.signal_history[-1]['方向'] != current_dir:
            st.session_state.signal_history.append({
                '时间': datetime.now().strftime("%H:%M"),
                '方向': current_dir,
                '市场': market_mode
            })
            st.session_state.signal_history = st.session_state.signal_history[-20:]

    with st.expander("📋 历史信号记录"):
        if st.session_state.signal_history:
            st.dataframe(pd.DataFrame(st.session_state.signal_history), use_container_width=True)
        else:
            st.info("暂无历史信号")
