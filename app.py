# -*- coding: utf-8 -*-
"""
🤖 AI 自进化交易平台 VAI v9.0 终极优化版
===========================================================
功能亮点：
- 多周期共振策略（5m/15m/1h） + 自定义指标组合
- 多交易所自动切换 + 重试机制 + 数据缓存
- 实时交易界面采用深色专业风格，三币种三层图表（价格+成交量+MACD）
- 移动端适配 + 止损/止盈水平线 + 信号强度打分
- 参数即时应用 + 回测参数优化器 + 报表导出
- 最大回撤保护 + 每日亏损限额 + 日志持久化
- 使用 st.fragment 实现增量更新，提升性能
===========================================================
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import nest_asyncio
from scipy.stats import t
import os
import time
from retry import retry  # pip install retry
import openpyxl
from io import BytesIO

nest_asyncio.apply()

# ==================== 深色主题CSS + 移动端适配 ====================
st.set_page_config(page_title="VAI v9.0 终极优化版", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .css-1d391kg { background-color: #161b22; }
    .stMetric { background-color: #21262d; border-radius: 8px; padding: 10px; }
    .stButton>button { background-color: #21262d; color: white; border: 1px solid #30363d; }
    .stButton>button:hover { background-color: #30363d; }
    /* 移动端适配：当屏幕宽度小于800px时，三列变为单列 */
    @media (max-width: 800px) {
        .css-1r6slb0 { flex-direction: column; }
        .css-1r6slb0 > div { width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== 从环境变量或secrets读取API密钥 ====================
def get_api_keys():
    api_key = ""
    secret = ""
    try:
        api_key = st.secrets.get("API_KEY", "")
        secret = st.secrets.get("SECRET", "")
    except:
        pass
    if not api_key:
        api_key = os.environ.get("BINANCE_API_KEY", "")
    if not secret:
        secret = os.environ.get("BINANCE_SECRET", "")
    return api_key, secret

# ==================== 配置 ====================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ACCOUNT_BALANCE = 10000.0
LEVERAGE = 100
MAX_TOTAL_RISK = 0.55
TIMEFRAMES = ['5m', '15m', '1h']

# 交易所配置
EXCHANGES = [
    {'name': 'binance', 'class': ccxt.binance, 'options': {'defaultType': 'future'}},
    {'name': 'okx', 'class': ccxt.okx, 'options': {'defaultType': 'swap'}},
    {'name': 'bybit', 'class': ccxt.bybit, 'options': {'defaultType': 'linear'}},
]

# ==================== 会话状态初始化 ====================
defaults = {
    'real_trading': False,
    'dry_run': True,
    'api_key': '',
    'secret': '',
    'positions': {sym: None for sym in SYMBOLS},
    'trade_log': [],
    'equity_history': [ACCOUNT_BALANCE],
    'signal_history': {sym: [] for sym in SYMBOLS},
    'hf_history': {sym: [] for sym in SYMBOLS},
    'strategy_weights': {sym: {'main': 0.62, 'hf': 0.38} for sym in SYMBOLS},
    'sim_step': 0,
    'best_params': None,
    'replay_step': 0,
    'replay_data': {},
    'heatmap_last_update': datetime.now(),
    'sim_prices': {},
    'daily_trade_count': 0,
    'last_trade_day': datetime.now().date(),
    'pending_signals': [],
    'total_trades': 0,
    'winning_trades': 0,
    'total_pnl': 0.0,
    'max_trades_per_day': 30,
    'preferred_exchange': 'binance',
    'use_hf': True,                     # 是否使用高频策略
    'use_ema_filter': True,              # 是否使用EMA趋势过滤
    'max_drawdown_pct': 20.0,            # 最大回撤百分比（超过则暂停开仓）
    'daily_loss_limit': 500.0,           # 每日亏损限额（超过则暂停开仓）
    'peak_equity': ACCOUNT_BALANCE,      # 历史最高权益（用于回撤计算）
    'trading_paused': False,              # 是否暂停开仓
    'opt_results': None,                  # 参数优化结果
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = {sym: None for sym in SYMBOLS}
if 'cached_ohlcv' not in st.session_state:
    st.session_state.cached_ohlcv = {}

# ==================== 自动填充API密钥 ====================
api_key_from_env, secret_from_env = get_api_keys()
if api_key_from_env and not st.session_state.api_key:
    st.session_state.api_key = api_key_from_env
if secret_from_env and not st.session_state.secret:
    st.session_state.secret = secret_from_env

# ==================== 模拟K线生成（仅作为最后回退）====================
def generate_simulated_ohlcv(symbol, timeframe, limit=300):
    key = f"{symbol}_{timeframe}"
    st.session_state.sim_step += 1
    np.random.seed(hash(key + str(st.session_state.sim_step)) % 2**32)
    if key not in st.session_state.sim_prices:
        base = 62500 if 'BTC' in symbol else 3350 if 'ETH' in symbol else 142
        st.session_state.sim_prices[key] = [base] * limit
    else:
        base = st.session_state.sim_prices[key][-1]
    
    prices = [base]
    vol = 0.014
    for _ in range(limit-1):
        vol = max(0.007, min(0.048, vol*0.968 + np.random.normal(0, 0.0028)))
        ret = t.rvs(df=3.8, loc=np.random.normal(0,0.00008), scale=vol)
        prices.append(prices[-1]*(1+ret))
    prices = np.array(prices)
    
    freq_map = {'5m': '5T', '15m': '15T', '1h': '1H'}
    freq = freq_map.get(timeframe, '15T')
    end_time = datetime.now()
    ts = pd.date_range(end=end_time, periods=limit, freq=freq)
    
    df = pd.DataFrame({
        'timestamp': ts,
        'open': prices*(1+np.random.uniform(-0.0028,0.0028,limit)),
        'high': prices*(1+np.abs(np.random.randn(limit))*0.009),
        'low': prices*(1-np.abs(np.random.randn(limit))*0.009),
        'close': prices,
        'volume': np.random.lognormal(8.7,0.55,limit).astype(int)
    })
    st.session_state.sim_prices[key] = prices
    return df

# ==================== 多交易所数据获取（带重试和缓存）====================
@retry(tries=3, delay=1, backoff=2)
def fetch_from_exchange(exch, exch_symbol, timeframe, limit, days_back):
    if days_back:
        since = int((datetime.now() - timedelta(days=days_back)).timestamp()*1000)
        return ex.fetch_ohlcv(exch_symbol, timeframe, since=since, limit=limit)
    else:
        return ex.fetch_ohlcv(exch_symbol, timeframe, limit=limit)

def fetch_ohlcv(symbol, timeframe, limit=300, days_back=None):
    cache_key = f"{symbol}_{timeframe}_{limit}"
    now = datetime.now()
    if cache_key in st.session_state.cached_ohlcv:
        cached_time, cached_df = st.session_state.cached_ohlcv[cache_key]
        if (now - cached_time).seconds < 20:
            return cached_df

    df = None
    for exch in EXCHANGES:
        try:
            ex = exch['class']({
                'enableRateLimit': True,
                'options': exch['options']
            })
            exch_symbol = symbol
            if exch['name'] == 'okx' and '/USDT' in symbol:
                exch_symbol = symbol.replace('/USDT', '/USDT:USDT')
            if exch['name'] == 'bybit' and '/USDT' in symbol:
                exch_symbol = symbol.replace('/USDT', '/USDT:USDT')
            ohlcv = fetch_from_exchange(ex, exch_symbol, timeframe, limit, days_back)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} 使用 {exch['name']} 数据源")
            break
        except Exception as e:
            continue

    if df is None:
        st.warning(f"⚠️ 所有交易所均获取数据失败，临时使用模拟数据")
        df = generate_simulated_ohlcv(symbol, timeframe, limit)

    st.session_state.cached_ohlcv[cache_key] = (now, df)
    return df

# ==================== 技术指标（增强版）====================
def add_indicators(df):
    if len(df) < 90:
        return df
    df = df.copy()
    df['ema12'] = ta.trend.ema_indicator(df['close'],12)
    df['ema26'] = ta.trend.ema_indicator(df['close'],26)
    df['ema20'] = ta.trend.ema_indicator(df['close'],20)
    df['ema50'] = ta.trend.ema_indicator(df['close'],50)
    df['rsi'] = ta.momentum.rsi(df['close'],14)
    df['adx'] = ta.trend.adx(df['high'],df['low'],df['close'],14)
    df['atr'] = ta.volatility.average_true_range(df['high'],df['low'],df['close'],14)
    bb = ta.volatility.BollingerBands(df['close'],20,2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper']-df['bb_lower'])/df['close']
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume']/df['volume_sma']
    df['recent_high_20'] = df['high'].rolling(20).max().shift(1)
    df['recent_low_20'] = df['low'].rolling(20).min().shift(1)
    df['atr_ma100'] = df['atr'].rolling(100).mean()
    df['bb_width_rank50'] = df['bb_width'].rolling(50).rank(pct=True) <= 0.22
    df['adx_below25'] = df['adx'] < 25
    df['adx_streak'] = df['adx_below25'].groupby((df['adx_below25'] != df['adx_below25'].shift()).cumsum()).cumsum()
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    # OBV
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    # 布林带% B
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    return df

# ==================== 多时间框架信号 ====================
def multi_tf_signal(symbol):
    signals = {}
    for tf in TIMEFRAMES:
        df = add_indicators(fetch_ohlcv(symbol, tf))
        if len(df) < 110:
            signals[tf] = "无数据"
            continue
        _, main_plan, main_dir = main_signal(df, symbol)
        _, _, hf_dir = hf_signal(df, symbol)
        final_dir = main_dir or hf_dir
        signals[tf] = final_dir if final_dir else "观望"
    return signals

def parse_dir(sig_str):
    if sig_str == "多" or "多" in str(sig_str):
        return '多'
    elif sig_str == "空" or "空" in str(sig_str):
        return '空'
    else:
        return None

# ==================== 交易逻辑 ====================
def get_exchange():
    if not (st.session_state.real_trading and st.session_state.api_key and st.session_state.secret):
        return None
    return ccxt.binance({
        'apiKey': st.session_state.api_key,
        'secret': st.session_state.secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

def execute_real_order(symbol, side, size_usdt):
    ex = get_exchange()
    if not ex or st.session_state.dry_run:
        return "✅ 乾跑模式执行成功"
    try:
        ex.set_leverage(LEVERAGE, symbol)
        ticker = ex.fetch_ticker(symbol)
        price = ticker['last']
        amount = round((size_usdt * LEVERAGE)/price,6)
        if side=='多':
            order = ex.create_market_buy_order(symbol,amount)
        else:
            order = ex.create_market_sell_order(symbol,amount)
        return f"✅ 真实订单成功 ID: {order['id']}"
    except Exception as e:
        return f"❌ 下单失败: {e}"

def get_current_price(symbol):
    df = fetch_ohlcv(symbol,'5m',5)
    return df['close'].iloc[-1]

def check_risk_limits():
    """检查风险限制，返回是否允许开仓"""
    current_equity = st.session_state.equity_history[-1]
    # 更新峰值
    if current_equity > st.session_state.peak_equity:
        st.session_state.peak_equity = current_equity
    # 计算当前回撤
    drawdown = (st.session_state.peak_equity - current_equity) / st.session_state.peak_equity * 100
    if drawdown > st.session_state.max_drawdown_pct:
        st.session_state.trading_paused = True
        st.warning(f"🚫 回撤 {drawdown:.1f}% 超过限制 {st.session_state.max_drawdown_pct}%，开仓暂停")
        return False
    # 检查每日亏损
    today_pnl = sum(log for log in st.session_state.trade_log if '平仓' in log and 'PnL' in log)
    # 简单从权益变化计算当日盈亏
    if len(st.session_state.equity_history) > 1:
        start_equity = st.session_state.equity_history[0] if st.session_state.last_trade_day != datetime.now().date() else ACCOUNT_BALANCE
        day_pnl = current_equity - start_equity
        if day_pnl < -st.session_state.daily_loss_limit:
            st.session_state.trading_paused = True
            st.warning(f"🚫 今日亏损 {-day_pnl:.2f} 超过限制 {st.session_state.daily_loss_limit}，开仓暂停")
            return False
    st.session_state.trading_paused = False
    return True

def open_position(symbol, side, entry, stop, size, current_price):
    # 风险检查
    if not check_risk_limits():
        st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} 开仓被风险控制阻止")
        return

    today = datetime.now().date()
    if st.session_state.last_trade_day != today:
        st.session_state.daily_trade_count = 0
        st.session_state.last_trade_day = today
        process_pending_signals()
    if st.session_state.daily_trade_count >= st.session_state.max_trades_per_day:
        st.session_state.pending_signals.append({
            'symbol': symbol, 'side': side, 'entry': entry,
            'stop': stop, 'size': size, 'time': datetime.now()
        })
        st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} 信号加入排队（已达日上限）")
        return
    if st.session_state.real_trading and not st.session_state.dry_run:
        msg = execute_real_order(symbol, side, size)
    else:
        msg = f"模拟开仓 {side} {size:.0f}USDT"
    st.session_state.positions[symbol] = {
        'side': side,
        'entry': entry,
        'stop': stop,
        'size': size,
        'breakeven': False
    }
    st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} 开仓 {symbol} {side} @{entry:.2f} 止损{stop:.2f} {msg}")
    st.session_state.daily_trade_count += 1

def close_position(symbol, pos, price, reason):
    pnl = pos['size'] * ((price/pos['entry']-1) if pos['side']=='多' else (1-price/pos['entry'])) * LEVERAGE
    st.session_state.total_trades += 1
    st.session_state.total_pnl += pnl
    if pnl > 0:
        st.session_state.winning_trades += 1

    if st.session_state.real_trading and not st.session_state.dry_run:
        close_side = '空' if pos['side']=='多' else '多'
        msg = execute_real_order(symbol, close_side, pos['size'])
    else:
        msg = "模拟平仓"
    st.session_state.positions[symbol] = None
    st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} 平仓 {symbol} {pos['side']} @{price:.2f} PnL:{pnl:.2f} 原因:{reason} {msg}")
    st.session_state.equity_history.append(st.session_state.equity_history[-1]+pnl)

def emergency_close_all():
    for symbol in SYMBOLS:
        pos = st.session_state.positions.get(symbol)
        if pos:
            close_position(symbol, pos, get_current_price(symbol), "紧急平仓")
    st.success("🚨 已执行紧急全平仓！")

def process_pending_signals():
    today = datetime.now().date()
    if st.session_state.last_trade_day != today:
        return
    remaining = st.session_state.max_trades_per_day - st.session_state.daily_trade_count
    if remaining <= 0 or not st.session_state.pending_signals:
        return
    for i in range(min(remaining, len(st.session_state.pending_signals))):
        sig = st.session_state.pending_signals.pop(0)
        price = get_current_price(sig['symbol'])
        open_position(sig['symbol'], sig['side'], price, sig['stop'], sig['size'], price)

# ==================== 信号策略 ====================
def main_signal(df, symbol):
    if len(df)<110: return "数据不足", None, None
    last=df.iloc[-1]
    price,atr=last['close'],last.get('atr',0)
    atr_ma=last.get('atr_ma100',atr*1.2)
    compression=(atr<0.78*atr_ma) and last.get('bb_width_rank50',False) and last.get('adx_streak',0)>=6
    if not compression: return "压缩中",None,None
    if price>last.get('recent_high_20',price):
        stop=min(last.get('recent_low_20',price*0.96),price-atr*st.session_state.get('ATR_STOP_MULT',1.2))
        risk=(price-stop)/price+0.0012
        size=min(ACCOUNT_BALANCE*st.session_state.get('RISK_PER_TRADE',0.02)/risk,ACCOUNT_BALANCE*MAX_TOTAL_RISK)
        plan={'方向':'多','入场':price,'止损':stop,'仓位':size}
        return "多头突破 🔥",plan,'多'
    elif price<last.get('recent_low_20',price):
        stop=max(last.get('recent_high_20',price*1.04),price+atr*st.session_state.get('ATR_STOP_MULT',1.2))
        risk=(stop-price)/price+0.0012
        size=min(ACCOUNT_BALANCE*st.session_state.get('RISK_PER_TRADE',0.02)/risk,ACCOUNT_BALANCE*MAX_TOTAL_RISK)
        plan={'方向':'空','入场':price,'止损':stop,'仓位':size}
        return "空头突破 🔥",plan,'空'
    return "等待突破",None,None

def hf_signal(df, symbol):
    if not st.session_state.use_hf:
        return None, None, None
    if len(df)<25: return None,None,None
    last=df.iloc[-1]
    if last['volume_ratio']<=1.65: return None,None,None
    direction='多' if last['rsi']>60 else '空' if last['rsi']<40 else None
    if not direction: return None,None,None
    hist=st.session_state.hf_history[symbol]
    streak=sum(1 for x in reversed(hist) if x>0) if hist and hist[-1]>0 else -sum(1 for x in reversed(hist) if x<0) if hist else 0
    mult=max(0.55,min(2.1,1+streak*0.18))
    size_usdt=ACCOUNT_BALANCE*st.session_state.get('HF_MAX_POS',0.15)*mult
    return f"HF {direction} {size_usdt:.0f}USDT",size_usdt,direction

def calculate_signal_strength(df, symbol):
    """计算信号强度（0-100）"""
    strength = 50  # 默认中性
    last = df.iloc[-1]
    # 基于RSI偏离度
    rsi = last['rsi']
    if rsi > 70:
        strength += 20
    elif rsi < 30:
        strength += 20
    # 基于成交量放大
    if last['volume_ratio'] > 2:
        strength += 10
    # 基于ADX趋势强度
    adx = last['adx']
    if adx > 25:
        strength += 10
    # 基于布林带宽度（低宽度预示突破）
    if last['bb_width'] < last['bb_width'].rolling(50).mean().iloc[-1]:
        strength += 10
    return min(100, max(0, strength))

# ==================== 异步信号处理 ====================
async def process_single_symbol(symbol):
    df = fetch_ohlcv(symbol, '5m', limit=300)
    df = add_indicators(df)
    last_row = df.iloc[-1]
    current_price = last_row['close']
    last_time = last_row['timestamp']
    if st.session_state.last_signal_time[symbol] == last_time:
        return
    st.session_state.last_signal_time[symbol] = last_time

    pos = st.session_state.positions.get(symbol)

    # 止损检查
    if pos and ((pos['side']=='多' and current_price<=pos['stop']) or (pos['side']=='空' and current_price>=pos['stop'])):
        close_position(symbol, pos, current_price, "止损")
        pos = None

    # 移动止损（保本）
    if pos and not pos.get('breakeven', False):
        atr = df['atr'].iloc[-1]
        if pos['side'] == '多':
            if current_price - pos['entry'] > atr:
                pos['stop'] = pos['entry']
                pos['breakeven'] = True
                st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} {symbol} 移动止损至保本")
        else:
            if pos['entry'] - current_price > atr:
                pos['stop'] = pos['entry']
                pos['breakeven'] = True
                st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} {symbol} 移动止损至保本")

    # 止盈检查
    if pos:
        take_profit_mult = st.session_state.get('TAKE_PROFIT_MULT', 2.0)
        atr = df['atr'].iloc[-1]
        if pos['side'] == '多':
            take_profit_price = pos['entry'] + atr * take_profit_mult
            if current_price >= take_profit_price:
                close_position(symbol, pos, current_price, "止盈")
                pos = None
        else:
            take_profit_price = pos['entry'] - atr * take_profit_mult
            if current_price <= take_profit_price:
                close_position(symbol, pos, current_price, "止盈")
                pos = None

    # 获取新信号
    _, main_plan, main_dir = main_signal(df, symbol)
    _, hf_size, hf_dir = hf_signal(df, symbol)

    # 获取多周期信号
    tf_signals = multi_tf_signal(symbol)
    dir_5m = parse_dir(tf_signals.get('5m', ''))
    dir_15m = parse_dir(tf_signals.get('15m', ''))
    dir_1h = parse_dir(tf_signals.get('1h', ''))

    # 三个周期必须完全一致
    if not (dir_5m and dir_15m and dir_1h and dir_5m == dir_15m == dir_1h):
        return

    # 趋势过滤：1小时EMA50（可选）
    if st.session_state.use_ema_filter:
        df_1h = fetch_ohlcv(symbol, '1h', limit=100)
        if len(df_1h) >= 50:
            ema50_1h = ta.trend.ema_indicator(df_1h['close'], 50).iloc[-1]
            if dir_5m == '多' and current_price < ema50_1h:
                return
            if dir_5m == '空' and current_price > ema50_1h:
                return

    # 主策略与高频必须共振且方向一致
    if not (main_dir and hf_dir and main_dir == hf_dir and main_dir == dir_5m):
        return

    # 所有条件满足
    size = main_plan['仓位']
    stop = main_plan['止损']
    entry = main_plan['入场']

    # 记录信号
    st.session_state.signal_history[symbol].append({
        'time': last_time, 'price': entry, 'side': main_dir,
        'type': '共振', 'size': size
    })
    if hf_dir:
        st.session_state.hf_history[symbol].append(1 if hf_dir=='多' else -1)

    # 反向信号平仓
    if pos and pos['side'] != main_dir:
        close_position(symbol, pos, current_price, "反向信号")
        pos = None

    # 开新仓
    if not pos:
        open_position(symbol, main_dir, entry, stop, size, current_price)

async def process_all_symbols():
    tasks = [process_single_symbol(sym) for sym in SYMBOLS]
    await asyncio.gather(*tasks)

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)

# ==================== 回测 ====================
def run_backtest(symbol, days=60, params=None):
    if params:
        # 使用优化参数覆盖
        for k, v in params.items():
            st.session_state[k] = v
    df = fetch_ohlcv(symbol, '15m', limit=days*96, days_back=days)
    df = add_indicators(df)
    equity = [ACCOUNT_BALANCE] * len(df)
    signals = []
    trades = []
    pos = None
    for i in range(100, len(df)):
        sub_df = df.iloc[:i+1]
        _, main_plan, _ = main_signal(sub_df, symbol)
        _, hf_size, hf_dir = hf_signal(sub_df, symbol)
        final_plan = main_plan or ({'方向': hf_dir, '入场': df.iloc[i]['close'],
                                    '止损': df.iloc[i]['close']*(0.995 if hf_dir=='多' else 1.005),
                                    '仓位': hf_size} if hf_dir else None)
        current_price = df.iloc[i]['close']
        if final_plan and not pos:
            side = final_plan['方向']
            entry = final_plan['入场']
            stop = final_plan['止损']
            size = final_plan['仓位']
            pos = {'side': side, 'entry': entry, 'stop': stop, 'size': size}
            signals.append({'idx': i, 'time': df.iloc[i]['timestamp'], 'price': entry,
                            'action': 'entry', 'side': side})
            trades.append({'time': df.iloc[i]['timestamp'], 'action': '开仓', 'side': side, 'price': entry})
        if pos:
            hit_sl = (pos['side'] == '多' and current_price <= pos['stop']) or \
                     (pos['side'] == '空' and current_price >= pos['stop'])
            if hit_sl:
                pnl = pos['size'] * ((current_price/pos['entry']-1) if pos['side']=='多' else (1-current_price/pos['entry'])) * LEVERAGE
                equity[i] = equity[i-1] + pnl
                signals.append({'idx': i, 'time': df.iloc[i]['timestamp'], 'price': current_price,
                                'action': 'exit', 'side': pos['side'], 'pnl': round(pnl,2)})
                trades.append({'time': df.iloc[i]['timestamp'], 'action': '平仓', 'side': pos['side'],
                               'price': current_price, 'pnl': pnl})
                pos = None
            else:
                equity[i] = equity[i-1]
        else:
            equity[i] = equity[i-1]
    for i in range(100):
        equity[i] = ACCOUNT_BALANCE
    return df, equity, signals, trades

def optimize_parameters(symbol, days=60):
    """简单网格搜索优化参数"""
    best_sharpe = -np.inf
    best_params = None
    # 待优化的参数范围
    risk_range = [1.0, 1.5, 2.0, 2.5]
    atr_range = [1.0, 1.2, 1.5, 1.8]
    hf_pos_range = [10, 15, 20]
    results = []
    for risk in risk_range:
        for atr in atr_range:
            for hf in hf_pos_range:
                params = {
                    'RISK_PER_TRADE': risk/100,
                    'ATR_STOP_MULT': atr,
                    'HF_MAX_POS': hf/100
                }
                df, equity, signals, trades = run_backtest(symbol, days, params)
                # 计算夏普比率
                returns = np.diff(equity) / ACCOUNT_BALANCE
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
                win_rate = len([t for t in trades if 'pnl' in t and t['pnl']>0]) / len(trades) if trades else 0
                results.append({
                    'risk': risk, 'atr': atr, 'hf': hf,
                    'sharpe': sharpe, 'win_rate': win_rate,
                    'trades': len(trades)
                })
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
    return best_params, pd.DataFrame(results)

# ==================== 热图 ====================
def create_dynamic_heatmap():
    data = []
    for sym in SYMBOLS:
        pos = st.session_state.positions.get(sym)
        if pos:
            price = get_current_price(sym)
            unreal = pos['size'] * ((price/pos['entry']-1) if pos['side']=='多' else (1-price/pos['entry'])) * LEVERAGE
            risk_pct = pos['size']/ACCOUNT_BALANCE*100
            data.append({'币种': sym, '方向': pos['side'], '仓位USDT': round(pos['size'],0),
                         '未实现PNL': round(unreal,1), '风险%': round(risk_pct,1),
                         '止损价': round(pos['stop'],2), '移动止损': '是' if pos.get('breakeven', False) else '否'})
        else:
            data.append({'币种': sym, '方向': '无', '仓位USDT': 0, '未实现PNL': 0, '风险%': 0,
                         '止损价': 0, '移动止损': '-'})
    df = pd.DataFrame(data).set_index('币种')
    fig = px.imshow(df[['仓位USDT','风险%','未实现PNL']], text_auto=True, aspect="auto",
                    color_continuous_scale='RdYlGn_r',
                    title=f"🔥 仓位热图（最后更新: {datetime.now().strftime('%H:%M:%S')}）")
    fig.update_layout(height=340)
    return fig, df

# ==================== 使用 st.fragment 实现增量更新图表 ====================
@st.fragment(run_every=25)
def update_chart(symbol):
    df_hf = add_indicators(fetch_ohlcv(symbol, '5m', limit=150))
    signals_tf = multi_tf_signal(symbol)
    consensus = "多" if list(signals_tf.values()).count('多') >= 2 else "空" if list(signals_tf.values()).count('空') >= 2 else "中性"
    st.caption(f"多TF共识：**{consensus}**")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.20, 0.25],
        vertical_spacing=0.02,
        subplot_titles=(f"{symbol} 价格", "成交量", "MACD")
    )
    # 价格K线
    fig.add_trace(go.Candlestick(
        x=df_hf['timestamp'],
        open=df_hf['open'],
        high=df_hf['high'],
        low=df_hf['low'],
        close=df_hf['close'],
        name="价格",
        increasing_line_color="#00ff9d",
        decreasing_line_color="#ff4d4d"
    ), row=1, col=1)

    # EMA20/50
    fig.add_trace(go.Scatter(
        x=df_hf['timestamp'], y=df_hf['ema20'],
        name="EMA20", line=dict(color="#ffaa00", width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_hf['timestamp'], y=df_hf['ema50'],
        name="EMA50", line=dict(color="#aa88ff", width=1.5)
    ), row=1, col=1)

    # MACD线
    fig.add_trace(go.Scatter(
        x=df_hf['timestamp'], y=df_hf['macd'],
        name="MACD", line=dict(color="#00b0ff")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_hf['timestamp'], y=df_hf['macd_signal'],
        name="信号线", line=dict(color="#ffd700")
    ), row=1, col=1)

    # 成交量
    colors = ['#00ff9d' if o < c else '#ff4d4d' for o, c in zip(df_hf['open'], df_hf['close'])]
    fig.add_trace(go.Bar(
        x=df_hf['timestamp'], y=df_hf['volume'],
        name="成交量", marker_color=colors, opacity=0.6
    ), row=2, col=1)

    # MACD柱
    colors_hist = ['#00ff9d' if h > 0 else '#ff4d4d' for h in df_hf['macd_diff']]
    fig.add_trace(go.Bar(
        x=df_hf['timestamp'], y=df_hf['macd_diff'],
        name="MACD柱", marker_color=colors_hist
    ), row=3, col=1)

    # 信号标注
    for sig in st.session_state.signal_history[symbol][-10:]:
        fig.add_annotation(
            x=sig['time'], y=sig['price'],
            text="▲" if sig['side']=='多' else "▼",
            showarrow=True, arrowhead=2, arrowsize=2,
            arrowcolor="lime" if sig['side']=='多' else "red",
            row=1, col=1
        )

    # 当前价格标签
    latest_price = df_hf['close'].iloc[-1]
    prev_price = df_hf['close'].iloc[-2]
    price_change = (latest_price - prev_price) / prev_price * 100
    price_label = f"当前: {latest_price:.2f} ({price_change:+.2f}%)"
    fig.add_annotation(
        x=df_hf['timestamp'].iloc[-1], y=latest_price,
        text=price_label,
        showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=2,
        arrowcolor="#ffffff", bgcolor="#21262d", bordercolor="#ffffff",
        borderwidth=1, borderpad=4, font=dict(color="white", size=12),
        ax=40, ay=-40, row=1, col=1
    )

    # 如果当前有持仓，显示止损/止盈水平线
    pos = st.session_state.positions.get(symbol)
    if pos:
        # 止损线
        fig.add_hline(y=pos['stop'], line_dash="dash", line_color="red",
                      annotation_text=f"止损 {pos['stop']:.2f}", row=1, col=1)
        # 止盈线（基于当前ATR估算）
        atr = df_hf['atr'].iloc[-1]
        take_profit_mult = st.session_state.get('TAKE_PROFIT_MULT', 2.0)
        if pos['side'] == '多':
            tp = pos['entry'] + atr * take_profit_mult
        else:
            tp = pos['entry'] - atr * take_profit_mult
        fig.add_hline(y=tp, line_dash="dash", line_color="lime",
                      annotation_text=f"止盈 {tp:.2f}", row=1, col=1)

    fig.update_layout(height=620, margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#ffffff"))
    st.plotly_chart(fig, use_container_width=True)

    # 详细多时间框架信号
    st.markdown("**多TF信号详情**")
    tf_cols = st.columns(3)
    for idx, (tf, sig) in enumerate(signals_tf.items()):
        tf_cols[idx].metric(tf, sig, delta_color="off")

    # 显示交易计划
    df_plan = add_indicators(fetch_ohlcv(symbol, '5m', limit=300))
    _, main_plan, main_dir = main_signal(df_plan, symbol)
    if main_plan:
        atr = df_plan['atr'].iloc[-1]
        take_profit_mult = st.session_state.get('TAKE_PROFIT_MULT', 2.0)
        tp_price = main_plan['入场'] + (atr * take_profit_mult) if main_dir == '多' else main_plan['入场'] - (atr * take_profit_mult)
        strength = calculate_signal_strength(df_plan, symbol)
        st.info(f"📊 **交易计划**：{main_dir} 进场 {main_plan['入场']:.2f}，止损 {main_plan['止损']:.2f}，止盈 {tp_price:.2f} | 强度 {strength}")
    else:
        st.caption("⏳ 无交易计划")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📊 VAI v9.0 终极版")
    st.metric("总权益", f"${st.session_state.equity_history[-1]:,.2f}")
    st.metric("今日已开单", f"{st.session_state.daily_trade_count}/{st.session_state.max_trades_per_day}")
    st.metric("排队信号数", len(st.session_state.pending_signals))
    if st.session_state.trading_paused:
        st.warning("⛔ 交易暂停（风控触发）")
    
    if st.button("🚨 紧急全平仓", type="primary", use_container_width=True):
        emergency_close_all()
        st.rerun()
    
    if st.button("🔄 重置会话", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==================== 主标题 ====================
st.markdown("# 🤖 AI 自进化交易平台 VAI v9.0 终极优化版", unsafe_allow_html=True)
st.caption("🌟 已开启多交易所切换 + 增强指标 + 止损功能 + 每25秒自动刷新")

# ==================== 主标签页 ====================
tab1, tab2, tab3, tab4 = st.tabs(["📈 实时交易", "🔙 回测中心", "📊 风险仪表板", "⚙️ 设定"])

with tab1:
    st.subheader("实时市场与信号")
    # 异步处理信号（放在循环外，避免重复）
    run_async(process_all_symbols())

    cols = st.columns(len(SYMBOLS))
    for i, symbol in enumerate(SYMBOLS):
        with cols[i]:
            st.subheader(symbol)
            # 使用 fragment 更新图表
            update_chart(symbol)

with tab2:
    st.header("🔙 回测中心")
    subtab1, subtab2, subtab3 = st.tabs(["单币种回测", "多币种并行回放", "参数优化器"])

    with subtab1:
        st.info("单币种回测功能（可扩展）")

    with subtab2:
        st.subheader("🎬 多币种并行回放")
        selected_symbols = st.multiselect("选择要并行回放的币种", SYMBOLS, default=SYMBOLS[:2])
        bt_days_multi = st.slider("回测天数（并行）", 7, 120, 45)

        if st.button("🚀 生成多币种回放数据", key="multi_replay_btn"):
            with st.spinner("正在为多币种生成回放数据..."):
                st.session_state.replay_data = {}
                for sym in selected_symbols:
                    df, equity, signals, trades = run_backtest(sym, bt_days_multi)
                    st.session_state.replay_data[sym] = {'df': df, 'equity': equity, 'signals': signals, 'trades': trades}
                st.success(f"已为 {len(selected_symbols)} 个币种生成回放数据！")

        if st.session_state.replay_data:
            max_len = max(len(d['df']) for d in st.session_state.replay_data.values())
            step = st.slider("同步回放进度", 0, max_len-1, st.session_state.replay_step, key="multi_replay_slider")
            st.session_state.replay_step = step

            # 导出按钮
            if st.button("📥 导出回放数据为CSV"):
                all_trades = []
                for sym, data in st.session_state.replay_data.items():
                    for t in data['trades']:
                        t['symbol'] = sym
                        all_trades.append(t)
                df_trades = pd.DataFrame(all_trades)
                csv = df_trades.to_csv(index=False).encode('utf-8')
                st.download_button("下载交易记录", data=csv, file_name="backtest_trades.csv", mime="text/csv")

            replay_cols = st.columns(len(selected_symbols))
            for idx, sym in enumerate(selected_symbols):
                with replay_cols[idx]:
                    st.subheader(sym)
                    data = st.session_state.replay_data[sym]
                    replay_df = data['df'].iloc[:step+1]
                    replay_signals = [s for s in data['signals'] if s['idx'] <= step]

                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(
                        x=replay_df['timestamp'],
                        open=replay_df['open'],
                        high=replay_df['high'],
                        low=replay_df['low'],
                        close=replay_df['close']
                    ), row=1, col=1)
                    for s in replay_signals:
                        color = "lime" if s['action']=='entry' else "red"
                        symb = "▲" if s['action']=='entry' else "▼"
                        fig.add_annotation(
                            x=s['time'], y=s['price'],
                            text=symb, showarrow=True,
                            arrowhead=2, arrowsize=2.5, arrowcolor=color,
                            row=1, col=1
                        )
                    eq = data['equity'][:step+1]
                    fig.add_trace(go.Scatter(y=eq, name="权益", line=dict(color="#00ff88")), row=2, col=1)
                    fig.update_layout(height=520, title=f"{sym} 回放")
                    st.plotly_chart(fig, use_container_width=True)

    with subtab3:
        st.subheader("🔍 参数优化器")
        opt_symbol = st.selectbox("选择优化币种", SYMBOLS)
        opt_days = st.slider("回测天数", 30, 120, 60)
        if st.button("开始优化"):
            with st.spinner("正在网格搜索最优参数..."):
                best_params, results_df = optimize_parameters(opt_symbol, opt_days)
                st.session_state.opt_results = results_df
                st.session_state.best_params = best_params
                st.success("优化完成！")
                st.write("最优参数：", best_params)
                st.dataframe(results_df)
                # 导出优化结果
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("下载优化结果", data=csv, file_name="optimization_results.csv", mime="text/csv")

with tab3:
    st.header("📊 风险仪表板")
    st.subheader("🔥 仓位热图（含移动止损状态）")
    heat_fig, heat_df = create_dynamic_heatmap()
    st.plotly_chart(heat_fig, use_container_width=True)
    st.dataframe(heat_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

    st.subheader("📈 策略性能雷达图")
    # 动态计算实际指标
    if st.session_state.total_trades > 0:
        win_rate = st.session_state.winning_trades / st.session_state.total_trades * 100
        avg_win = st.session_state.total_pnl / st.session_state.total_trades if st.session_state.total_trades > 0 else 0
        # 简单估算夏普等（实际需更复杂计算）
        metrics = {
            '胜率': win_rate,
            '总盈亏': st.session_state.total_pnl,
            '交易次数': st.session_state.total_trades,
            'Sharpe(估)': 1.0,  # 占位
            'Profit Factor': 1.0,
        }
    else:
        metrics = {'胜率': 0, '总盈亏': 0, '交易次数': 0, 'Sharpe(估)': 0, 'Profit Factor': 0}
    fig_radar = px.line_polar(
        pd.DataFrame([metrics]),
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        line_close=True,
        title="策略性能雷达图"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("📊 交易统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总交易次数", st.session_state.total_trades)
    with col2:
        win_rate = (st.session_state.winning_trades / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0
        st.metric("胜率", f"{win_rate:.1f}%")
    with col3:
        st.metric("总盈亏", f"${st.session_state.total_pnl:.2f}")
    with col4:
        if st.button("重置统计"):
            st.session_state.total_trades = 0
            st.session_state.winning_trades = 0
            st.session_state.total_pnl = 0.0
            st.rerun()

    st.subheader("交易日志")
    log_df = pd.DataFrame(st.session_state.trade_log[-50:], columns=["记录"])
    st.dataframe(log_df, use_container_width=True)
    # 导出日志
    if st.button("导出日志"):
        df_log = pd.DataFrame(st.session_state.trade_log, columns=["记录"])
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("下载日志", data=csv, file_name="trade_log.csv", mime="text/csv")

with tab4:
    st.header("⚙️ 设定")
    st.session_state.real_trading = st.checkbox("启用真实交易", st.session_state.real_trading)
    st.session_state.dry_run = st.checkbox("乾跑模式（不下真实单）", st.session_state.dry_run)
    if st.session_state.real_trading:
        st.session_state.api_key = st.text_input("Binance API Key", st.session_state.api_key, type="password")
        st.session_state.secret = st.text_input("Binance Secret", st.session_state.secret, type="password")
        if not st.session_state.api_key or not st.session_state.secret:
            st.warning("请输入API密钥或使用 Streamlit Secrets / 环境变量配置")

    st.subheader("策略参数")
    col1, col2 = st.columns(2)
    with col1:
        st.slider("每笔风险 (%)", 1.0, 5.0, float(st.session_state.get('RISK_PER_TRADE', 2.0)*100), 0.1, key="RISK_PER_TRADE", format="%.1f")
        st.slider("高频最大仓位 (%)", 5.0, 30.0, float(st.session_state.get('HF_MAX_POS', 15.0)*100), 1.0, key="HF_MAX_POS", format="%.1f")
    with col2:
        st.slider("ATR止损倍数", 0.8, 2.5, st.session_state.get('ATR_STOP_MULT', 1.2), 0.05, key="ATR_STOP_MULT")
        st.slider("止盈倍数 (ATR倍数)", 1.0, 5.0, st.session_state.get('TAKE_PROFIT_MULT', 2.0), 0.1, key="TAKE_PROFIT_MULT")
    st.number_input("每日开单上限", min_value=1, max_value=100, value=st.session_state.max_trades_per_day, key="max_trades_per_day")
    st.selectbox("首选数据源交易所", ["binance", "okx", "bybit"], key="preferred_exchange")
    if st.button("更新首选交易所"):
        st.success("首选交易所已更新")

    st.subheader("策略开关")
    st.checkbox("使用高频策略", value=st.session_state.use_hf, key="use_hf")
    st.checkbox("使用EMA趋势过滤", value=st.session_state.use_ema_filter, key="use_ema_filter")

    st.subheader("风险控制")
    st.number_input("最大回撤百分比 (%)", min_value=5.0, max_value=50.0, value=st.session_state.max_drawdown_pct, key="max_drawdown_pct")
    st.number_input("每日亏损限额 (USDT)", min_value=100.0, max_value=5000.0, value=st.session_state.daily_loss_limit, key="daily_loss_limit")
    if st.button("应用风险设置"):
        st.success("风险设置已更新")

    if st.button("保存日志到文件"):
        df_log = pd.DataFrame(st.session_state.trade_log, columns=["记录"])
        df_log.to_csv("trade_log.csv", index=False)
        st.success("日志已保存到 trade_log.csv")

# ==================== 自动刷新 ====================
st_autorefresh(interval=25000, key="auto_refresh")
st.markdown("""
<div style="text-align:center; color:#666; font-size:14px;">
    ⭐ 短线优化版 VAI v9.0 终极优化版 · 每25秒自动刷新 · 多交易所 + 增强指标 + 移动止损/止盈
</div>
""", unsafe_allow_html=True)
