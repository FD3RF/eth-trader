# -*- coding: utf-8 -*-
"""
🤖 AI 自进化交易平台 VAI v9.0 终极整合版
===========================================================
功能：
- 多周期共振策略（5m/15m/1h）
- 主策略与高频策略融合
- 移动止损、止盈、每日开单上限（动态配置）
- 多交易所自动切换（Binance/OKX/Bybit）
- 实时交易界面采用深色专业风格，三币种三层图表（价格+成交量+MACD）
- 每个币种下方显示详细多时间框架信号
- 完整回测中心、风险仪表板、交易统计
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

nest_asyncio.apply()

# ==================== 深色主题CSS ====================
st.set_page_config(page_title="VAI v9.0 终极整合版", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .css-1d391kg { background-color: #161b22; }
    .stMetric { background-color: #21262d; border-radius: 8px; padding: 10px; }
    .stButton>button { background-color: #21262d; color: white; border: 1px solid #30363d; }
    .stButton>button:hover { background-color: #30363d; }
</style>
""", unsafe_allow_html=True)

# ==================== 配置 ====================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ACCOUNT_BALANCE = 10000.0
LEVERAGE = 100
MAX_TOTAL_RISK = 0.55
TIMEFRAMES = ['5m', '15m', '1h']

# 交易所配置（按优先级）
EXCHANGES = [
    {'name': 'binance', 'class': ccxt.binance, 'options': {'defaultType': 'future'}},
    {'name': 'okx', 'class': ccxt.okx, 'options': {'defaultType': 'swap'}},
    {'name': 'bybit', 'class': ccxt.bybit, 'options': {'defaultType': 'linear'}},
]

# ==================== 会话状态初始化 ====================
defaults = {
    'use_simulated': True,
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
    'max_trades_per_day': 30,          # 每日开单上限
    'preferred_exchange': 'binance',    # 首选交易所
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = {sym: None for sym in SYMBOLS}
if 'cached_ohlcv' not in st.session_state:
    st.session_state.cached_ohlcv = {}

# ==================== 模拟K线生成 ====================
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

# ==================== 多交易所数据获取 ====================
def fetch_ohlcv(symbol, timeframe, limit=300, days_back=None):
    cache_key = f"{symbol}_{timeframe}_{limit}"
    now = datetime.now()
    if cache_key in st.session_state.cached_ohlcv:
        cached_time, cached_df = st.session_state.cached_ohlcv[cache_key]
        if (now - cached_time).seconds < 20:
            return cached_df

    if st.session_state.use_simulated:
        df = generate_simulated_ohlcv(symbol, timeframe, limit)
    else:
        df = None
        # 按优先级尝试交易所
        for exch in EXCHANGES:
            try:
                ex = exch['class']({
                    'enableRateLimit': True,
                    'options': exch['options']
                })
                # 对 symbol 进行可能的格式转换
                exch_symbol = symbol
                if exch['name'] == 'okx' and '/USDT' in symbol:
                    exch_symbol = symbol.replace('/USDT', '/USDT:USDT')
                if exch['name'] == 'bybit' and '/USDT' in symbol:
                    exch_symbol = symbol.replace('/USDT', '/USDT:USDT')
                if days_back:
                    since = int((datetime.now() - timedelta(days=days_back)).timestamp()*1000)
                    ohlcv = ex.fetch_ohlcv(exch_symbol, timeframe, since=since, limit=limit)
                else:
                    ohlcv = ex.fetch_ohlcv(exch_symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                st.session_state.trade_log.append(f"{datetime.now().strftime('%H:%M')} 使用 {exch['name']} 数据源")
                break
            except Exception as e:
                continue
        if df is None:
            st.warning("所有交易所均获取失败，使用模拟数据")
            df = generate_simulated_ohlcv(symbol, timeframe, limit)

    st.session_state.cached_ohlcv[cache_key] = (now, df)
    return df

# ==================== 技术指标（增强版）====================
def add_indicators(df):
    if len(df) < 90:
        return df
    df = df.copy()
    # 原有指标
    df['ema12'] = ta.trend.ema_indicator(df['close'],12)
    df['ema26'] = ta.trend.ema_indicator(df['close'],26)
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
    # VWAP（近似）
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
        signals[tf] = f"{final_dir} (强)" if main_dir and hf_dir else f"{final_dir} (中)" if final_dir else "观望"
    return signals

def parse_dir(sig_str):
    if '多' in sig_str:
        return '多'
    elif '空' in sig_str:
        return '空'
    else:
        return None

# ==================== 交易逻辑 ====================
def get_exchange():
    if not (st.session_state.real_trading and st.session_state.api_key and st.session_state.secret):
        return None
    # 使用 Binance 进行交易（下单）
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

def open_position(symbol, side, entry, stop, size, current_price):
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
    # 更新统计
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

    # 移动止损（保本）——动态显示在持仓信息中，这里更新状态
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

    # 趋势过滤：1小时EMA50
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
def run_backtest(symbol, days=60):
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

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📊 VAI v9.0 终极版")
    st.metric("总权益", f"${st.session_state.equity_history[-1]:,.2f}")
    st.metric("今日已开单", f"{st.session_state.daily_trade_count}/{st.session_state.max_trades_per_day}")
    st.metric("排队信号数", len(st.session_state.pending_signals))
    
    if st.button("🚨 紧急全平仓", type="primary", use_container_width=True):
        emergency_close_all()
        st.rerun()
    
    if st.button("🔄 重置会话", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==================== 主标题 ====================
st.markdown("# 🤖 AI 自进化交易平台 VAI v9.0 终极整合版 · 多周期共振策略", unsafe_allow_html=True)
st.caption("🌟 已开启多交易所切换 + 增强指标 + 止盈/移动止损 · 每25秒自动刷新")

# ==================== 主标签页 ====================
tab1, tab2, tab3, tab4 = st.tabs(["📈 实时交易", "🔙 回测中心", "📊 风险仪表板", "⚙️ 设定"])

with tab1:
    st.subheader("实时市场与信号")
    run_async(process_all_symbols())

    cols = st.columns(len(SYMBOLS))
    for i, symbol in enumerate(SYMBOLS):
        with cols[i]:
            st.subheader(symbol)
            df_hf = add_indicators(fetch_ohlcv(symbol, '5m', limit=150))
            signals_tf = multi_tf_signal(symbol)
            consensus = "多" if any("多" in v for v in signals_tf.values()) else "空" if any("空" in v for v in signals_tf.values()) else "中性"
            st.caption(f"多TF共识：**{consensus}**")

            # 三层图表（价格+成交量+MACD）
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.55, 0.20, 0.25],
                vertical_spacing=0.02,
                subplot_titles=(f"{symbol} 价格", "成交量", "MACD")
            )
            # 价格K线 + MACD线 + 信号线
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
            fig.add_trace(go.Scatter(
                x=df_hf['timestamp'], y=df_hf['macd'],
                name="MACD", line=dict(color="#00b0ff")
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_hf['timestamp'], y=df_hf['macd_signal'],
                name="信号线", line=dict(color="#ffd700")
            ), row=1, col=1)
            # 成交量（绿涨红跌）
            colors = ['#00ff9d' if o < c else '#ff4d4d' for o, c in zip(df_hf['open'], df_hf['close'])]
            fig.add_trace(go.Bar(
                x=df_hf['timestamp'], y=df_hf['volume'],
                name="成交量", marker_color=colors
            ), row=2, col=1)
            # MACD柱
            colors_hist = ['#00ff9d' if h > 0 else '#ff4d4d' for h in df_hf['macd_diff']]
            fig.add_trace(go.Bar(
                x=df_hf['timestamp'], y=df_hf['macd_diff'],
                name="MACD柱", marker_color=colors_hist
            ), row=3, col=1)

            # 信号标注（最近10个）
            for sig in st.session_state.signal_history[symbol][-10:]:
                fig.add_annotation(
                    x=sig['time'], y=sig['price'],
                    text="▲" if sig['side']=='多' else "▼",
                    showarrow=True, arrowhead=2, arrowsize=2,
                    arrowcolor="lime" if sig['side']=='多' else "red",
                    row=1, col=1
                )

            fig.update_layout(height=620, margin=dict(t=30, b=10, l=10, r=10),
                              plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font=dict(color="#ffffff"))
            st.plotly_chart(fig, use_container_width=True)

            # 详细多时间框架信号显示
            st.markdown("**多TF信号详情**")
            tf_cols = st.columns(3)
            for idx, (tf, sig) in enumerate(signals_tf.items()):
                tf_cols[idx].metric(tf, sig, delta_color="off")

with tab2:
    st.header("🔙 回测中心")
    subtab1, subtab2, subtab3 = st.tabs(["单币种回测", "多币种并行回放", "策略参数优化器"])

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
                    df, equity, signals, _ = run_backtest(sym, bt_days_multi)
                    st.session_state.replay_data[sym] = {'df': df, 'equity': equity, 'signals': signals}
                st.success(f"已为 {len(selected_symbols)} 个币种生成回放数据！")

        if st.session_state.replay_data:
            max_len = max(len(d['df']) for d in st.session_state.replay_data.values())
            step = st.slider("同步回放进度", 0, max_len-1, st.session_state.replay_step, key="multi_replay_slider")
            st.session_state.replay_step = step

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
        st.info("参数优化功能待实现（可集成网格搜索）")

with tab3:
    st.header("📊 风险仪表板")
    st.subheader("🔥 仓位热图（含移动止损状态）")
    heat_fig, heat_df = create_dynamic_heatmap()
    st.plotly_chart(heat_fig, use_container_width=True)
    st.dataframe(heat_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

    st.subheader("📈 策略性能雷达图")
    # 示例指标（可根据回测结果动态计算）
    metrics = {'Sharpe': 1.8, 'Calmar': 2.1, 'Profit Factor': 1.65, 'Sortino': 2.3, '胜率': 58}
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
    log_df = pd.DataFrame(st.session_state.trade_log[-20:], columns=["记录"])
    st.dataframe(log_df, use_container_width=True)

with tab4:
    st.header("⚙️ 设定")
    st.session_state.use_simulated = st.checkbox("使用模拟数据", st.session_state.use_simulated)
    st.session_state.real_trading = st.checkbox("启用真实交易", st.session_state.real_trading)
    st.session_state.dry_run = st.checkbox("乾跑模式（不下真实单）", st.session_state.dry_run)
    if st.session_state.real_trading:
        api_key_from_secrets = st.secrets.get("API_KEY", "")
        secret_from_secrets = st.secrets.get("SECRET", "")
        if api_key_from_secrets and not st.session_state.api_key:
            st.session_state.api_key = api_key_from_secrets
        if secret_from_secrets and not st.session_state.secret:
            st.session_state.secret = secret_from_secrets
        st.session_state.api_key = st.text_input("Binance API Key", st.session_state.api_key, type="password")
        st.session_state.secret = st.text_input("Binance Secret", st.session_state.secret, type="password")
        if not st.session_state.api_key or not st.session_state.secret:
            st.warning("请输入API密钥或使用 Streamlit Secrets 配置")

    st.slider("每笔风险 (%)", 1.0, 5.0, 2.0, 0.1, key="RISK_PER_TRADE")
    st.slider("高频最大仓位 (%)", 5.0, 30.0, 15.0, 1.0, key="HF_MAX_POS")
    st.slider("ATR止损倍数", 0.8, 2.5, 1.2, 0.05, key="ATR_STOP_MULT")
    st.slider("止盈倍数 (ATR倍数)", 1.0, 5.0, 2.0, 0.1, key="TAKE_PROFIT_MULT")
    st.number_input("每日开单上限", min_value=1, max_value=100, value=st.session_state.max_trades_per_day, key="max_trades_per_day")
    st.selectbox("首选数据源交易所", ["binance", "okx", "bybit"], key="preferred_exchange")
    if st.button("更新首选交易所"):
        st.success("首选交易所已更新")

# ==================== 自动刷新 ====================
st_autorefresh(interval=25000, key="auto_refresh")
st.markdown("""
<div style="text-align:center; color:#666; font-size:14px;">
    ⭐ 短线优化版 VAI v9.0 终极版 · 每25秒自动刷新 · 多交易所 + 增强指标 + 移动止损/止盈
</div>
""", unsafe_allow_html=True)
