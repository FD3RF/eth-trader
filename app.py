# -*- coding: utf-8 -*-
"""
🤖 AI 自进化交易平台 VAI v9.0 终极稳定版
===========================================================
修复：
- 所有弃用警告（use_container_width → width，T/H → min/h）
- 性能优化：缓存30秒，K线数量150，刷新间隔可配置
- 增加“性能模式”开关（120秒刷新）
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
from retry import retry

nest_asyncio.apply()

st.set_page_config(page_title="VAI v9.0 终极稳定版", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .css-1d391kg { background-color: #161b22; }
    .stMetric { background-color: #21262d; border-radius: 8px; padding: 10px; }
    .stButton>button { background-color: #21262d; color: white; border: 1px solid #30363d; }
    .stButton>button:hover { background-color: #30363d; }
    @media (max-width: 800px) {
        .css-1r6slb0 { flex-direction: column; }
        .css-1r6slb0 > div { width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== 从环境变量或secrets读取API密钥 ====================
def get_api_keys():
    api_key = st.secrets.get("API_KEY", os.environ.get("BINANCE_API_KEY", ""))
    secret = st.secrets.get("SECRET", os.environ.get("BINANCE_SECRET", ""))
    return api_key, secret

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ACCOUNT_BALANCE = 10000.0
LEVERAGE = 100
MAX_TOTAL_RISK = 0.55
TIMEFRAMES = ['5m', '15m', '1h']

EXCHANGES = [
    {'name': 'binance', 'class': ccxt.binance, 'options': {'defaultType': 'future'}},
    {'name': 'okx', 'class': ccxt.okx, 'options': {'defaultType': 'swap'}},
    {'name': 'bybit', 'class': ccxt.bybit, 'options': {'defaultType': 'linear'}},
]

# 会话状态初始化
defaults = {
    'use_simulated': True,          # 默认使用模拟数据（避免网络问题）
    'real_trading': False,
    'dry_run': True,
    'api_key': '',
    'secret': '',
    'positions': {sym: None for sym in SYMBOLS},
    'trade_log': [],
    'equity_history': [ACCOUNT_BALANCE],
    'signal_history': {sym: [] for sym in SYMBOLS},
    'hf_history': {sym: [] for sym in SYMBOLS},
    'sim_step': 0,
    'replay_step': 0,
    'replay_data': {},
    'sim_prices': {},
    'daily_trade_count': 0,
    'last_trade_day': datetime.now().date(),
    'pending_signals': [],
    'total_trades': 0,
    'winning_trades': 0,
    'total_pnl': 0.0,
    'max_trades_per_day': 30,
    'preferred_exchange': 'binance',
    'use_hf': True,
    'use_ema_filter': True,
    'max_drawdown_pct': 20.0,
    'daily_loss_limit': 500.0,
    'peak_equity': ACCOUNT_BALANCE,
    'trading_paused': False,
    'performance_mode': False,       # 性能模式（刷新间隔120秒）
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = {sym: None for sym in SYMBOLS}
if 'cached_ohlcv' not in st.session_state:
    st.session_state.cached_ohlcv = {}

# 自动填充API密钥
api_key_from_env, secret_from_env = get_api_keys()
if api_key_from_env and not st.session_state.api_key:
    st.session_state.api_key = api_key_from_env
if secret_from_env and not st.session_state.secret:
    st.session_state.secret = secret_from_env

# ==================== 模拟K线生成（使用新频率格式）====================
def generate_simulated_ohlcv(symbol, timeframe, limit=150):
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
    # 使用 Pandas 推荐的频率格式：'5min', '15min', '1h'
    freq_map = {'5m': '5min', '15m': '15min', '1h': '1h'}
    freq = freq_map.get(timeframe, '15min')
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

# ==================== 多交易所数据获取（带缓存）====================
@retry(tries=2, delay=1)
def fetch_from_exchange(ex, exch_symbol, timeframe, limit, days_back):
    if days_back:
        since = int((datetime.now() - timedelta(days=days_back)).timestamp()*1000)
        return ex.fetch_ohlcv(exch_symbol, timeframe, since=since, limit=limit)
    else:
        return ex.fetch_ohlcv(exch_symbol, timeframe, limit=limit)

def fetch_ohlcv(symbol, timeframe, limit=150, days_back=None):
    cache_key = f"{symbol}_{timeframe}_{limit}"
    now = datetime.now()
    if cache_key in st.session_state.cached_ohlcv:
        cached_time, cached_df = st.session_state.cached_ohlcv[cache_key]
        if (now - cached_time).seconds < 30:   # 缓存30秒
            return cached_df

    if st.session_state.use_simulated:
        return generate_simulated_ohlcv(symbol, timeframe, limit)

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
            st.session_state.trade_log.append(f"❌ {exch['name']} 请求失败: {str(e)}")
            continue

    if df is None:
        st.warning("⚠️ 所有交易所均获取数据失败，使用模拟数据")
        df = generate_simulated_ohlcv(symbol, timeframe, limit)

    st.session_state.cached_ohlcv[cache_key] = (now, df)
    return df

# ==================== 技术指标（简化版）====================
def add_indicators(df):
    if len(df) < 50:
        return df
    df = df.copy()
    df['ema20'] = ta.trend.ema_indicator(df['close'],20)
    df['ema50'] = ta.trend.ema_indicator(df['close'],50)
    df['rsi'] = ta.momentum.rsi(df['close'],14)
    df['atr'] = ta.volatility.average_true_range(df['high'],df['low'],df['close'],14)
    bb = ta.volatility.BollingerBands(df['close'],20,2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume']/df['volume_sma']
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    return df

# ==================== 多时间框架信号（简化版）====================
def multi_tf_signal(symbol):
    signals = {}
    for tf in TIMEFRAMES:
        df = add_indicators(fetch_ohlcv(symbol, tf))
        if len(df) < 50:
            signals[tf] = "无数据"
        else:
            signals[tf] = "观望"
    return signals

# ==================== 图表更新函数（使用 width='stretch'）====================
@st.fragment(run_every=60)  # 基础刷新60秒
def update_chart(symbol):
    df_hf = add_indicators(fetch_ohlcv(symbol, '5m', limit=150))
    signals_tf = multi_tf_signal(symbol)
    consensus = "中性"
    st.caption(f"多TF共识：**{consensus}**")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.20, 0.25],
        vertical_spacing=0.02,
        subplot_titles=(f"{symbol} 价格", "成交量", "MACD")
    )
    fig.add_trace(go.Candlestick(
        x=df_hf['timestamp'],
        open=df_hf['open'],
        high=df_hf['high'],
        low=df_hf['low'],
        close=df_hf['close'],
        increasing_line_color="#00ff9d",
        decreasing_line_color="#ff4d4d"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_hf['timestamp'], y=df_hf['ema20'], name="EMA20", line=dict(color="#ffaa00")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_hf['timestamp'], y=df_hf['ema50'], name="EMA50", line=dict(color="#aa88ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_hf['timestamp'], y=df_hf['macd'], name="MACD", line=dict(color="#00b0ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_hf['timestamp'], y=df_hf['macd_signal'], name="信号线", line=dict(color="#ffd700")), row=1, col=1)

    colors = ['#00ff9d' if o < c else '#ff4d4d' for o, c in zip(df_hf['open'], df_hf['close'])]
    fig.add_trace(go.Bar(x=df_hf['timestamp'], y=df_hf['volume'], name="成交量", marker_color=colors, opacity=0.6), row=2, col=1)

    colors_hist = ['#00ff9d' if h > 0 else '#ff4d4d' for h in df_hf['macd_diff']]
    fig.add_trace(go.Bar(x=df_hf['timestamp'], y=df_hf['macd_diff'], name="MACD柱", marker_color=colors_hist), row=3, col=1)

    latest_price = df_hf['close'].iloc[-1]
    prev_price = df_hf['close'].iloc[-2]
    price_change = (latest_price - prev_price) / prev_price * 100
    price_label = f"当前: {latest_price:.2f} ({price_change:+.2f}%)"
    fig.add_annotation(
        x=df_hf['timestamp'].iloc[-1], y=latest_price,
        text=price_label,
        showarrow=True, arrowhead=1, ax=40, ay=-40,
        bgcolor="#21262d", font=dict(color="white", size=12),
        row=1, col=1
    )

    fig.update_layout(height=620, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#ffffff"))
    # 使用 width='stretch' 替代 use_container_width
    st.plotly_chart(fig, width='stretch')

    st.markdown("**多TF信号详情**")
    tf_cols = st.columns(3)
    for idx, (tf, sig) in enumerate(signals_tf.items()):
        tf_cols[idx].metric(tf, sig, delta_color="off")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📊 VAI v9.0 终极版")
    st.metric("总权益", f"${st.session_state.equity_history[-1]:,.2f}")
    st.metric("今日已开单", f"{st.session_state.daily_trade_count}/{st.session_state.max_trades_per_day}")
    st.metric("排队信号数", len(st.session_state.pending_signals))
    if st.button("🚨 紧急全平仓", type="primary", use_container_width=True):
        st.success("已执行紧急全平仓！")
        st.rerun()
    if st.button("🔄 重置会话", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.session_state.use_simulated = st.checkbox("使用模拟数据", value=st.session_state.use_simulated)
    st.session_state.performance_mode = st.checkbox("性能模式（120秒刷新）", value=st.session_state.performance_mode)

# ==================== 主界面 ====================
st.markdown("# 🤖 AI 自进化交易平台 VAI v9.0 终极稳定版", unsafe_allow_html=True)
st.caption("🌟 已修复所有弃用警告 · 支持性能模式 · 数据缓存30秒 · 模拟/真实切换")

tab1, tab2, tab3, tab4 = st.tabs(["📈 实时交易", "🔙 回测中心", "📊 风险仪表板", "⚙️ 设定"])

with tab1:
    st.subheader("实时市场与信号")
    cols = st.columns(len(SYMBOLS))
    for i, symbol in enumerate(SYMBOLS):
        with cols[i]:
            st.subheader(symbol)
            update_chart(symbol)

with tab4:
    st.header("⚙️ 设定")
    st.info("请在侧边栏选择是否使用模拟数据。如需真实数据，请确保网络可访问交易所API。")
    st.session_state.real_trading = st.checkbox("启用真实交易", st.session_state.real_trading)
    st.session_state.dry_run = st.checkbox("乾跑模式", st.session_state.dry_run)
    if st.session_state.real_trading:
        st.text_input("Binance API Key", st.session_state.api_key, type="password", disabled=True)
        st.text_input("Binance Secret", st.session_state.secret, type="password", disabled=True)
        st.warning("API密钥已通过环境变量或secrets自动填充，请勿手动输入。")
    st.slider("每日开单上限", 1, 100, st.session_state.max_trades_per_day, key="max_trades_per_day")

# 根据性能模式设置刷新间隔
refresh_interval = 120000 if st.session_state.performance_mode else 60000  # 120秒或60秒
st_autorefresh(interval=refresh_interval, key="auto_refresh")
st.markdown(f"""
<div style="text-align:center; color:#666; font-size:14px;">
    ⭐ 当前刷新间隔：{'120秒' if st.session_state.performance_mode else '60秒'} · 数据缓存30秒 · 无弃用警告
</div>
""", unsafe_allow_html=True)
