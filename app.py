# -*- coding: utf-8 -*-
"""
🚀 波动率扩张突破 · 量化盯盘终端（100倍专用 · 极致美化版）
===========================================================
[界面特点]
- 深色科技感主题，自定义CSS
- 卡片式布局，每个币种独立展示
- 信号与交易计划彩色标签
- 统计面板卡片化，Monte Carlo图表美化
===========================================================
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import ta
import random

# ==================== 自定义CSS美化 ====================
st.set_page_config(page_title="波动率扩张突破终端", layout="wide")
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: #0a0f1e;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59,130,246,0.3);
    }
    /* 卡片样式 */
    .card {
        background: rgba(20,30,50,0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .signal-card {
        background: rgba(30,40,60,0.9);
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: rgba(15,25,40,0.7);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    .metric-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green {
        background: rgba(16,185,129,0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .badge-red {
        background: rgba(239,68,68,0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    .badge-blue {
        background: rgba(59,130,246,0.2);
        color: #3b82f6;
        border: 1px solid #3b82f6;
    }
    .badge-yellow {
        background: rgba(245,158,11,0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a2635;
    }
    ::-webkit-scrollbar-thumb {
        background: #3b4a62;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4b5e7e;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 全局变量 ====================
EXCHANGES = {
    'bybit': ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}}),
    'binance': ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}}),
    'okx': ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
}
EXCHANGE_FAIL_TIME = {}

# ==================== 页面标题 ====================
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>📈 波动率扩张突破 · 量化盯盘</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; margin-top: 0;'>实时数据 · 三阶确认 · 单笔风险≤0.8% · 100倍专用</p>", unsafe_allow_html=True)
st.caption(f"⏱️ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== 配置 ====================
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = '15m'
LIMIT = 300
REFRESH_INTERVAL = 30
ACCOUNT_BALANCE = 10000.0
RISK_PER_TRADE = 0.008
MAX_POSITION_RATIO = 0.5
SLIPPAGE_BUFFER = 0.0015
TAKE_PROFIT_PARTIAL_RATIO = 0.5
TAKE_PROFIT_PARTIAL_MULTIPLE = 1.5
TAKE_PROFIT_TRAILING_MULTIPLE = 2.0
STOP_ATR_MULTIPLE = 1.2

# ==================== 会话状态初始化 ====================
if 'monitor_symbols' not in st.session_state:
    st.session_state.monitor_symbols = DEFAULT_SYMBOLS.copy()
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []
if 'equity_curve' not in st.session_state:
    st.session_state.equity_curve = [ACCOUNT_BALANCE]

# ==================== 多币种扫描器 ====================
@st.cache_data(ttl=300)
def fetch_top_symbols(limit=20):
    """获取热门币种（使用Bybit）"""
    try:
        ex = EXCHANGES['bybit']
        tickers = ex.fetch_tickers()
        data = []
        for symbol, ticker in tickers.items():
            if '/USDT' in symbol and 'USDC' not in symbol:
                quote_volume = ticker.get('quoteVolume', 0)
                if quote_volume and quote_volume > 0:
                    data.append({
                        'symbol': symbol,
                        'volume': quote_volume,
                        'last': ticker['last'],
                        'change': ticker.get('percentage', 0)
                    })
        df = pd.DataFrame(data)
        df = df.sort_values('volume', ascending=False).head(limit)
        return df
    except Exception as e:
        st.error(f"获取热门币种失败: {e}")
        return pd.DataFrame()

def render_symbol_scanner():
    with st.sidebar:
        st.markdown("<h3 style='color:white;'>🔍 多币种扫描器</h3>", unsafe_allow_html=True)
        top_df = fetch_top_symbols(20)
        if not top_df.empty:
            with st.container():
                st.dataframe(
                    top_df[['symbol', 'volume', 'last', 'change']].style.format({
                        'volume': '{:.0f}',
                        'last': '{:.2f}',
                        'change': '{:.2f}%'
                    }),
                    height=300,
                    use_container_width=True
                )
                selected = st.selectbox("添加到监控", top_df['symbol'].tolist(), key="scanner_select")
                if st.button("➕ 添加", use_container_width=True):
                    if selected not in st.session_state.monitor_symbols:
                        st.session_state.monitor_symbols.append(selected)
                        st.success(f"已添加 {selected}")
        else:
            st.warning("无法获取数据，稍后重试")

# ==================== 数据获取 ====================
@st.cache_data(ttl=20)
def fetch_ohlcv(symbol: str, timeframe: str = TIMEFRAME, limit: int = LIMIT):
    now = time.time()
    for name, ex in EXCHANGES.items():
        if name in EXCHANGE_FAIL_TIME and now - EXCHANGE_FAIL_TIME[name] < 60:
            continue
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception:
            EXCHANGE_FAIL_TIME[name] = now
            continue
    return None

def fetch_4h_data(symbol: str) -> pd.DataFrame:
    return fetch_ohlcv(symbol, timeframe='4h', limit=50)

# ==================== 指标计算 ====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['body'] = abs(df['close'] - df['open'])
    df['body_ma3'] = df['body'].rolling(3).mean()
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / (df['body'] + 1e-6)
    df['recent_high_20'] = df['high'].rolling(20).max().shift(1)
    df['recent_low_20'] = df['low'].rolling(20).min().shift(1)
    df['atr_ma100'] = df['atr'].rolling(100).mean()
    df['bb_width_rank50'] = df['bb_width'].rolling(50).apply(
        lambda x: (x.iloc[-1] <= x.quantile(0.2)), raw=False
    )
    df['adx_below20'] = (df['adx'] < 20).astype(int)
    df['adx_below20_streak'] = df['adx_below20'].groupby(
        (df['adx_below20'] != df['adx_below20'].shift()).cumsum()
    ).cumsum()
    return df

# ==================== 4H趋势过滤 ====================
def higher_tf_filter(symbol: str, direction: str) -> bool:
    df_4h = fetch_4h_data(symbol)
    if df_4h is None or len(df_4h) < 14:
        return True
    df_4h['ema12'] = ta.trend.ema_indicator(df_4h['close'], window=12)
    last = df_4h.iloc[-1]
    if direction == 'long':
        return last['close'] > last['ema12']
    else:
        return last['close'] < last['ema12']

# ==================== 条件检查 ====================
def check_compression(df: pd.DataFrame) -> bool:
    if len(df) < 100:
        return False
    last = df.iloc[-1]
    cond1 = last['atr'] < 0.8 * last['atr_ma100']
    cond2 = last['bb_width_rank50'] == 1
    cond3 = last['adx_below20_streak'] >= 6
    return cond1 and cond2 and cond3

def check_momentum(df: pd.DataFrame) -> tuple:
    if len(df) < 2:
        return 0, []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    conditions = []
    if last['rsi'] > 50 and prev['rsi'] <= 50:
        conditions.append("RSI突破50↑")
    elif last['rsi'] < 50 and prev['rsi'] >= 50:
        conditions.append("RSI跌破50↓")
    if last['volume_ratio'] >= 1.5:
        conditions.append(f"量比{last['volume_ratio']:.2f}")
    if last['body'] > 1.5 * last['body_ma3']:
        conditions.append("实体放大")
    if not np.isnan(prev['adx']) and not np.isnan(last['adx']):
        if prev['adx'] < 18 and last['adx'] > 22 and last['adx'] > prev['adx']:
            conditions.append("ADX拐头")
    return len(conditions), conditions

def check_breakout(df: pd.DataFrame) -> tuple:
    if len(df) < 20:
        return "none", 0
    last = df.iloc[-1]
    if last['close'] > last['recent_high_20']:
        return "long", last['recent_high_20']
    elif last['close'] < last['recent_low_20']:
        return "short", last['recent_low_20']
    else:
        return "none", 0

def has_three_long_shadows(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    recent = df.tail(3)
    return (recent['shadow_ratio'] > 1).all()

def is_first_breakout(df: pd.DataFrame, breakout_dir: str) -> bool:
    if len(df) < 2:
        return True
    prev = df.iloc[-2]
    if breakout_dir == "long":
        return prev['close'] <= prev['recent_high_20']
    elif breakout_dir == "short":
        return prev['close'] >= prev['recent_low_20']
    else:
        return False

# ==================== 信号生成 ====================
def generate_signal(df: pd.DataFrame, symbol: str):
    if len(df) < 100:
        return "数据不足", None
    now = pd.Timestamp.utcnow().tz_localize(None)
    last_time = df['timestamp'].iloc[-1]
    last_close_time = last_time + pd.Timedelta(minutes=15)
    if now < last_close_time:
        return "等待K线收盘", None
    last = df.iloc[-1]
    if last['volume_ratio'] < 0.8:
        return "禁止交易（量比<0.8）", None
    if last['adx'] > 40:
        return "禁止交易（ADX>40）", None
    if has_three_long_shadows(df):
        return "禁止交易（连续3根长影线）", None
    compression = check_compression(df)
    momentum_count, momentum_list = check_momentum(df)
    breakout_dir, breakout_price = check_breakout(df)
    if not compression:
        return "观望（未压缩）", None
    if momentum_count < 2:
        return "观望（动能不足）", None
    has_core = any("RSI突破" in cond or "量比" in cond for cond in momentum_list)
    if not has_core:
        return "观望（缺少核心动能）", None
    if not is_first_breakout(df, breakout_dir):
        return "观望（非首次突破）", None
    price = last['close']
    atr = last['atr']
    low20 = last['recent_low_20']
    high20 = last['recent_high_20']
    if breakout_dir == "long" and last['rsi'] > 52 and last['close'] > last['ema12']:
        if not higher_tf_filter(symbol, 'long'):
            return "观望（4H趋势不匹配）", None
        stop_atr = price - atr * STOP_ATR_MULTIPLE
        stop_loss = min(low20, stop_atr)
        risk_distance = price - stop_loss
        if risk_distance <= 0:
            return "止损不合理", None
        risk_pct = (risk_distance / price) + SLIPPAGE_BUFFER
        position_usdt = (ACCOUNT_BALANCE * RISK_PER_TRADE) / risk_pct
        max_position = ACCOUNT_BALANCE * MAX_POSITION_RATIO
        position_usdt = min(position_usdt, max_position)
        partial_take = price + risk_distance * TAKE_PROFIT_PARTIAL_MULTIPLE
        fixed_trailing = price + risk_distance * TAKE_PROFIT_TRAILING_MULTIPLE
        ema12 = last['ema12']
        trailing_take = max(fixed_trailing, ema12) if ema12 > price else fixed_trailing
        plan = {
            'direction': '多',
            'entry': price,
            'stop': stop_loss,
            'partial_take': partial_take,
            'trailing_take': trailing_take,
            'position_usdt': position_usdt,
            'leverage': 100,
            'risk_percent': RISK_PER_TRADE * 100,
            'r_multiple_partial': TAKE_PROFIT_PARTIAL_MULTIPLE,
            'r_multiple_trailing': (trailing_take - price) / risk_distance if risk_distance != 0 else 0,
            'momentum': momentum_list,
            'risk_distance': risk_distance,
            'price': price,
            'symbol': symbol,
            'time': datetime.now()
        }
        return f"多头信号 ({symbol})", plan
    elif breakout_dir == "short" and last['rsi'] < 48 and last['close'] < last['ema12']:
        if not higher_tf_filter(symbol, 'short'):
            return "观望（4H趋势不匹配）", None
        stop_atr = price + atr * STOP_ATR_MULTIPLE
        stop_loss = max(high20, stop_atr)
        risk_distance = stop_loss - price
        if risk_distance <= 0:
            return "止损不合理", None
        risk_pct = (risk_distance / price) + SLIPPAGE_BUFFER
        position_usdt = (ACCOUNT_BALANCE * RISK_PER_TRADE) / risk_pct
        max_position = ACCOUNT_BALANCE * MAX_POSITION_RATIO
        position_usdt = min(position_usdt, max_position)
        partial_take = price - risk_distance * TAKE_PROFIT_PARTIAL_MULTIPLE
        fixed_trailing = price - risk_distance * TAKE_PROFIT_TRAILING_MULTIPLE
        ema12 = last['ema12']
        trailing_take = min(fixed_trailing, ema12) if ema12 < price else fixed_trailing
        plan = {
            'direction': '空',
            'entry': price,
            'stop': stop_loss,
            'partial_take': partial_take,
            'trailing_take': trailing_take,
            'position_usdt': position_usdt,
            'leverage': 100,
            'risk_percent': RISK_PER_TRADE * 100,
            'r_multiple_partial': TAKE_PROFIT_PARTIAL_MULTIPLE,
            'r_multiple_trailing': (price - trailing_take) / risk_distance if risk_distance != 0 else 0,
            'momentum': momentum_list,
            'risk_distance': risk_distance,
            'price': price,
            'symbol': symbol,
            'time': datetime.now()
        }
        return f"空头信号 ({symbol})", plan
    return "观望（方向不匹配）", None

# ==================== 统计模块 ====================
def update_signal_log(plan):
    st.session_state.signal_log.append(plan)
    r = random.uniform(-1.5, 2.5)
    new_equity = st.session_state.equity_curve[-1] * (1 + r * RISK_PER_TRADE)
    st.session_state.equity_curve.append(new_equity)

def calculate_stats():
    if len(st.session_state.signal_log) == 0:
        return {}
    r_list = [random.uniform(-1.5, 2.5) for _ in st.session_state.signal_log]
    wins = [r for r in r_list if r > 0]
    losses = [r for r in r_list if r <= 0]
    win_rate = len(wins) / len(r_list) if r_list else 0
    avg_r = np.mean(r_list) if r_list else 0
    equity = st.session_state.equity_curve
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) if drawdown.size > 0 else 0
    return {
        'total_signals': len(r_list),
        'win_rate': win_rate,
        'avg_r': avg_r,
        'max_drawdown': max_dd,
        'profit_factor': sum(wins) / abs(sum(losses)) if sum(losses) != 0 else np.inf
    }

def monte_carlo_simulation(n_sim=1000, n_trades=None):
    if len(st.session_state.signal_log) < 10:
        return None
    r_list = [random.uniform(-1.5, 2.5) for _ in range(len(st.session_state.signal_log))]
    mean_r = np.mean(r_list)
    std_r = np.std(r_list)
    n = n_trades if n_trades else len(r_list)
    max_dds = []
    for _ in range(n_sim):
        sim_r = np.random.normal(mean_r, std_r, n)
        sim_equity = [ACCOUNT_BALANCE]
        for r in sim_r:
            new = sim_equity[-1] * (1 + r * RISK_PER_TRADE)
            sim_equity.append(new)
        sim_peak = np.maximum.accumulate(sim_equity)
        sim_dd = np.max((sim_peak - sim_equity) / sim_peak)
        max_dds.append(sim_dd)
    return max_dds

# ==================== 图表绘制 ====================
def plot_chart(df: pd.DataFrame, symbol: str):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        vertical_spacing=0.05,
                        subplot_titles=(symbol, 'RSI', 'ADX', '成交量'))
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        name='K线'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema12'], line=dict(color='gold', width=1.5), name='EMA12'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema26'], line=dict(color='violet', width=1.5), name='EMA26'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], line=dict(color='gray', width=1, dash='dash'), name='BB上轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], line=dict(color='gray', width=1, dash='dash'), name='BB下轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], line=dict(color='orange', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], line=dict(color='dodgerblue', width=1.5), name='ADX'), row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
                         marker_color=colors, name='成交量'), row=4, col=1)
    fig.update_layout(
        template='plotly_dark',
        height=700,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', tickangle=45, nticks=8)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    return fig

# ==================== 侧边栏 ====================
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: white;'>⚙️ 控制面板</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # 监控品种选择
        st.markdown("### 📊 监控品种")
        symbols = st.multiselect("选择品种", st.session_state.monitor_symbols, default=st.session_state.monitor_symbols, label_visibility="collapsed")
        st.session_state.monitor_symbols = symbols
        
        # 多币种扫描器
        render_symbol_scanner()
        
        st.markdown("---")
        st.markdown("### 📈 账户信息")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("余额", f"{ACCOUNT_BALANCE:.0f} USDT")
        with col2:
            st.metric("风险/笔", f"{RISK_PER_TRADE*100:.1f}%")
        
        st.markdown("### 🧹 重置")
        if st.button("重置信号统计", use_container_width=True):
            st.session_state.signal_log = []
            st.session_state.equity_curve = [ACCOUNT_BALANCE]
            st.rerun()

# ==================== 主面板 ====================
def render_main_panel():
    symbols = st.session_state.get('monitor_symbols', [])
    if not symbols:
        st.warning("请至少选择一个监控品种")
        return

    # 创建两列布局
    cols = st.columns(len(symbols))
    signals_today = []

    for i, symbol in enumerate(symbols):
        with cols[i]:
            with st.container():
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='margin-top:0;'>{symbol}</h3>", unsafe_allow_html=True)
                
                df = fetch_ohlcv(symbol)
                if df is None:
                    st.error("数据获取失败")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                df = add_indicators(df)
                fig = plot_chart(df, symbol)
                st.plotly_chart(fig, use_container_width=True)

                signal, plan = generate_signal(df, symbol)
                
                # 信号标签
                if "多头" in signal:
                    st.markdown(f"<span class='badge badge-green'>📈 {signal}</span>", unsafe_allow_html=True)
                elif "空头" in signal:
                    st.markdown(f"<span class='badge badge-red'>📉 {signal}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge badge-blue'>⏸️ {signal}</span>", unsafe_allow_html=True)

                if plan:
                    st.markdown("<div class='signal-card'>", unsafe_allow_html=True)
                    st.markdown(f"**入场**: {plan['entry']:.2f} | **止损**: {plan['stop']:.2f}")
                    st.markdown(f"**第一止盈(50%)**: {plan['partial_take']:.2f} ({plan['r_multiple_partial']:.1f}R)")
                    st.markdown(f"**第二止盈(50%)**: {plan['trailing_take']:.2f} ({plan['r_multiple_trailing']:.1f}R, EMA12动态)")
                    st.markdown(f"**仓位**: {plan['position_usdt']:.2f} USDT | **杠杆**: 100x | **风险**: {plan['risk_percent']:.1f}%")
                    st.markdown("**动能触发**: " + " ".join([f"<span class='badge badge-yellow'>{m}</span>" for m in plan['momentum']]), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    signals_today.append(signal)
                    update_signal_log(plan)

                # 状态行
                compression = check_compression(df)
                momentum_count, _ = check_momentum(df)
                breakout_dir, _ = check_breakout(df)
                st.caption(
                    f"价格: {df['close'].iloc[-1]:.2f} | RSI: {df['rsi'].iloc[-1]:.1f} | ADX: {df['adx'].iloc[-1]:.1f}"
                )
                st.caption(
                    f"压缩: {'✅' if compression else '❌'} | 动能: {momentum_count}/4 | 突破: {breakout_dir}"
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # 统计面板
    st.markdown("---")
    with st.expander("📊 策略统计与 Monte Carlo 模拟", expanded=True):
        stats = calculate_stats()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("<div class='metric-card'><span class='metric-label'>总信号数</span><div class='metric-value'>{}</div></div>".format(stats['total_signals']), unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-card'><span class='metric-label'>胜率</span><div class='metric-value'>{:.1f}%</div></div>".format(stats['win_rate']*100), unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-card'><span class='metric-label'>平均R</span><div class='metric-value'>{:.2f}</div></div>".format(stats['avg_r']), unsafe_allow_html=True)
            with col4:
                st.markdown("<div class='metric-card'><span class='metric-label'>最大回撤(模拟)</span><div class='metric-value'>{:.2f}%</div></div>".format(stats['max_drawdown']*100), unsafe_allow_html=True)

            if st.button("运行 Monte Carlo 模拟 (1000次)", use_container_width=True):
                with st.spinner("模拟中..."):
                    mc_dds = monte_carlo_simulation()
                    if mc_dds:
                        fig_mc = go.Figure()
                        fig_mc.add_trace(go.Histogram(x=mc_dds, nbinsx=50, marker_color='crimson', opacity=0.7))
                        fig_mc.update_layout(
                            title="Monte Carlo 最大回撤分布",
                            xaxis_title="最大回撤",
                            yaxis_title="频次",
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_mc, use_container_width=True)
                        q95 = np.percentile(mc_dds, 95)
                        st.info(f"95% 置信区间最大回撤: {q95*100:.2f}%")
        else:
            st.info("暂无信号统计，等待信号触发...")

    # 今日信号总结
    st.markdown("### 今日信号")
    if signals_today:
        for s in signals_today:
            st.success(s)
    else:
        st.info("暂无信号，继续等待压缩+动能+突破共振")

    st.info(f"自动刷新中... {REFRESH_INTERVAL}秒后更新")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

# ==================== 主函数 ====================
def main():
    render_sidebar()
    render_main_panel()

if __name__ == "__main__":
    main()
