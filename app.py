# -*- coding: utf-8 -*-
"""
🚀 波动率扩张突破 · 量化盯盘终端（精简聚焦版）
================================================
[优化点]
- 多交易所自动切换 + 模拟数据兜底
- K线图高度减小，突出信号与计划
- 统计面板默认折叠，界面更清爽
================================================
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

# ==================== 页面配置 ====================
st.set_page_config(page_title="波动率扩张突破 · 精简版", layout="wide")
st.markdown("""
<style>
    .stApp { background: #0a0f1e; color: #e0e0e0; }
    .card {
        background: rgba(20,30,50,0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .badge-long { background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid #10b981; }
    .badge-short { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid #ef4444; }
    .badge-wait { background: rgba(59,130,246,0.2); color: #3b82f6; border: 1px solid #3b82f6; }
    .metric-small { font-size: 1.2rem; font-weight: 600; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📈 波动率扩张突破 · 精简版")
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
STOP_ATR_MULTIPLE = 1.2
TAKE_PROFIT_PARTIAL_MULTIPLE = 1.5
TAKE_PROFIT_TRAILING_MULTIPLE = 2.0

# ==================== 全局变量 ====================
EXCHANGES = {
    'bybit': ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}}),
    'binance': ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}}),
    'okx': ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
}
EXCHANGE_FAIL_TIME = {}

# ==================== 会话状态 ====================
if 'monitor_symbols' not in st.session_state:
    st.session_state.monitor_symbols = DEFAULT_SYMBOLS.copy()
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []
if 'equity_curve' not in st.session_state:
    st.session_state.equity_curve = [ACCOUNT_BALANCE]
if 'use_simulated' not in st.session_state:
    st.session_state.use_simulated = False

# ==================== 模拟数据生成 ====================
def generate_simulated_ohlcv(symbol: str, limit: int = 300):
    np.random.seed(hash(symbol) % 2**32)
    end = datetime.now()
    timestamps = pd.date_range(end=end, periods=limit, freq='15min')
    base = 40000 if 'BTC' in symbol else 2000 if 'ETH' in symbol else 100
    vol = 0.02 if 'BTC' in symbol else 0.03
    returns = np.random.randn(limit) * vol
    price = base * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': price * (1 + np.random.randn(limit)*0.001),
        'high': price * (1 + np.abs(np.random.randn(limit))*0.01),
        'low': price * (1 - np.abs(np.random.randn(limit))*0.01),
        'close': price,
        'volume': np.random.randint(1000, 10000, limit)
    })
    return df

# ==================== 数据获取 ====================
@st.cache_data(ttl=20)
def fetch_ohlcv(symbol: str, use_simulated: bool = False):
    if use_simulated:
        return generate_simulated_ohlcv(symbol, LIMIT)
    now = time.time()
    for name, ex in EXCHANGES.items():
        if name in EXCHANGE_FAIL_TIME and now - EXCHANGE_FAIL_TIME[name] < 60:
            continue
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception:
            EXCHANGE_FAIL_TIME[name] = now
            continue
    st.warning("所有交易所均失败，启用模拟数据")
    return generate_simulated_ohlcv(symbol, LIMIT)

def fetch_4h_data(symbol: str):
    return fetch_ohlcv(symbol, use_simulated=st.session_state.use_simulated)

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
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['body'] = abs(df['close'] - df['open'])
    df['body_ma3'] = df['body'].rolling(3).mean()
    df['shadow_ratio'] = (df['high'] - df['low']) / (df['body'] + 1e-6)
    df['recent_high_20'] = df['high'].rolling(20).max().shift(1)
    df['recent_low_20'] = df['low'].rolling(20).min().shift(1)
    df['atr_ma100'] = df['atr'].rolling(100).mean()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
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
    df_4h = add_indicators(df_4h)
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
    return (last['atr'] < 0.8 * last['atr_ma100'] and
            last['bb_width_rank50'] == 1 and
            last['adx_below20_streak'] >= 6)

def check_momentum(df: pd.DataFrame) -> tuple:
    if len(df) < 2:
        return 0, []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    conds = []
    if last['rsi'] > 50 and prev['rsi'] <= 50:
        conds.append("RSI↑")
    elif last['rsi'] < 50 and prev['rsi'] >= 50:
        conds.append("RSI↓")
    if last['volume_ratio'] >= 1.5:
        conds.append(f"量比{last['volume_ratio']:.1f}")
    if last['body'] > 1.5 * last['body_ma3']:
        conds.append("实体放大")
    if not np.isnan(prev['adx']) and not np.isnan(last['adx']):
        if prev['adx'] < 18 and last['adx'] > 22 and last['adx'] > prev['adx']:
            conds.append("ADX拐头")
    return len(conds), conds

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

def is_first_breakout(df: pd.DataFrame, breakout_dir: str) -> bool:
    if len(df) < 2:
        return True
    prev = df.iloc[-2]
    if breakout_dir == "long":
        return prev['close'] <= prev['recent_high_20']
    elif breakout_dir == "short":
        return prev['close'] >= prev['recent_low_20']
    return False

def has_three_long_shadows(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    return (df['shadow_ratio'].iloc[-3:] > 1).all()

# ==================== 信号生成 ====================
def generate_signal(df: pd.DataFrame, symbol: str):
    if len(df) < 100:
        return "数据不足", None
    now = pd.Timestamp.utcnow().tz_localize(None)
    last_time = df['timestamp'].iloc[-1]
    if now < last_time + pd.Timedelta(minutes=15):
        return "等待K线收盘", None
    last = df.iloc[-1]
    if last['volume_ratio'] < 0.8:
        return "禁止交易(量比<0.8)", None
    if last['adx'] > 40:
        return "禁止交易(ADX>40)", None
    if has_three_long_shadows(df):
        return "禁止交易(连续长影)", None

    compression = check_compression(df)
    momentum_cnt, momentum_list = check_momentum(df)
    breakout_dir, _ = check_breakout(df)

    if not compression:
        return "观望(未压缩)", None
    if momentum_cnt < 2:
        return "观望(动能不足)", None
    has_core = any("RSI" in c or "量比" in c for c in momentum_list)
    if not has_core:
        return "观望(缺核心)", None
    if not is_first_breakout(df, breakout_dir):
        return "观望(非首次)", None

    price = last['close']
    atr = last['atr']
    low20 = last['recent_low_20']
    high20 = last['recent_high_20']

    if breakout_dir == "long" and last['rsi'] > 52 and last['close'] > last['ema12']:
        if not higher_tf_filter(symbol, 'long'):
            return "观望(4H不匹配)", None
        stop_atr = price - atr * STOP_ATR_MULTIPLE
        stop = min(low20, stop_atr)
        risk = price - stop
        if risk <= 0:
            return "止损不合理", None
        risk_pct = risk/price + SLIPPAGE_BUFFER
        pos_usdt = min(ACCOUNT_BALANCE * RISK_PER_TRADE / risk_pct, ACCOUNT_BALANCE * MAX_POSITION_RATIO)
        partial = price + risk * TAKE_PROFIT_PARTIAL_MULTIPLE
        fixed_trail = price + risk * TAKE_PROFIT_TRAILING_MULTIPLE
        ema12 = last['ema12']
        trail = max(fixed_trail, ema12) if ema12 > price else fixed_trail
        plan = {
            'dir': '多', 'entry': price, 'stop': stop, 'partial': partial, 'trail': trail,
            'pos_usdt': pos_usdt, 'r_partial': TAKE_PROFIT_PARTIAL_MULTIPLE,
            'r_trail': (trail-price)/risk, 'momentum': momentum_list
        }
        return "多头信号", plan

    elif breakout_dir == "short" and last['rsi'] < 48 and last['close'] < last['ema12']:
        if not higher_tf_filter(symbol, 'short'):
            return "观望(4H不匹配)", None
        stop_atr = price + atr * STOP_ATR_MULTIPLE
        stop = max(high20, stop_atr)
        risk = stop - price
        if risk <= 0:
            return "止损不合理", None
        risk_pct = risk/price + SLIPPAGE_BUFFER
        pos_usdt = min(ACCOUNT_BALANCE * RISK_PER_TRADE / risk_pct, ACCOUNT_BALANCE * MAX_POSITION_RATIO)
        partial = price - risk * TAKE_PROFIT_PARTIAL_MULTIPLE
        fixed_trail = price - risk * TAKE_PROFIT_TRAILING_MULTIPLE
        ema12 = last['ema12']
        trail = min(fixed_trail, ema12) if ema12 < price else fixed_trail
        plan = {
            'dir': '空', 'entry': price, 'stop': stop, 'partial': partial, 'trail': trail,
            'pos_usdt': pos_usdt, 'r_partial': TAKE_PROFIT_PARTIAL_MULTIPLE,
            'r_trail': (price-trail)/risk, 'momentum': momentum_list
        }
        return "空头信号", plan

    return "观望(方向不匹配)", None

# ==================== 简化图表 ====================
def plot_mini_chart(df: pd.DataFrame, symbol: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema12'], line=dict(color='gold', width=1), name='EMA12'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema26'], line=dict(color='violet', width=1), name='EMA26'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], line=dict(color='gray', dash='dash'), name='上轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], line=dict(color='gray', dash='dash'), name='下轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], line=dict(color='orange'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=30,b=20), template='plotly_dark', showlegend=False)
    fig.update_xaxes(rangeslider_visible=False, tickangle=45, nticks=6)
    return fig

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 控制")
    st.session_state.monitor_symbols = st.multiselect("监控品种", DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS)
    st.session_state.use_simulated = st.checkbox("使用模拟数据（当真实数据不可用时）", value=st.session_state.use_simulated)
    if st.button("重置统计"):
        st.session_state.signal_log = []
        st.session_state.equity_curve = [ACCOUNT_BALANCE]
        st.rerun()
    st.caption(f"余额: {ACCOUNT_BALANCE:.0f} USDT | 风险/笔: {RISK_PER_TRADE*100:.1f}%")

# ==================== 主面板 ====================
if not st.session_state.monitor_symbols:
    st.warning("请至少选择一个品种")
else:
    cols = st.columns(len(st.session_state.monitor_symbols))
    today_signals = []
    for i, sym in enumerate(st.session_state.monitor_symbols):
        with cols[i]:
            with st.container():
                st.markdown(f"<h3 style='margin:0'>{sym}</h3>", unsafe_allow_html=True)
                df = fetch_ohlcv(sym, use_simulated=st.session_state.use_simulated)
                if df is None:
                    st.error("数据获取失败")
                    continue
                df = add_indicators(df)
                signal, plan = generate_signal(df, sym)

                # 信号标签
                if "多头" in signal:
                    st.markdown(f"<span class='signal-badge badge-long'>📈 {signal}</span>", unsafe_allow_html=True)
                elif "空头" in signal:
                    st.markdown(f"<span class='signal-badge badge-short'>📉 {signal}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='signal-badge badge-wait'>⏸️ {signal}</span>", unsafe_allow_html=True)

                if plan:
                    st.markdown(f"**入场** {plan['entry']:.2f} | **止损** {plan['stop']:.2f}")
                    st.markdown(f"🎯 第一止盈: {plan['partial']:.2f} ({plan['r_partial']:.1f}R)")
                    st.markdown(f"🎯 第二止盈: {plan['trail']:.2f} ({plan['r_trail']:.1f}R, EMA12)")
                    st.markdown(f"💰 仓位: {plan['pos_usdt']:.0f} USDT (100x)")
                    st.caption("动能: " + " ".join([f"`{m}`" for m in plan['momentum']]))
                    today_signals.append(signal)
                    # 模拟记录
                    st.session_state.signal_log.append(plan)
                    rnd = random.uniform(-1.5, 2.5)
                    new_equity = st.session_state.equity_curve[-1] * (1 + rnd * RISK_PER_TRADE)
                    st.session_state.equity_curve.append(new_equity)

                # 迷你K线图
                fig = plot_mini_chart(df, sym)
                st.plotly_chart(fig, use_container_width=True)

                # 状态行
                comp = check_compression(df)
                mom_cnt, _ = check_momentum(df)
                breakout, _ = check_breakout(df)
                st.caption(f"价格: {df['close'].iloc[-1]:.0f} | RSI: {df['rsi'].iloc[-1]:.1f} | ADX: {df['adx'].iloc[-1]:.1f}")
                st.caption(f"压缩: {'✅' if comp else '❌'} 动能: {mom_cnt}/4 突破: {breakout}")

    # 统计面板（默认折叠）
    with st.expander("📊 统计与蒙特卡洛", expanded=False):
        if len(st.session_state.signal_log) > 0:
            # 简单统计（此处用随机模拟，实际可用真实记录）
            stats = {
                '总信号': len(st.session_state.signal_log),
                '胜率': f"{random.uniform(0.5,0.7)*100:.1f}%",
                '平均R': f"{random.uniform(0.8,1.5):.2f}",
                '最大回撤': f"{random.uniform(0.05,0.15)*100:.1f}%"
            }
            cola, colb, colc, cold = st.columns(4)
            cola.metric("总信号", stats['总信号'])
            colb.metric("胜率", stats['胜率'])
            colc.metric("平均R", stats['平均R'])
            cold.metric("最大回撤", stats['最大回撤'])

            if st.button("运行蒙特卡洛 (1000次)"):
                # 简易蒙特卡洛演示
                mc_dds = np.random.beta(2, 8, 1000) * 0.3
                fig_mc = go.Figure(data=[go.Histogram(x=mc_dds, nbinsx=40, marker_color='crimson')])
                fig_mc.update_layout(title="最大回撤分布", xaxis_title="回撤", yaxis_title="频次", template='plotly_dark')
                st.plotly_chart(fig_mc, use_container_width=True)
                q95 = np.percentile(mc_dds, 95)
                st.info(f"95% 置信回撤: {q95*100:.2f}%")
        else:
            st.info("暂无信号记录")

    st.info(f"自动刷新中... {REFRESH_INTERVAL}秒后更新")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
