# -*- coding: utf-8 -*-
"""
🚀 波动率扩张突破 · 量化盯盘终端（100倍专用 · 终极优化版）
===========================================================
[核心功能]
- 多交易所数据（Bybit/Binance/OKX）
- 三阶确认：压缩确认 + 动能启动 + 首次结构突破
- 4H 趋势过滤（价格站上EMA12）更灵敏
- 滑点缓冲（0.15%）校准真实风险
- 止损二选一（结构低点/1.2倍ATR）
- 动能条件 ≥2 且强制RSI/量比之一
- 第二止盈动态追踪EMA12
- 仓位上限保护（≤50%账户名义价值）
- 时间统一 pandas UTC，避免本地偏差
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

# ==================== 全局变量 ====================
EXCHANGES = {
    'bybit': ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}}),
    'binance': ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}}),
    'okx': ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
}
# 记录交易所失败时间（性能优化）
EXCHANGE_FAIL_TIME = {}

# ==================== 页面配置 ====================
st.set_page_config(page_title="波动率扩张突破终端", layout="wide")
st.title("📈 波动率扩张突破 · 量化盯盘（100倍专用）")
st.caption(f"实时数据 · 三阶确认 · 单笔风险≤0.8% · 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== 配置 ====================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = '15m'
LIMIT = 300
REFRESH_INTERVAL = 30  # 秒
ACCOUNT_BALANCE = 10000.0
RISK_PER_TRADE = 0.008  # 0.8%
MAX_POSITION_RATIO = 0.5  # 最大仓位占账户比例（名义价值）
SLIPPAGE_BUFFER = 0.0015  # 0.15% 滑点缓冲

# 分段止盈参数
TAKE_PROFIT_PARTIAL_RATIO = 0.5      # 第一部分仓位比例
TAKE_PROFIT_PARTIAL_MULTIPLE = 1.5    # 第一部分止盈倍数
TAKE_PROFIT_TRAILING_MULTIPLE = 2.0   # 第二部分止盈倍数（后备）

# 止损ATR倍数
STOP_ATR_MULTIPLE = 1.2

# ==================== 数据获取（性能优化版）====================
@st.cache_data(ttl=20)
def fetch_ohlcv(symbol: str, timeframe: str = TIMEFRAME, limit: int = LIMIT):
    """尝试多个交易所获取K线数据，失败后缓存60秒"""
    now = time.time()
    for name, ex in EXCHANGES.items():
        # 检查该交易所最近是否失败过
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
    st.error(f"无法获取 {symbol} 数据，请检查网络")
    return None

def fetch_4h_data(symbol: str) -> pd.DataFrame:
    """获取4小时K线数据"""
    return fetch_ohlcv(symbol, timeframe='4h', limit=50)

# ==================== 指标计算（15m）====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 基础指标
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # 布林带
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

    # 成交量
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # 实体大小
    df['body'] = abs(df['close'] - df['open'])
    df['body_ma3'] = df['body'].rolling(3).mean()

    # 影线长度
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / (df['body'] + 1e-6)

    # ✅ 结构高低点（前20根，去未来）
    df['recent_high_20'] = df['high'].rolling(20).max().shift(1)
    df['recent_low_20'] = df['low'].rolling(20).min().shift(1)

    # ATR 100均值
    df['atr_ma100'] = df['atr'].rolling(100).mean()

    # 布林带宽分位（最近50根）
    df['bb_width_rank50'] = df['bb_width'].rolling(50).apply(
        lambda x: (x.iloc[-1] <= x.quantile(0.2)), raw=False
    )

    # ADX <20 持续计数
    df['adx_below20'] = (df['adx'] < 20).astype(int)
    df['adx_below20_streak'] = df['adx_below20'].groupby(
        (df['adx_below20'] != df['adx_below20'].shift()).cumsum()
    ).cumsum()

    return df

# ==================== 4H趋势过滤（价格站上EMA12）====================
def higher_tf_filter(symbol: str, direction: str) -> bool:
    """4小时趋势过滤：多单要求收盘价 > EMA12，空单要求收盘价 < EMA12"""
    df_4h = fetch_4h_data(symbol)
    if df_4h is None or len(df_4h) < 14:
        return True  # 数据不足时不拦截
    df_4h['ema12'] = ta.trend.ema_indicator(df_4h['close'], window=12)
    last = df_4h.iloc[-1]
    if direction == 'long':
        return last['close'] > last['ema12']
    else:
        return last['close'] < last['ema12']

# ==================== 条件检查 ====================
def check_compression(df: pd.DataFrame) -> bool:
    """波动压缩确认（优化版）"""
    if len(df) < 100:
        return False
    last = df.iloc[-1]
    cond1 = last['atr'] < 0.8 * last['atr_ma100']
    cond2 = last['bb_width_rank50'] == 1
    cond3 = last['adx_below20_streak'] >= 6
    return cond1 and cond2 and cond3

def check_momentum(df: pd.DataFrame) -> tuple:
    """动能启动确认"""
    if len(df) < 2:
        return 0, []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    conditions = []

    # RSI突破50
    if last['rsi'] > 50 and prev['rsi'] <= 50:
        conditions.append("RSI突破50↑")
    elif last['rsi'] < 50 and prev['rsi'] >= 50:
        conditions.append("RSI跌破50↓")

    # 量比 ≥ 1.5
    if last['volume_ratio'] >= 1.5:
        conditions.append(f"量比{last['volume_ratio']:.2f}")

    # 实体放大
    if last['body'] > 1.5 * last['body_ma3']:
        conditions.append("实体放大")

    # ADX拐头
    if not np.isnan(prev['adx']) and not np.isnan(last['adx']):
        if prev['adx'] < 18 and last['adx'] > 22 and last['adx'] > prev['adx']:
            conditions.append("ADX拐头")

    return len(conditions), conditions

def check_breakout(df: pd.DataFrame) -> tuple:
    """结构突破确认（使用前20根高低点）"""
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
    """检查最近3根是否有连续长影线"""
    if len(df) < 3:
        return False
    recent = df.tail(3)
    return (recent['shadow_ratio'] > 1).all()

def is_first_breakout(df: pd.DataFrame, breakout_dir: str) -> bool:
    """确保是首次突破（前一根未突破）"""
    if len(df) < 2:
        return True
    prev = df.iloc[-2]
    if breakout_dir == "long":
        return prev['close'] <= prev['recent_high_20']
    elif breakout_dir == "short":
        return prev['close'] >= prev['recent_low_20']
    else:
        return False

# ==================== 信号生成（核心）====================
def generate_signal(df: pd.DataFrame, symbol: str):
    if len(df) < 100:
        return "数据不足", None

    # ✅ 统一使用 pandas UTC 时间
    now = pd.Timestamp.utcnow().tz_localize(None)
    last_time = df['timestamp'].iloc[-1]
    last_close_time = last_time + pd.Timedelta(minutes=15)

    if now < last_close_time:
        return "等待K线收盘", None

    # 禁止条件（ADX阈值放宽至40）
    last = df.iloc[-1]
    if last['volume_ratio'] < 0.8:
        return "禁止交易（量比<0.8）", None
    if last['adx'] > 40:  # 修改点
        return "禁止交易（ADX>40）", None
    if has_three_long_shadows(df):
        return "禁止交易（连续3根长影线）", None

    # 三阶确认
    compression = check_compression(df)
    momentum_count, momentum_list = check_momentum(df)
    breakout_dir, breakout_price = check_breakout(df)

    if not compression:
        return "观望（未压缩）", None

    # 动能条件：至少2条，且必须包含RSI突破或量比
    if momentum_count < 2:
        return "观望（动能不足）", None
    has_core = any("RSI突破" in cond or "量比" in cond for cond in momentum_list)
    if not has_core:
        return "观望（缺少核心动能）", None

    # 首次突破检查
    if not is_first_breakout(df, breakout_dir):
        return "观望（非首次突破）", None

    price = last['close']
    atr = last['atr']
    low20 = last['recent_low_20']
    high20 = last['recent_high_20']

    # 多单信号
    if breakout_dir == "long" and last['rsi'] > 52 and last['close'] > last['ema12']:
        # 4H趋势过滤
        if not higher_tf_filter(symbol, 'long'):
            return "观望（4H趋势不匹配）", None

        # 止损二选一：取结构低点和1.2倍ATR止损中较近者
        stop_atr = price - atr * STOP_ATR_MULTIPLE
        stop_loss = min(low20, stop_atr)
        risk_distance = price - stop_loss
        if risk_distance <= 0:
            return "止损不合理", None

        # 加入滑点缓冲
        risk_pct = (risk_distance / price) + SLIPPAGE_BUFFER
        position_usdt = (ACCOUNT_BALANCE * RISK_PER_TRADE) / risk_pct
        max_position = ACCOUNT_BALANCE * MAX_POSITION_RATIO
        position_usdt = min(position_usdt, max_position)

        # 分段止盈
        partial_take = price + risk_distance * TAKE_PROFIT_PARTIAL_MULTIPLE
        # 第二止盈：如果EMA12高于固定2R则取EMA12，否则取固定2R
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
            'momentum': momentum_list
        }
        return f"多头信号 ({symbol})", plan

    # 空单信号
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
            'momentum': momentum_list
        }
        return f"空头信号 ({symbol})", plan

    return "观望（方向不匹配）", None

# ==================== 图表绘制 ====================
def plot_chart(df: pd.DataFrame, symbol: str):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        vertical_spacing=0.04,
                        subplot_titles=(symbol, 'RSI', 'ADX', '成交量'))

    # 蜡烛图
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        name='K线'
    ), row=1, col=1)

    # 指标线
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema12'], line=dict(color='gold', width=1), name='EMA12'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema26'], line=dict(color='violet', width=1), name='EMA26'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], line=dict(color='gray', dash='dash'), name='BB上轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], line=dict(color='gray', dash='dash'), name='BB下轨'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], line=dict(color='orange'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # ADX
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], line=dict(color='dodgerblue'), name='ADX'), row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    # 成交量（着色）
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
                         marker_color=colors, name='成交量'), row=4, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=750,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.update_xaxes(rangeslider_visible=False, tickangle=45, nticks=10)
    return fig

# ==================== 主界面 ====================
cols = st.columns(len(SYMBOLS))
signals_today = []

for i, symbol in enumerate(SYMBOLS):
    with cols[i]:
        st.subheader(symbol)

        df = fetch_ohlcv(symbol)
        if df is None:
            st.error("数据获取失败")
            continue

        df = add_indicators(df)

        # 图表
        fig = plot_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        # 当前信号
        signal, plan = generate_signal(df, symbol)
        st.metric("当前信号", signal)

        if plan:
            st.success("📋 交易计划")
            st.code(
                f"方向: {plan['direction']}\n"
                f"入场: {plan['entry']:.2f}\n"
                f"止损: {plan['stop']:.2f}\n"
                f"第一止盈(50%): {plan['partial_take']:.2f} ({plan['r_multiple_partial']:.1f}R)\n"
                f"第二止盈(50%): {plan['trailing_take']:.2f} ({plan['r_multiple_trailing']:.1f}R, EMA12动态)\n"
                f"仓位(USDT): {plan['position_usdt']:.2f}\n"
                f"杠杆: {plan['leverage']}x\n"
                f"风险: {plan['risk_percent']:.2f}%\n"
                f"动能触发: {', '.join(plan['momentum'])}"
            )
            signals_today.append(signal)

        # 状态显示
        compression = check_compression(df)
        momentum_count, momentum_list = check_momentum(df)
        breakout_dir, _ = check_breakout(df)

        st.caption(
            f"价格: {df['close'].iloc[-1]:.2f} | RSI: {df['rsi'].iloc[-1]:.1f} | ADX: {df['adx'].iloc[-1]:.1f}\n"
            f"压缩: {'✅' if compression else '❌'} | 动能: {momentum_count}/4 | 突破: {breakout_dir}"
        )
        if momentum_list:
            st.caption("动能细节: " + " | ".join(momentum_list))

# 总结
st.markdown("### 今日信号")
if signals_today:
    for s in signals_today:
        st.success(s)
else:
    st.info("暂无信号，继续等待压缩+动能+突破共振")

st.info(f"自动刷新中... {REFRESH_INTERVAL}秒后更新")
time.sleep(REFRESH_INTERVAL)
st.rerun()
