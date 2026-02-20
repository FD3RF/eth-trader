# -*- coding: utf-8 -*-
"""
🚀 币安15分钟合约盯盘工具（高胜率版 · 图表美化版）
===================================================
[功能说明]
- 实时获取BTC/USDT和ETH/USDT 15m K线（优先Bybit，备选Binance/OKX）
- 计算核心指标：EMA12/26、RSI14、布林带、ADX14、ATR、成交量比率
- 双模式信号：趋势模式（ADX>23） + 震荡模式（布林带+RSI+成交量）
- 明确显示当前趋势方向 + 交易计划（入场/止损/止盈）
- 图表美化：专业配色、成交量涨跌色、清晰网格
- 自动刷新（每30秒）
===================================================
"""

import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import ta

# 页面配置
st.set_page_config(page_title="币安15m盯盘工具（美化版）", layout="wide")
st.title("🚀 币安15分钟合约实时盯盘（高胜率版 · 图表美化）")
st.caption("实时数据 · 双模式信号 · 每天开单 · 当前时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ==================== 配置 ====================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = '15m'
LIMIT = 200  # 获取最近200根K线
REFRESH_INTERVAL = 30  # 秒

# ==================== 多交易所获取数据 ====================
@st.cache_data(ttl=20)
def fetch_ohlcv(symbol: str):
    exchanges = [
        ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}}),
        ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}}),
        ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    ]
    for ex in exchanges:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception:
            continue
    st.error(f"无法获取 {symbol} 数据，请检查网络连接或尝试使用VPN。")
    return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    return df

def generate_signal(df: pd.DataFrame, symbol: str):
    if len(df) < 50:
        return "数据不足", None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last['close']
    ema12 = last['ema12']
    ema26 = last['ema26']
    rsi = last['rsi']
    adx = last['adx']
    bb_upper = last['bb_upper']
    bb_lower = last['bb_lower']
    bb_middle = last['bb_middle']
    atr = last['atr']
    volume_ratio = last['volume_ratio']
    signal = "观望"
    plan = None
    if adx > 23:
        if ema12 > ema26 and prev['ema12'] <= prev['ema26']:
            signal = f"多头趋势信号 ({symbol})"
            plan = f"入场多：{price:.2f}\n止损：{price - atr*1.5:.2f}\n止盈：{price + atr*3:.2f}"
        elif ema12 < ema26 and prev['ema12'] >= prev['ema26']:
            signal = f"空头趋势信号 ({symbol})"
            plan = f"入场空：{price:.2f}\n止损：{price + atr*1.5:.2f}\n止盈：{price - atr*3:.2f}"
    else:
        if price <= bb_lower * 1.01 and rsi < 40 and volume_ratio > 1.2:
            signal = f"震荡多头信号 ({symbol})"
            plan = f"入场多：{price:.2f}（下轨反弹）\n止损：{price - atr*1.2:.2f}\n止盈：{bb_middle:.2f}"
        elif price >= bb_upper * 0.99 and rsi > 60 and volume_ratio > 1.2:
            signal = f"震荡空头信号 ({symbol})"
            plan = f"入场空：{price:.2f}（上轨回落）\n止损：{price + atr*1.2:.2f}\n止盈：{bb_middle:.2f}"
    return signal, plan

# ==================== 美化后的绘图函数 ====================
def plot_advanced_kline(df: pd.DataFrame, symbol: str):
    """生成美化后的K线图（包含成交量涨跌色）"""
    # 计算涨跌颜色（用于成交量）
    df['color'] = ['green' if close >= open else 'red' for close, open in zip(df['close'], df['open'])]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.05,
        subplot_titles=(symbol, 'RSI', 'ADX')
    )

    # ---- 主图：K线 + 指标 ----
    # 蜡烛图（颜色设置为涨绿跌红）
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线',
        increasing_line_color='#26a69a',   # 柔和绿色
        decreasing_line_color='#ef5350'    # 柔和红色
    ), row=1, col=1)

    # 指标线
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema12'], line=dict(color='#FFD700', width=1.5), name='EMA12'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema26'], line=dict(color='#BA55D3', width=1.5), name='EMA26'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], line=dict(color='#AAAAAA', width=1, dash='dash'), name='布林上轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], line=dict(color='#AAAAAA', width=1, dash='dash'), name='布林下轨'), row=1, col=1)

    # ---- 子图2：RSI ----
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], line=dict(color='#FFA500', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF4444", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#44FF44", opacity=0.5, row=2, col=1)

    # ---- 子图3：ADX + 成交量（叠加在ADX图上，用不同y轴） ----
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], line=dict(color='#1E90FF', width=1.5), name='ADX'), row=3, col=1)
    fig.add_hline(y=23, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    # 成交量柱状图（使用次y轴）
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['volume'],
        marker_color=df['color'],  # 根据涨跌着色
        name='成交量',
        opacity=0.5,
        yaxis='y4'  # 指定使用第4个y轴（自动创建）
    ), row=3, col=1)

    # 更新布局：美化
    fig.update_layout(
        template='plotly_dark',
        height=650,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=12),
        # 设置y轴格式
        yaxis=dict(title='价格', tickformat='.0f'),
        yaxis2=dict(title='RSI', range=[0, 100]),
        yaxis3=dict(title='ADX', range=[0, 60]),
        yaxis4=dict(title='成交量', overlaying='y3', side='right')  # 成交量轴与ADX共享x轴，显示在右侧
    )

    # 更新x轴格式
    fig.update_xaxes(
        rangeslider_visible=False,
        tickformat='%m-%d %H:%M',
        tickangle=45,
        nticks=10
    )

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

        # 显示美化后的图表
        fig = plot_advanced_kline(df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        # 信号和计划
        signal, plan = generate_signal(df, symbol)
        st.metric("当前信号", signal)

        if plan:
            st.success("📋 交易计划")
            st.code(plan)
            signals_today.append(signal)

        st.caption(f"最新价格: {df['close'].iloc[-1]:.2f} | RSI: {df['rsi'].iloc[-1]:.1f} | ADX: {df['adx'].iloc[-1]:.1f} | 量比: {df['volume_ratio'].iloc[-1]:.2f}")

# 当日信号总结
st.markdown("### 当日信号总结")
if any("信号" in s for s in signals_today):
    st.success("今日有明确信号！优先执行趋势信号")
    for s in signals_today:
        st.write("• " + s)
else:
    st.info("今日暂无强信号，继续等待高概率机会（震荡市耐心为上）")

st.info(f"自动刷新中... 下次更新: {REFRESH_INTERVAL}秒后")
time.sleep(REFRESH_INTERVAL)
st.rerun()
