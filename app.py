# -*- coding: utf-8 -*-
"""
🚀 币安15分钟合约盯盘工具（实时版 · 高胜率信号）
===================================================
[功能说明]
- 实时获取BTC/USDT和ETH/USDT 15m K线（优先Bybit，备选Binance/OKX）
- 计算核心指标：EMA12/26、RSI14、布林带、ADX14、ATR
- 双模式信号：趋势模式（高胜率） + 震荡模式（补每天开单）
- 明确显示当前趋势方向 + 交易计划（入场/止损/止盈）
- 自动刷新（每30秒）
- 胜率导向：趋势信号优先，震荡补位，目标每天1-3单
- 可直接运行：streamlit run stare.py
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
st.set_page_config(page_title="币安15m盯盘工具", layout="wide")
st.title("🚀 币安15分钟合约实时盯盘（高胜率版）")
st.caption("实时数据 · 双模式信号 · 每天开单 · 当前时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ==================== 配置 ====================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = '15m'
LIMIT = 200  # 获取最近200根K线
REFRESH_INTERVAL = 30  # 秒

# ==================== 多交易所获取数据 ====================
@st.cache_data(ttl=20)  # 缓存20秒，避免频繁请求
def fetch_ohlcv(symbol: str):
    """
    尝试多个交易所获取K线数据（合约），顺序：Bybit -> Binance -> OKX
    Bybit 在中国大陆通常可用，优先使用。
    """
    exchanges = [
        ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}}),   # USDT永续合约
        ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}}), # 币安合约（可能受限）
        ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})         # OKX永续合约
    ]

    for ex in exchanges:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # 可在侧边栏显示数据源（调试用，正式可注释）
            # st.sidebar.success(f"数据源: {ex.name} - {symbol}")
            return df
        except Exception as e:
            # st.sidebar.warning(f"{ex.name} 获取 {symbol} 失败: {str(e)[:50]}")
            continue

    # 所有交易所都失败
    st.error(f"无法获取 {symbol} 数据，请检查网络连接或尝试使用VPN。")
    return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
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
    return df

# ==================== 信号生成 ====================
def generate_signal(df: pd.DataFrame, symbol: str):
    """根据最新数据生成交易信号和计划"""
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

    signal = "观望"
    plan = None

    # 趋势模式（高胜率优先）
    if adx > 25:  # 强趋势
        if ema12 > ema26 and prev['ema12'] <= prev['ema26']:  # 金叉
            signal = f"强势多头信号 ({symbol})"
            plan = f"入场：{price:.2f} 多\n止损：{price - atr*1.5:.2f}\n止盈：{price + atr*3:.2f} (2倍风险)"
        elif ema12 < ema26 and prev['ema12'] >= prev['ema26']:  # 死叉
            signal = f"强势空头信号 ({symbol})"
            plan = f"入场：{price:.2f} 空\n止损：{price + atr*1.5:.2f}\n止盈：{price - atr*3:.2f} (2倍风险)"

    # 震荡模式（补每天开单）
    elif adx <= 25:
        if price <= bb_lower and rsi < 35:
            signal = f"震荡多头信号 ({symbol})"
            plan = f"入场：{price:.2f} 多（下轨反弹）\n止损：{price - atr*1.2:.2f}\n止盈：{bb_middle:.2f} (中轨)"
        elif price >= bb_upper and rsi > 65:
            signal = f"震荡空头信号 ({symbol})"
            plan = f"入场：{price:.2f} 空（上轨回落）\n止损：{price + atr*1.2:.2f}\n止盈：{bb_middle:.2f} (中轨)"

    return signal, plan

# ==================== 主界面 ====================
cols = st.columns(len(SYMBOLS))
signals_today = []

for i, symbol in enumerate(SYMBOLS):
    with cols[i]:
        st.subheader(symbol)

        df = fetch_ohlcv(symbol)
        if df is None:
            st.error("数据获取失败，请检查网络或稍后重试")
            continue

        df = add_indicators(df)

        # K线图
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.2, 0.2],
                            vertical_spacing=0.05)

        fig.add_trace(go.Candlestick(x=df['timestamp'],
                                     open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'],
                                     name="K线"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema12'], line=dict(color='yellow'), name="EMA12"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema26'], line=dict(color='purple'), name="EMA26"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], line=dict(color='gray', dash='dash'), name="布林上轨"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], line=dict(color='gray', dash='dash'), name="布林下轨"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], line=dict(color='orange'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], line=dict(color='blue'), name="ADX"), row=3, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(height=600, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # 当前信号
        signal, plan = generate_signal(df, symbol)
        st.metric("当前信号", signal)

        if plan:
            st.success("交易计划")
            st.code(plan)
            signals_today.append(signal)

        st.caption(f"最新价格: {df['close'].iloc[-1]:.2f} | RSI: {df['rsi'].iloc[-1]:.1f} | ADX: {df['adx'].iloc[-1]:.1f}")

# ==================== 总结 ====================
st.markdown("### 当日信号总结")
if any("信号" in s for s in signals_today):
    st.success("今日有明确信号！优先执行趋势信号")
    for s in signals_today:
        if "信号" in s:
            st.write("• " + s)
else:
    st.info("今日暂无强信号，继续等待高概率机会（震荡市耐心为上）")

# 自动刷新提示
st.info(f"自动刷新中... 下次更新: {REFRESH_INTERVAL}秒后")

# 等待指定时间后刷新页面
time.sleep(REFRESH_INTERVAL)
st.rerun()
