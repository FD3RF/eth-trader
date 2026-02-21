# -*- coding: utf-8 -*-
"""
ETH 短线策略监控器 (1m/5m)
============================================
- 从 Binance 获取实时K线
- 指标：VWAP (20周期), EMA(9), EMA(21), ATR(14)
- 信号条件：价格突破VWAP + EMA金叉/死叉 + 成交量放大
- 自动计算止损(1.5*ATR)和止盈(2*ATR)
- 显示盈亏比和回撤预警
============================================
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import time

st.set_page_config(page_title="ETH 短线监控器", layout="wide")
st.title("📈 ETH 短线策略监控器 (1分钟/5分钟)")
st.caption("数据源：Binance · 仅监控不下单 · 自动刷新每5秒")

# ==================== 获取数据 ====================
@st.cache_data(ttl=5, show_spinner=False)
def fetch_ohlcv(symbol='ETH/USDT', timeframe='1m', limit=150):
    """从 Binance 获取K线数据"""
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # 永续合约数据
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

# ==================== 计算指标 ====================
def calculate_indicators(df):
    """计算所需指标：ATR, EMA9, EMA21, VWAP, 成交量均值"""
    if len(df) < 30:
        return df
    df = df.copy()
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    # EMA
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    # VWAP (20周期成交量加权平均价)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    # 成交量均值（前5根）
    df['vol_ma5'] = df['volume'].shift(1).rolling(5).mean()  # 当前不包含自身
    return df

# ==================== 检测信号 ====================
def check_signals(df):
    """返回最新信号（如果存在）"""
    if len(df) < 30:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 多头条件
    long_condition = (
        last['close'] > last['vwap'] and
        prev['ema9'] <= prev['ema21'] and
        last['ema9'] > last['ema21'] and
        last['volume'] > last['vol_ma5']
    )
    # 空头条件
    short_condition = (
        last['close'] < last['vwap'] and
        prev['ema9'] >= prev['ema21'] and
        last['ema9'] < last['ema21'] and
        last['volume'] > last['vol_ma5']
    )
    
    if long_condition:
        return {'direction': '多', 'price': last['close'], 'atr': last['atr']}
    elif short_condition:
        return {'direction': '空', 'price': last['close'], 'atr': last['atr']}
    else:
        return None

# ==================== 初始化会话状态 ====================
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = {
        '1m': {'direction': None, 'entry': None, 'sl': None, 'tp': None, 'time': None},
        '5m': {'direction': None, 'entry': None, 'sl': None, 'tp': None, 'time': None}
    }

# ==================== 主面板 ====================
col1, col2 = st.columns(2)

for idx, tf in enumerate(['1m', '5m']):
    with [col1, col2][idx]:
        st.subheader(f"{tf} 周期")
        
        # 获取数据
        df = fetch_ohlcv(timeframe=tf)
        if df.empty:
            st.warning("等待数据...")
            continue
        
        # 计算指标
        df = calculate_indicators(df)
        if len(df) < 30:
            st.warning("数据不足")
            continue
        
        # 最新价格和ATR
        last = df.iloc[-1]
        price = last['close']
        atr = last['atr']
        
        # 检测信号
        signal = check_signals(df)
        now = datetime.now()
        
        # 如果检测到新信号，更新 session_state
        if signal:
            direction = signal['direction']
            entry = signal['price']
            atr_val = signal['atr']
            # 计算止损/止盈
            if direction == '多':
                sl = entry - 1.5 * atr_val
                tp = entry + 2.0 * atr_val
            else:
                sl = entry + 1.5 * atr_val
                tp = entry - 2.0 * atr_val
            # 计算盈亏比
            if abs(entry - sl) > 0:
                if direction == '多':
                    rr = (tp - entry) / (entry - sl)
                else:
                    rr = (entry - tp) / (sl - entry)
            else:
                rr = 0
            
            # 更新状态（如果价格有明显变化才视为新信号，防止频繁同方向重复）
            last_sig = st.session_state.last_signal[tf]
            if last_sig['direction'] != direction or abs(entry - (last_sig['entry'] or 0)) > 0.01 * entry:
                st.session_state.last_signal[tf] = {
                    'direction': direction,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'rr': rr,
                    'time': now
                }
        
        # 获取当前状态
        sig = st.session_state.last_signal[tf]
        
        # 显示当前价格和ATR
        col_price, col_atr, col_vol = st.columns(3)
        col_price.metric("当前价", f"{price:.2f}")
        col_atr.metric("ATR(14)", f"{atr:.2f}")
        col_vol.metric("成交量", f"{last['volume']:.0f}")
        
        # 显示信号状态
        if sig['direction']:
            st.success(f"当前信号: **{sig['direction']}**")
            st.metric("入场建议", f"{sig['entry']:.2f}")
            st.metric("止损建议", f"{sig['sl']:.2f}")
            st.metric("止盈建议", f"{sig['tp']:.2f}")
            st.metric("盈亏比预期", f"{sig['rr']:.2f}")
            
            # 回撤预警（基于当前价格与入场价的偏离）
            if sig['direction'] == '多':
                drawdown = (price - sig['entry']) / sig['entry'] * 100
                warning = drawdown < -0.3
            else:
                drawdown = (sig['entry'] - price) / sig['entry'] * 100
                warning = drawdown < -0.3
            
            if warning:
                st.error(f"⚠️ 回撤超过 0.3%！当前回撤: {drawdown:.2f}%")
            else:
                st.info(f"当前回撤: {drawdown:.2f}%")
        else:
            st.info("无信号")
        
        # 显示最近K线时间
        st.caption(f"最新K线: {last['timestamp'].strftime('%H:%M:%S')}")

# 自动刷新
st_autorefresh(interval=5000, key="auto_refresh")  # 5秒刷新

st.markdown("---")
st.caption("策略逻辑：价格突破VWAP + EMA9/21金叉/死叉 + 成交量放大 · 止损 1.5×ATR · 止盈 2×ATR")
