# -*- coding: utf-8 -*-
"""
🤖 ETH 合約短線策略監控系統 V1.0
核心邏輯：1m + 5m 雙週期共振
指標：VWAP + EMA9/21 金叉/死叉 + 成交量爆量
風險：100倍槓桿，SL=1.5×ATR 或 0.3% 強制，TP=3×ATR（盈虧比 2:1）
只監控不交易，實時從 Binance 期貨獲取數據
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import time

st.set_page_config(page_title="ETH 短線監控", layout="wide")
st.title("🚀 ETH 合約短線策略監控系統 V1.0")
st.caption("1m + 5m 雙週期共振 • VWAP + EMA9/21 + ATR14 • 實時監控 • 每8秒刷新")

# ==================== 配置 ====================
SYMBOL = "ETH/USDT:USDT"   # Binance 永續合約
EXCHANGE = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# ==================== 數據獲取 ====================
@st.cache_data(ttl=8)
def fetch_klines(timeframe, limit=500):
    ohlcv = EXCHANGE.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    df = df.copy()
    df['ema9'] = ta.trend.ema_indicator(df['close'], 9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], 21)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
    
    # VWAP（累積計算，接近當日VWAP效果）
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['tpv'] = typical_price * df['volume']
    df['cum_tpv'] = df['tpv'].cumsum()
    df['cum_vol'] = df['volume'].cumsum()
    df['vwap'] = df['cum_tpv'] / df['cum_vol']
    
    # 成交量5根均值
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    return df

# ==================== 信號邏輯 ====================
def detect_signal(df):
    if len(df) < 30:
        return None, None, None, None, None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # EMA 金叉/死叉
    ema_cross_long = (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21'])
    ema_cross_short = (prev['ema9'] > prev['ema21']) and (last['ema9'] < last['ema21'])
    
    # 價格突破VWAP + 成交量爆量
    vol_condition = last['volume'] > last['vol_ma5'] * 1.3
    
    long_signal = ema_cross_long and (last['close'] > last['vwap']) and vol_condition
    short_signal = ema_cross_short and (last['close'] < last['vwap']) and vol_condition
    
    if long_signal or short_signal:
        direction = "多頭計劃 🔥" if long_signal else "空頭計劃 🔥"
        entry = last['close']
        atr = last['atr']
        sl = entry - 1.5 * atr if long_signal else entry + 1.5 * atr
        # 0.3% 強制保護
        sl = max(sl, entry * 0.997) if long_signal else min(sl, entry * 1.003)
        tp = entry + 3 * atr if long_signal else entry - 3 * atr
        rr = abs((tp - entry) / (entry - sl)) if long_signal else abs((entry - tp) / (sl - entry))
        
        return direction, round(entry, 4), round(sl, 4), round(tp, 4), round(rr, 2)
    
    return "觀望", None, None, None, None

# ==================== Streamlit 儀表板 ====================
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("1分鐘圖表")
    df1 = add_indicators(fetch_klines('1m'))
    signal1, entry1, sl1, tp1, rr1 = detect_signal(df1)
    
    fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.20, 0.25])
    fig1.add_trace(go.Candlestick(x=df1['timestamp'], open=df1['open'], high=df1['high'], low=df1['low'], close=df1['close']), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1['timestamp'], y=df1['vwap'], name="VWAP", line=dict(color="#ffd700", width=2)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1['timestamp'], y=df1['ema9'], name="EMA9", line=dict(color="#00ff9d")), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1['timestamp'], y=df1['ema21'], name="EMA21", line=dict(color="#ff4d4d")), row=1, col=1)
    fig1.add_trace(go.Bar(x=df1['timestamp'], y=df1['volume'], name="成交量"), row=2, col=1)
    fig1.add_trace(go.Bar(x=df1['timestamp'], y=df1['macd_hist'] if 'macd_hist' in df1 else df1['volume']*0, name="MACD柱"), row=3, col=1)
    st.plotly_chart(fig1, use_container_width=True, width="stretch")

with col2:
    st.subheader("5分鐘圖表")
    df5 = add_indicators(fetch_klines('5m'))
    signal5, entry5, sl5, tp5, rr5 = detect_signal(df5)
    
    fig5 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.20, 0.25])
    fig5.add_trace(go.Candlestick(x=df5['timestamp'], open=df5['open'], high=df5['high'], low=df5['low'], close=df5['close']), row=1, col=1)
    fig5.add_trace(go.Scatter(x=df5['timestamp'], y=df5['vwap'], name="VWAP", line=dict(color="#ffd700", width=2)), row=1, col=1)
    fig5.add_trace(go.Scatter(x=df5['timestamp'], y=df5['ema9'], name="EMA9", line=dict(color="#00ff9d")), row=1, col=1)
    fig5.add_trace(go.Scatter(x=df5['timestamp'], y=df5['ema21'], name="EMA21", line=dict(color="#ff4d4d")), row=1, col=1)
    st.plotly_chart(fig5, use_container_width=True, width="stretch")

# ==================== 信號總結 ====================
st.divider()
col_sig1, col_sig2, col_sig3 = st.columns(3)

with col_sig1:
    st.metric("1分鐘信號", signal1 or "觀望")
    if signal1 and "計劃" in signal1:
        st.success(f"入場建議價: **{entry1}**")
        st.error(f"止損建議價: **{sl1}**")
        st.success(f"止盈建議價: **{tp1}**")
        st.info(f"盈虧比預期: **{rr1}:1**")

with col_sig2:
    st.metric("5分鐘信號", signal5 or "觀望")
    if signal5 and "計劃" in signal5:
        st.success(f"入場建議價: **{entry5}**")
        st.error(f"止損建議價: **{sl5}**")
        st.success(f"止盈建議價: **{tp5}**")
        st.info(f"盈虧比預期: **{rr5}:1**")

with col_sig3:
    st.subheader("當前市場狀態")
    price = fetch_klines('1m')['close'].iloc[-1]
    st.metric("ETH 最新價", f"${price:,.2f}", f"{(price - fetch_klines('1m')['close'].iloc[-2])/fetch_klines('1m')['close'].iloc[-2]*100:+.2f}%")

# ==================== 終端實時打印 ====================
with st.expander("📜 終端實時日誌"):
    st.write(f"[{datetime.now().strftime('%H:%M:%S')}] 1m信號: {signal1 or '觀望'} | 5m信號: {signal5 or '觀望'}")
    if signal1 and "計劃" in signal1:
        st.write(f"   → 多頭計劃 | 入場 {entry1} | SL {sl1} | TP {tp1} | RR {rr1}:1")
    if signal5 and "計劃" in signal5:
        st.write(f"   → 空頭計劃 | 入場 {entry5} | SL {sl5} | TP {tp5} | RR {rr5}:1")

st_autorefresh(interval=8000, key="auto")

st.caption("只監控 • 不執行任何下單 • 數據來自 Binance 永續合約")
