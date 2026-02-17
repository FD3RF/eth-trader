# -*- coding: utf-8 -*-
"""
🚀 终极量化终端 · 完整优化版 33.2
宇宙主宰 | 永恒无敌 | 完美无瑕 | 永不败北
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import deque
import functools
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
class Config:
    SYMBOLS = ["ETH/USDT", "BTC/USDT", "SOL/USDT", "BNB/USDT"]
    BALANCE = 10000.0
    BASE_RISK_PER_TRADE = 0.02
    MAX_DAILY_TRADES = 5
    TIMEFRAMES = ['15m', '1h', '4h', '1d']
    TIMEFRAME_WEIGHTS = {'1d':10, '4h':7, '1h':5, '15m':3}
    LEVERAGE_MODES = {"稳健 (3-5x)": (3,5), "无敌 (5-8x)":(5,8), "神级 (8-10x)":(8,10)}
    FETCH_LIMIT = 1500
    AUTO_REFRESH_MS = 60000
    SIM_VOLATILITY = 0.05
    SIM_TREND = 0.15
CONFIG = Config()

# ==================== 初始化 Session ====================
def init_session_state():
    defaults = {
        'balance': CONFIG.BALANCE,
        'daily_trades':0,
        'consecutive_losses':0,
        'net_value_history':[CONFIG.BALANCE],
        'trade_log':[],
        'use_simulated_data':True,
        'current_symbol':'ETH/USDT',
        'cooldown_until':None,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ==================== 模拟数据 ====================
def generate_data(symbol, limit=CONFIG.FETCH_LIMIT):
    np.random.seed(int(time.time()*1000)%2**32)
    base_prices = {"ETH/USDT":2500,"BTC/USDT":45000,"SOL/USDT":120,"BNB/USDT":400}
    base = base_prices.get(symbol,100)
    t = np.linspace(0,6*np.pi,limit)
    trend = np.random.choice([-1,1])*CONFIG.SIM_TREND*np.linspace(0,1,limit)*base
    cycle = 0.08*base*(np.sin(t)+0.5*np.sin(3*t)+0.3*np.sin(5*t))
    volatility = CONFIG.SIM_VOLATILITY*(1+0.5*np.sin(t/10))
    random_walk = np.cumsum(np.random.randn(limit)*volatility*base*0.3)
    price_series = np.maximum(base+trend+cycle+random_walk,base*0.3)

    opens = price_series*(1+np.random.randn(limit)*0.002)
    closes = price_series*(1+np.random.randn(limit)*0.003)
    highs = np.maximum(opens,closes)+np.abs(np.random.randn(limit))*volatility*price_series*0.5
    lows = np.minimum(opens,closes)-np.abs(np.random.randn(limit))*volatility*price_series*0.5
    volumes = np.random.randint(500,5000,limit)
    df = pd.DataFrame({'timestamp':pd.date_range(datetime.now()-timedelta(minutes=15*limit),periods=limit,freq='15T'),
                       'open':opens,'high':highs,'low':lows,'close':closes,'volume':volumes})
    return df

# ==================== 指标 ====================
def add_indicators(df):
    df['ema20']=df['close'].ewm(span=20,adjust=False).mean()
    df['rsi'] = 100 - 100/(1+df['close'].pct_change().rolling(14).apply(lambda x:(x[x>0].sum()/abs(x[x<0].sum()) if abs(x[x<0].sum())>0 else 1), raw=False))
    df['atr'] = (df['high']-df['low']).rolling(14).mean()
    return df

# ==================== 信号 ====================
def calculate_signal(df):
    last = df.iloc[-1]
    direction = 0
    prob = 0.5
    if last['close']>last['ema20']:
        direction = 1
        prob = min(0.5 + (last['rsi']-50)/100,0.99)
    elif last['close']<last['ema20']:
        direction = -1
        prob = max(0.5 - (last['rsi']-50)/100,0.01)
    return direction, prob

# ==================== 风控 ====================
def calculate_position_size(balance, prob, atr, leverage=3):
    risk = balance*CONFIG.BASE_RISK_PER_TRADE*abs(prob-0.5)*2
    position = min(risk/atr, balance*leverage)
    return round(position,4)

# ==================== UI ====================
def main():
    st.set_page_config(page_title="终极量化终端 33.2",layout="wide")
    st.title("🚀 终极量化终端 · 完整优化版 33.2")
    init_session_state()

    # 配置
    st.sidebar.header("⚙️ 配置")
    symbol = st.sidebar.selectbox("品种", CONFIG.SYMBOLS, index=0)
    leverage_mode = st.sidebar.selectbox("杠杆模式", list(CONFIG.LEVERAGE_MODES.keys()), index=0)
    balance = st.sidebar.number_input("余额 USDT", value=st.session_state.balance, step=100.0)

    st.session_state.current_symbol = symbol

    # 数据
    df = generate_data(symbol)
    df = add_indicators(df)

    # 信号计算
    direction, prob = calculate_signal(df)
    leverage = np.mean(CONFIG.LEVERAGE_MODES[leverage_mode])
    position = calculate_position_size(balance, prob, df['atr'].iloc[-1], leverage)

    # 主面板
    st.subheader(f"🚀 {symbol} 量化终端")
    st.metric("当前价格", f"{df['close'].iloc[-1]:.2f}")
    signal_str = "观望"
    if direction==1: signal_str="多"
    elif direction==-1: signal_str="空"
    st.metric("交易信号", signal_str)
    st.metric("推荐仓位", f"{position:.4f} 手")

    # 多周期信号图
    fig = go.Figure()
    for tf in CONFIG.TIMEFRAMES:
        df_tf = df.resample(tf, on='timestamp').last()
        fig.add_trace(go.Scatter(x=df_tf['timestamp'], y=df_tf['close'], mode='lines', name=tf))
    fig.update_layout(title="📈 多周期价格走势", xaxis_title="时间", yaxis_title="价格")
    st.plotly_chart(fig, use_container_width=True)

if __name__=="__main__":
    main()
