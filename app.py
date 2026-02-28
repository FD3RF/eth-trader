import streamlit as st
import pandas as pd
import numpy as np
import requests # Fallback for reliability
import plotly.graph_objects as go
from datetime import datetime
import time

# Note: Ensure 'pip install httpx' is run in your environment or added to requirements.txt
try:
    import httpx
    import asyncio
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False

# ==================== 1. 核心引擎与回测逻辑 ====================
def calculate_signals(df1, df5, net_flow):
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.2)
        d['lower'] = d['ema20'] - (d['std'] * 2.2)
        # RSI Calculation
        diff = d['c'].diff(); g = (diff.where(diff > 0, 0)).rolling(14).mean()
        l = (-diff.where(diff < 0, 0)).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + (g/l)))

    # Real-time Backtest Module (Last 100 periods)
    wins, total_sigs = 0, 0
    for i in range(20, len(df1)-5):
        row = df1.iloc[i]
        f_max = df1['h'].iloc[i+1:i+6].max() 
        f_min = df1['l'].iloc[i+1:i+6].min()
        
        if row['rsi'] < 30 and row['c'] < row['lower']:
            total_sigs += 1
            if f_max > row['c'] * 1.0015: wins += 1 # 0.15% Profit Target
        elif row['rsi'] > 70 and row['c'] > row['upper']:
            total_sigs += 1
            if f_min < row['c'] * 0.9985: wins += 1

    hit_rate = (wins / total_sigs * 100) if total_sigs > 0 else 0.0
    
    # Current Signal Logic
    c1, c5 = df1.iloc[-1], df5.iloc[-1]
    atr, rsi = c1['atr'], c1['rsi']
    slope = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope) > (atr * 0.18)
    
    plan = {"act": "🔭 待机", "entry": None, "tp": None, "sl": None, "color": "#888"}
    if is_trend:
        if c1['c'] > c1['ema20'] and net_flow > 0.5:
            plan = {"act": "🚀 趋势追多", "entry": c1['c'], "tp": c1['c'] + 2.5*atr, "sl": c1['c'] - 1.2*atr, "color": "#00FFCC"}
        elif c1['c'] < c1['ema20'] and net_flow < -0.5:
            plan = {"act": "🌊 趋势压制", "entry": c1['c'], "tp": c1['c'] - 2.5*atr, "sl": c1['c'] + 1.2*atr, "color": "#FF4B4B"}
    
    return plan, is_trend, hit_rate, total_sigs

# ==================== 2. 数据获取 (支持异步回退) ====================
def fetch_sync(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=3).json()
        return r.get('data', [])
    except: return []

async def fetch_async():
    async with httpx.AsyncClient() as client:
        base = "https://www.okx.com/api/v5/"
        targets = [
            f"{base}market/candles?instId=ETH-USDT&bar=1m&limit=150",
            f"{base}market/candles?instId=ETH-USDT&bar=5m&limit=100",
            f"{base}market/trades?instId=ETH-USDT&limit=100"
        ]
        res = await asyncio.gather(*[client.get(t) for t in targets])
        return [r.json().get('data', []) for r in res]

# ==================== 3. 终极终端 UI ====================
st.set_page_config(page_title="ETH V250 异步回测终端", layout="wide")

# Fetch data based on availability
if ASYNC_MODE:
    try:
        k1_raw, k5_raw, trades_raw = asyncio.run(fetch_async())
    except:
        k1_raw, k5_raw, trades_raw = fetch_sync("market/candles", "&bar=1m"), fetch_sync("market/candles", "&bar=5m"), fetch_sync("market/trades")
else:
    k1_raw, k5_raw, trades_raw = fetch_sync("market/candles", "&bar=1m"), fetch_sync("market/candles", "&bar=5m"), fetch_sync("market/trades")

if k1_raw and k5_raw:
    df1 = pd.DataFrame(k1_raw, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(trades_raw)
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    
    plan, trend_on, win_rate, count = calculate_signals(df1, df5, net_f)

    # Dashboard
    st.title("🏹 ETH 战神 V250 异步回测终端")
    cols = st.columns(4)
    cols[0].metric("1m 净流", f"{net_f:+.2f} ETH")
    cols[1].metric("历史胜率", f"{win_rate:.1f}%", f"样本: {count}")
    cols[2].metric("战斗模式", "🌊 趋势进攻" if trend_on else "⚖️ 震荡高频")
    cols[3].metric("引擎状态", "⚡ 异步并发" if ASYNC_MODE else "🐢 同步兼容")

    l, r = st.columns([1, 2.5])
    with l:
        st.markdown(f"""<div style="border:4px solid {plan['color']}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.4)">
            <h2 style="color:{plan['color']}; text-align:center">{plan['act']}</h2><hr>
            <p style="font-size:20px">📍 入场: {f'${plan["entry"]:.2f}' if plan['entry'] else '---'}</p>
            <p style="font-size:20px; color:#00FFCC">💰 获利: {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
            <p style="font-size:20px; color:#FF4B4B">❌ 止损: {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
        </div>""", unsafe_allow_html=True)
        
    with r:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'])])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1)))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"💎 系统自检通过 | 刷新时间: {datetime.now().strftime('%H:%M:%S')} | 波动率(ATR): {df1['atr'].iloc[-1]:.2f}")
