import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V200 零度终端", layout="wide")

def fetch_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=3).json()
        return r.get('data', []) if r.get('code') == '0' else []
    except: return []

# ==================== 2. V200 战神引擎 (顶级双模) ====================
def warrior_engine_v200(df1, df5, net_flow, buy_ratio):
    # 底层计算层
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.2)
        d['lower'] = d['ema20'] - (d['std'] * 2.2)
        # 顶级 RSI 计算
        diff = d['c'].diff(); g = (diff.where(diff > 0, 0)).rolling(14).mean()
        l = (-diff.where(diff < 0, 0)).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + (g/l)))

    c1, c5 = df1.iloc[-1], df5.iloc[-1]
    atr, rsi = c1['atr'], c1['rsi']
    vol_ratio = c1['v'] / df1['v'].rolling(20).mean().iloc[-1]
    
    # 模式识别：5M 波段斜率 + 波动强度
    slope_5m = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope_5m) > (atr * 0.2)
    
    plan = {"status": "🔭 待机", "action": "全频扫描中", "entry": None, "tp": None, "sl": None, "color": "#888", "tip": "市场进入零度波动，等待爆发"}

    # --- 模式 A：5M 波段顺势 (高频截击) ---
    if is_trend:
        if c1['c'] > c1['ema20'] and net_flow > 1.0:
            plan = {"status": "🚀 趋势波段", "action": "做多 (LONG)", "entry": c1['c'], "tp": c1['c'] + 3.0*atr, "sl": c1['c'] - 1.2*atr, "color": "#00FFCC", "tip": "5M多头共振，建议 10% 仓位追击"}
        elif c1['c'] < c1['ema20'] and net_flow < -1.0:
            plan = {"status": "🌊 趋势波段", "action": "做空 (SHORT)", "entry": c1['c'], "tp": c1['c'] - 3.0*atr, "sl": c1['c'] + 1.2*atr, "color": "#FF4B4B", "tip": "5M破位下行，抛压释放中"}

    # --- 模式 B：极限抄底摸顶 (高频反转) ---
    elif rsi < 32 and c1['c'] < c1['lower']:
        plan = {"status": "🏹 极限抄底", "action": "博反弹 (BUY)", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] - 0.7*atr, "color": "#00FFCC", "tip": "严重超卖 + 布林击穿，小仓位博弈"}
    elif rsi > 68 and c1['c'] > c1['upper']:
        plan = {"status": "🎯 极限摸顶", "action": "博回调 (SELL)", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] + 0.7*atr, "color": "#FF4B4B", "tip": "严重超买 + 抛压预警，及时收割"}

    return plan, is_trend, vol_ratio, atr, rsi

# ==================== 3. 渲染顶尖实战 UI ====================
k1, k5, t = fetch_data("market/candles", "&bar=1m&limit=100"), fetch_data("market/candles", "&bar=5m&limit=100"), fetch_data("market/trades", "&limit=100")

if k1 and k5 and t:
    df1 = pd.DataFrame(k1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t); net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    plan, it, vr, av, rv = warrior_engine_v200(df1, df5, net_f, buy_r)

    st.title("🛡️ ETH 战神 V200 零度终极终端")
    c = st.columns(4)
    c[0].metric("净流 (ETH)", f"{net_f:+.2f}", delta=f"{buy_r:.1f}%")
    c[1].metric("战斗模式", "🌊 趋势" if it else "⚖️ 震荡")
    c[2].metric("相对强弱 RSI", f"{rv:.1f}")
    c[3].metric("实时量比", f"{vr:.2f}x")

    st.markdown("---")
    l, r = st.columns([1, 2.2])

    with l:
        st.markdown(f"""<div style="border:3px solid {plan['color']}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.5)">
            <h1 style="color:{plan['color']}; margin:0">{plan['action']}</h1>
            <h2 style="color:#aaa">{plan['status']}</h2><hr>
            <p style="font-size:20px">📍 <b>入场点:</b> {f'${plan["entry"]:.2f}' if plan['entry'] else '等待信号'}</p>
            <p style="font-size:20px; color:#00FFCC">💰 <b>止盈位:</b> {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
            <p style="font-size:20px; color:#FF4B4B">❌ <b>止损位:</b> {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
            <hr><p style="color:{plan['color']}">💡 {plan['tip']}</p></div>""", unsafe_allow_html=True)
        if plan['entry']: st.code(f"EXECUTE: {plan['action']} @{plan['entry']} SL:{plan['sl']:.2f} TP:{plan['tp']:.2f}")

    with r:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1M K")])
        for line, col in [('ema20','yellow'), ('upper','gray'), ('lower','gray')]:
            fig.add_trace(go.Scatter(x=df1.index, y=df1[line], line=dict(color=col, width=1, dash='dot' if 'upper' in line or 'lower' in line else 'solid'), name=line))
        fig.update_layout(template="plotly_dark", height=480, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"🚀 V200 终极态在线 | 刷新时间: {datetime.now().strftime('%H:%M:%S')} | 波动率: {av:.2f}")
else:
    st.warning("数据链路重连中... 请检查 API 权重或网络连接。")
