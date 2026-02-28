import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统配置 ====================
st.set_page_config(page_title="ETH V15-80 战地终端", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 进攻型核心引擎 ====================
def battle_commander_v1580(df1, df5, net_flow, buy_ratio):
    # 类型校准
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2)
        d['lower'] = d['ema20'] - (d['std'] * 2)

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    atr = curr1['atr']
    
    # 庄家因子 (V80 核心)
    vol_avg = df1['v'].rolling(20).mean().iloc[-1]
    v_ratio = curr1['v'] / vol_avg if vol_avg > 0 else 1.0
    is_real = v_ratio > 1.02 # 降低门槛：只要稍微放量就视为有效
    
    # 趋势判定
    slope = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-3]) / 3
    is_trend = abs(slope) > (atr * 0.05)

    # 计划初始化
    plan = {"status": "🔭 待机", "action": "观望", "pos": "0%", "entry": None, "tp": None, "sl": None, "reason": "市场动能不足"}
    
    # 【进攻逻辑 A: 趋势火炮】
    if is_trend and is_real:
        if curr1['c'] > curr1['ema20'] and net_flow > 1:
            plan = {
                "status": "🚀 趋势火炮", "action": "做多 (LONG)", "pos": "8%", 
                "entry": curr1['c'], "tp": curr1['c'] + 2.5*atr, "sl": curr1['c'] - 1.2*atr,
                "reason": "趋势向上 + 资金净流入 + 放量确认"
            }
        elif curr1['c'] < curr1['ema20'] and net_flow < -1:
            plan = {
                "status": "🔥 趋势崩塌", "action": "做空 (SHORT)", "pos": "8%", 
                "entry": curr1['c'], "tp": curr1['c'] - 2.5*atr, "sl": curr1['c'] + 1.2*atr,
                "reason": "趋势向下 + 资金撤离 + 恐慌放量"
            }
    
    # 【进攻逻辑 B: 震荡狙击】
    elif not is_trend:
        if curr1['c'] <= curr1['lower'] and buy_ratio > 60:
            plan = {
                "status": "🏹 底部埋伏", "action": "做多 (LONG)", "pos": "5%", 
                "entry": curr1['c'], "tp": curr1['ema20'], "sl": curr1['c'] - 1.0*atr,
                "reason": "触碰布林底轨 + 实时买压强劲"
            }
        elif curr1['c'] >= curr1['upper'] and buy_ratio < 40:
            plan = {
                "status": "🎯 高位狙击", "action": "做空 (SHORT)", "pos": "5%", 
                "entry": curr1['c'], "tp": curr1['ema20'], "sl": curr1['c'] + 1.0*atr,
                "reason": "触碰布林顶轨 + 买力耗尽"
            }

    return plan, is_trend, is_real, atr, v_ratio

# ==================== 3. UI 渲染 ====================
k1_raw = fetch_okx("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx("market/trades", "&limit=50")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t_raw['data'])
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    plan, is_trend, is_real, atr, vol_ratio = battle_commander_v1580(df1, df5, net_f, buy_r)

    st.title("🛡️ ETH 战地指挥官 V15-80 终极版")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1min 净流", f"{net_f:+.2f} ETH")
    c2.metric("实时买压", f"{buy_r:.1f}%")
    c3.metric("市场模式", "🌊 趋势" if is_trend else "⚖️ 震荡")
    c4.metric("动能校验", "✅ 真实" if is_real else "⚠️ 虚假")

    st.markdown("---")
    l_col, r_col = st.columns([1, 2])
    
    with l_col:
        color = "#00FFCC" if "多" in plan['action'] else "#FF4B4B" if "空" in plan['action'] else "#888"
        st.markdown(f"""
            <div style="border:3px solid {color}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.4)">
                <h1 style="color:{color}; margin:0">{plan['status']}</h1>
                <h2 style="margin:5px 0">{plan['action']}</h2>
                <hr>
                <p style="font-size:18px">📍 <b>入场点:</b> {f'${plan["entry"]:.2f}' if plan['entry'] else '---'}</p>
                <p style="font-size:18px; color:#00FFCC">💰 <b>止盈点:</b> {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
                <p style="font-size:18px; color:#FF4B4B">❌ <b>止损点:</b> {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
                <p style="font-size:16px">📊 <b>建议仓位:</b> {plan['pos']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with r_col:
        st.info(f"**战术分析**: {plan['reason']}")
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="K线")])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='gray', width=1, dash='dot'), name="上轨"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='gray', width=1, dash='dot'), name="下轨"))
        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"🕵️ **庄家底牌**: 量比 `{vol_ratio:.2f}` | 波动率 ATR `{atr:.2f}` | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.error("数据链路中断，正在重连...")
