import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V15-80 战地指挥官", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 进攻型核心引擎 (整合 V15-V80) ====================
def battle_commander_engine(df1, df5, net_flow, buy_ratio):
    # 统一转换类型
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        # 布林带 (V75 核心)
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2)
        d['lower'] = d['ema20'] - (d['std'] * 2)

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    atr = curr1['atr']
    
    # 【V15 基本面雷达】
    f_score = (2 if net_flow > 3 else -2 if net_flow < -3 else 0) + (1 if buy_ratio > 55 else -1 if buy_ratio < 45 else 0)
    
    # 【V80 庄家辨伪】
    vol_q = curr1['v'] / df1['v'].rolling(20).mean().iloc[-1]
    is_real = vol_q > 1.05 # 降低门槛，只要放量就视为真实
    
    # 【V75 趋势/震荡双模】
    slope5 = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope5) > (atr * 0.1)

    # 【V65 AI 战术生成器】
    plan = {"status": "🔭 待机", "action": "观望", "pos": "0%", "entry": None, "tp": None, "sl": None, "reason": "市场动能不足"}
    
    # 策略 A: 趋势进攻 (V80 核心)
    if is_trend and is_real:
        if curr1['c'] > curr1['ema20'] and f_score >= 2:
            plan = {
                "status": "🚀 趋势火炮", "action": "做多 (LONG)", "pos": "5-10%", 
                "entry": curr1['c'], "tp": curr1['c'] + 2.5*atr, "sl": curr1['c'] - 1.5*atr,
                "reason": "5M趋势向上 + 1M资金流入确认"
            }
        elif curr1['c'] < curr1['ema20'] and f_score <= -2:
            plan = {
                "status": "🔥 趋势崩塌", "action": "做空 (SHORT)", "pos": "5-10%", 
                "entry": curr1['c'], "tp": curr1['c'] - 2.5*atr, "sl": curr1['c'] + 1.5*atr,
                "reason": "5M下行结构 + 抛压真实释放"
            }
    
    # 策略 B: 震荡收割 (V75 核心)
    elif not is_trend:
        if curr1['c'] <= curr1['lower'] and f_score >= 1:
            plan = {
                "status": "🏹 底部埋伏", "action": "做多 (LONG)", "pos": "3%", 
                "entry": curr1['c'], "tp": curr1['ema20'], "sl": curr1['c'] - 1.2*atr,
                "reason": "触碰布林下轨 + 资金暗中护盘"
            }
        elif curr1['c'] >= curr1['upper'] and f_score <= -1:
            plan = {
                "status": "🎯 高位狙击", "action": "做空 (SHORT)", "pos": "3%", 
                "entry": curr1['c'], "tp": curr1['ema20'], "sl": curr1['c'] + 1.2*atr,
                "reason": "触碰布林上轨 + 获利盘了结"
            }

    return plan, f_score, is_trend, is_real, curr1['atr']

# ==================== 3. 界面渲染 (回归 V15 暴力美学) ====================
k1_raw = fetch_okx("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx("market/trades", "&limit=50")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','vol','vq','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','vol','vq','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t_raw['data'])
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    plan, f_score, is_trend, is_real, atr = battle_commander_engine(df1, df5, net_f, buy_r)

    # --- 顶部分栏: 核心数据 ---
    st.title("🛡️ ETH 战地指挥官 V15-80 终极版")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("资金净流", f"{net_f:+.2f} ETH", delta=f"{f_score} 分")
    c2.metric("买压占比", f"{buy_r:.1f}%")
    c3.metric("市场模式", "🌊 趋势" if is_trend else "⚖️ 震荡")
    c4.metric("动能校验", "✅ 真实" if is_real else "⚠️ 虚假")

    # --- 核心: 详细交易计划策略表 (V65 灵魂) ---
    st.markdown("---")
    st.subheader("📝 实时作战指令计划")
    
    col_l, col_r = st.columns([1, 2])
    with col_l:
        color = "#00FFCC" if "多" in plan['action'] else "#FF4B4B" if "空" in plan['action'] else "#888"
        st.markdown(f"""
            <div style="border:3px solid {color}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.3)">
                <h1 style="color:{color}; margin:0">{plan['status']}</h1>
                <h2 style="margin:5px 0">{plan['action']}</h2>
                <hr>
                <p style="font-size:18px">📍 <b>入场点:</b> {f"${plan['entry']:.2f}" if plan['entry'] else "---"}</p>
                <p style="font-size:18px; color:#00FFCC">💰 <b>止盈点:</b> {f"${plan['tp']:.2f}" if plan['tp'] else "---"}</p>
                <p style="font-size:18px; color:#FF4B4B">❌ <b>止损点:</b> {f"${plan['sl']:.2f}" if plan['sl'] else "---"}</p>
                <p style="font-size:16px">📊 <b>建议仓位:</b> {plan['pos']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_r:
        st.info(f"**战术分析建议**: {plan['reason']}")
        # 实时 K 线图 (带布林带和 EMA)
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1M K")])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="上轨"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="下轨"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # --- 底部分栏: 庄家雷达 ---
    st.markdown("---")
    st.write(f"🕵️ **庄家动态**: 1M 量比 `{vol_ratio:.2f}` | 波动率 ATR `{atr:.2f}` | 当前时间: {datetime.now().strftime('%H:%M:%S')}")

else:
    st.error("正在强行接入 OKX 战地数据源...")
