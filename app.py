import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统配置 ====================
st.set_page_config(page_title="ETH V175 战神终端", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 终极双模战术引擎 ====================
def warrior_engine_v175(df1, df5, net_flow, buy_ratio):
    # 指标计算 (严格按照 15.0-165.0 演进标准)
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.1)
        d['lower'] = d['ema20'] - (d['std'] * 2.1)

    c1, c5 = df1.iloc[-1], df5.iloc[-1]
    atr_val = c1['atr']
    
    # 修复 V160 报错：强制定义量比
    vol_avg = df1['v'].rolling(20).mean().iloc[-1]
    v_ratio = c1['v'] / vol_avg if vol_avg > 0 else 1.0
    
    # 模式识别：5分钟趋势斜率
    slope_5m = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope_5m) > (atr_val * 0.1)

    # 计划初始化
    plan = {"mode": "📡 深度扫描", "action": "等待时机", "entry": None, "tp": None, "sl": None, "desc": "动能不足，暂无高质量信号"}

    # --- 模式 A：5Min 波段进攻 (V80-V160 结晶) ---
    if is_trend:
        if c1['c'] < c1['ema20'] and net_flow < -0.2:
            plan = {
                "mode": "🌊 趋势波段-空头", "action": "做空 (SHORT)",
                "entry": c1['c'], "tp": c1['c'] - 2.8*atr_val, "sl": c1['c'] + 1.2*atr_val,
                "desc": "5M趋势破位 + 净流出确认，建议仓位 8-10%"
            }
        elif c1['c'] > c1['ema20'] and net_flow > 0.2:
            plan = {
                "mode": "🚀 趋势波段-多头", "action": "做多 (LONG)",
                "entry": c1['c'], "tp": c1['c'] + 2.8*atr_val, "sl": c1['c'] - 1.2*atr_val,
                "desc": "5M多头共振 + 资金点火，建议仓位 8-10%"
            }

    # --- 模式 B：高频抄底摸顶 (V15-V75 结晶) ---
    else:
        if c1['c'] <= c1['lower'] and buy_ratio > 55:
            plan = {
                "mode": "🏹 高频抄底-极限", "action": "博多 (BUY)",
                "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] - 0.8*atr_val,
                "desc": "触碰布林下轨 + 买压回归，建议仓位 5%"
            }
        elif c1['c'] >= c1['upper'] and buy_ratio < 45:
            plan = {
                "mode": "🎯 高频摸顶-抛压", "action": "博空 (SELL)",
                "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] + 0.8*atr_val,
                "desc": "触碰布林上轨 + 抛压骤增，建议仓位 5%"
            }

    return plan, is_trend, v_ratio, atr_val

# ==================== 3. 界面渲染 ====================
# 获取数据
k1_raw = fetch_okx("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx("market/trades", "&limit=60")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t_raw['data'])
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    # 运行战神引擎
    plan, is_trend, vol_r, atr_v = warrior_engine_v175(df1, df5, net_f, buy_r)

    st.title("🏹 ETH 战神 V175 终极实战终端")
    
    # 顶部面板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("资金净流 (1m)", f"{net_f:+.2f} ETH")
    c2.metric("当前战斗模式", plan['mode'])
    c3.metric("实时量比", f"{vol_r:.2f}x")
    c4.metric("波动系数", f"{atr_v:.2f}")

    st.markdown("---")
    l_col, r_col = st.columns([1, 2])

    with l_col:
        accent = "#FF4B4B" if "空" in plan['mode'] or "SELL" in plan['action'] else "#00FFCC" if "多" in plan['mode'] or "BUY" in plan['action'] else "#888"
        st.markdown(f"""
            <div style="border:3px solid {accent}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.3)">
                <h1 style="color:{accent}; margin:0">{plan['action']}</h1>
                <h3 style="margin:10px 0">{plan['mode']}</h3>
                <hr>
                <p style="font-size:18px">📍 <b>入场点参考:</b> {f'${plan["entry"]:.2f}' if plan['entry'] else '等待信号'}</p>
                <p style="font-size:18px; color:#00FFCC">💰 <b>目标止盈位:</b> {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
                <p style="font-size:18px; color:#FF4B4B">❌ <b>硬性止损位:</b> {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
                <hr>
                <p style="font-size:16px; color:#aaa">💡 <b>战地建议:</b> {plan['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    with r_col:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1m K")])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="BB上"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="BB下"))
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**⚡ 战神核心在线** | 刷新时间: {datetime.now().strftime('%H:%M:%S')} | 庄家买压: `{buy_r:.1f}%` | 建议仓位: `5-10%`")
else:
    st.error("正在强行建立数据通信链路...")
