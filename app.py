import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统底层引擎 (高性能) ====================
st.set_page_config(page_title="ETH V220 战神无双版", layout="wide")

def get_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r.get('data', []) if r.get('code') == '0' else []
    except: return []

# ==================== 2. 顶级双模算法 (5M波段+高频抄底) ====================
def tactical_warrior_v220(df1, df5, net_flow, buy_ratio):
    # 指标流水线
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.2)
        d['lower'] = d['ema20'] - (d['std'] * 2.2)
        # 优化版 RSI
        delta = d['c'].diff(); g = (delta.where(delta > 0, 0)).rolling(14).mean()
        l = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + (g/l)))

    c1, c5 = df1.iloc[-1], df5.iloc[-1]
    atr, rsi = c1['atr'], c1['rsi']
    v_avg = df1['v'].rolling(20).mean().iloc[-1]
    vol_ratio = c1['v'] / v_avg if v_avg > 0 else 1.0
    
    # 5M 趋势判定 (波段核心)
    slope_5m = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope_5m) > (atr * 0.15)

    plan = {"action": "📡 待机中", "entry": None, "tp": None, "sl": None, "color": "#888", "desc": "全频扫描中，等待高质量信号"}

    # --- [模式 A] 5M 波段进攻 (趋势截击) ---
    if is_trend:
        if c1['c'] > c1['ema20'] and net_flow > 0.3:
            plan = {"action": "🚀 强势做多", "entry": c1['c'], "tp": c1['c'] + 3.0*atr, "sl": c1['c'] - 1.2*atr, "color": "#00FFCC", "desc": "5M趋势主升浪开启，顺势追击"}
        elif c1['c'] < c1['ema20'] and net_flow < -0.3:
            plan = {"action": "🌊 强势做空", "entry": c1['c'], "tp": c1['c'] - 3.0*atr, "sl": c1['c'] + 1.2*atr, "color": "#FF4B4B", "desc": "5M趋势破位下行，截击空头"}

    # --- [模式 B] 高频极限抄底 (反转狙击) ---
    elif rsi < 31 and c1['c'] < c1['lower']:
        plan = {"action": "🏹 极限抄底", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] - 0.7*atr, "color": "#00FFCC", "desc": "超卖+布林下轨穿透，博取高频反弹"}
    elif rsi > 69 and c1['c'] > c1['upper']:
        plan = {"action": "🎯 高频摸顶", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] + 0.7*atr, "color": "#FF4B4B", "desc": "超买+布林上轨穿透，高频收割空单"}

    return plan, is_trend, vol_ratio, atr, rsi

# ==================== 3. 渲染顶尖实战 UI ====================
raw1 = get_okx_data("market/candles", "&bar=1m&limit=100")
raw5 = get_okx_data("market/candles", "&bar=5m&limit=100")
trades = get_okx_data("market/trades", "&limit=100")

if raw1 and raw5 and trades:
    df1 = pd.DataFrame(raw1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(raw5, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(trades); net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    plan, trend_mode, vr, av, rv = tactical_warrior_v220(df1, df5, net_f, buy_r)

    st.title("🛡️ ETH 战神 V220 无双闭环版")
    col = st.columns(4)
    col[0].metric("净流 (1m)", f"{net_f:+.2f} ETH", delta=f"{buy_r:.1f}%")
    col[1].metric("战斗状态", "🌊 趋势模式" if trend_mode else "⚖️ 震荡模式")
    col[2].metric("RSI 指标", f"{rv:.1f}")
    col[3].metric("实时量比", f"{vr:.2f}x")

    st.markdown("---")
    l_p, r_p = st.columns([1, 2.2])

    with l_p:
        st.markdown(f"""<div style="border:3px solid {plan['color']}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.5)">
            <h1 style="color:{plan['color']}; margin:0">{plan['action']}</h1><hr>
            <p style="font-size:20px">📍 <b>入场参考:</b> {f'${plan["entry"]:.2f}' if plan['entry'] else '等待信号'}</p>
            <p style="font-size:20px; color:#00FFCC">💰 <b>止盈目标:</b> {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
            <p style="font-size:20px; color:#FF4B4B">❌ <b>硬性止损:</b> {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
            <hr><p style="color:#aaa">💡 <b>指挥部建议:</b> {plan['desc']}</p></div>""", unsafe_allow_html=True)
    
    with r_p:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1M K")])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1.5), name="EMA20"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='gray', width=1, dash='dot'), name="BB上"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='gray', width=1, dash='dot'), name="BB下"))
        fig.update_layout(template="plotly_dark", height=480, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"🚀 V220 系统正常运行中 | 刷新时间: {datetime.now().strftime('%H:%M:%S')} | 波动率 ATR: {av:.2f}")
else:
    st.error("数据链路中断，请检查 API 或网络环境")
