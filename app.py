import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统核心配置 (终极稳健) ====================
st.set_page_config(page_title="ETH V240 战神·真理终章", layout="wide")

def fetch_okx_logic(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=3).json()
        return r.get('data', []) if r.get('code') == '0' else []
    except: return []

# ==================== 2. 真理引擎 (15.0-230.0 结晶) ====================
def truth_engine_v240(df1, df5, net_flow, buy_ratio):
    # 底层基因对齐
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.2)
        d['lower'] = d['ema20'] - (d['std'] * 2.2)
        # RSI 裁决计算
        diff = d['c'].diff(); g = (diff.where(diff > 0, 0)).rolling(14).mean()
        l = (-diff.where(diff < 0, 0)).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + (g/l)))

    c1, c5 = df1.iloc[-1], df5.iloc[-1]
    atr, rsi = c1['atr'], c1['rsi']
    v_avg = df1['v'].rolling(20).mean().iloc[-1]
    vol_ratio = c1['v'] / v_avg if v_avg > 0 else 1.0 # 物理防错
    
    # 真理模式识别 (动态阈值)
    slope = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope) > (atr * 0.18)
    
    res = {"act": "🔭 战前待机", "entry": None, "tp": None, "sl": None, "color": "#888", "desc": "正在捕捉高质量时空节点..."}

    # --- [战术 A] 5M 趋势截断 ---
    if is_trend:
        if c1['c'] > c1['ema20'] and net_flow > 0.8:
            res = {"act": "🚀 趋势追多", "entry": c1['c'], "tp": c1['c'] + 3.0*atr, "sl": c1['c'] - 1.1*atr, "color": "#00FFCC", "desc": "多头基因觉醒，主力资金已入场"}
        elif c1['c'] < c1['ema20'] and net_flow < -0.8:
            res = {"act": "🌊 趋势压制", "entry": c1['c'], "tp": c1['c'] - 3.0*atr, "sl": c1['c'] + 1.1*atr, "color": "#FF4B4B", "desc": "空头基因占优，阻力位向下平移"}

    # --- [战术 B] 高频极限狙击 ---
    elif rsi < 30 and c1['c'] < c1['lower']:
        res = {"act": "🏹 极限抄底", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] - 0.7*atr, "color": "#00FFCC", "desc": "触发超卖裁决，寻找超短线回升"}
    elif rsi > 70 and c1['c'] > c1['upper']:
        res = {"act": "🎯 高频收割", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] + 0.7*atr, "color": "#FF4B4B", "desc": "触发超买裁决，等待快速回调均线"}

    return res, is_trend, vol_ratio, atr, rsi

# ==================== 3. 终极实战 UI ====================
raw1, raw5 = fetch_okx_logic("market/candles", "&bar=1m&limit=100"), fetch_okx_logic("market/candles", "&bar=5m&limit=100")
trades = fetch_okx_logic("market/trades", "&limit=100")

if raw1 and raw5 and trades:
    df1 = pd.DataFrame(raw1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(raw5, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(trades); net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    plan, trend_on, vr, av, rv = truth_engine_v240(df1, df5, net_f, buy_r)

    st.title("🏹 ETH 战神 V240 真理终章终端")
    c = st.columns(4)
    c[0].metric("1m 净流", f"{net_f:+.2f} ETH", f"{buy_r:.1f}%")
    c[1].metric("战斗核心", "🌊 趋势进攻" if trend_on else "⚖️ 高频狙击")
    c[2].metric("真理 RSI", f"{rv:.1f}")
    c[3].metric("动态量比", f"{vr:.2f}x")

    st.markdown("---")
    l, r = st.columns([1, 2.3])

    with l:
        st.markdown(f"""<div style="border:4px solid {plan['color']}; padding:22px; border-radius:20px; background:rgba(0,0,0,0.5); box-shadow: 0 0 15px {plan['color']}44">
            <h1 style="color:{plan['color']}; margin:0; text-align:center">{plan['act']}</h1><hr>
            <p style="font-size:22px">📍 <b>入场参考:</b> {f'${plan["entry"]:.2f}' if plan['entry'] else '---'}</p>
            <p style="font-size:22px; color:#00FFCC">💰 <b>获利裁决:</b> {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
            <p style="font-size:22px; color:#FF4B4B">❌ <b>硬核止损:</b> {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
            <hr><p style="color:#eee; font-size:17px">💡 <b>裁决建议:</b> {plan['desc']}</p></div>""", unsafe_allow_html=True)
        if plan['entry']: st.code(f"EXEC: {plan['act']} @{plan['entry']} TP:{plan['tp']:.2f} SL:{plan['sl']:.2f}")

    with r:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1M")])
        for line, col, style in [('ema20','yellow','solid'), ('upper','gray','dot'), ('lower','gray','dot')]:
            fig.add_trace(go.Scatter(x=df1.index, y=df1[line], line=dict(color=col, width=1.2, dash=style), name=line))
        fig.update_layout(template="plotly_dark", height=520, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"💎 真理逻辑全开 | 当前波动: {av:.2f} | 最终心跳: {datetime.now().strftime('%H:%M:%S')}")
else: st.warning("战地链路重启中...")
