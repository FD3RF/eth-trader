import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V230 最终裁决版", layout="wide")

def fetch_okx_v5(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=3).json()
        return r.get('data', []) if r.get('code') == '0' else []
    except: return []

# ==================== 2. V230 战神引擎 (顶级双模合一) ====================
def ultimate_engine_v230(df1, df5, net_flow, buy_ratio):
    # 底层计算闭环
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
    # 彻底杜绝 vol_ratio 报错
    v_avg = df1['v'].rolling(20).mean().iloc[-1]
    v_ratio = c1['v'] / v_avg if v_avg > 0 else 1.0
    
    # 模式识别逻辑
    slope_5m = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope_5m) > (atr * 0.18)

    plan = {"action": "🔭 待机扫描", "entry": None, "tp": None, "sl": None, "color": "#888", "msg": "当前能效不足 80%，建议观望"}

    # --- [模式 A] 5M 趋势进攻 (截图 V220 验证) ---
    if is_trend:
        if c1['c'] > c1['ema20'] and net_flow > 0.5:
            plan = {"action": "🚀 趋势追多", "entry": c1['c'], "tp": c1['c'] + 3.1*atr, "sl": c1['c'] - 1.1*atr, "color": "#00FFCC", "msg": "5M 强趋势共振，资金点火，建议 10% 仓位"}
        elif c1['c'] < c1['ema20'] and net_flow < -0.5:
            plan = {"action": "🌊 趋势截空", "entry": c1['c'], "tp": c1['c'] - 3.1*atr, "sl": c1['c'] + 1.1*atr, "color": "#FF4B4B", "msg": "5M 下行通道开启，恐慌盘释放，建议 10% 仓位"}

    # --- [模式 B] 极限高频反转 (狙击模式) ---
    elif rsi < 28 and c1['c'] < c1['lower']:
        plan = {"action": "🏹 极限抄底", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] - 0.7*atr, "color": "#00FFCC", "msg": "超卖击穿布林下轨，博取 V 型反弹，建议 5% 仓位"}
    elif rsi > 72 and c1['c'] > c1['upper']:
        plan = {"action": "🎯 高频抛空", "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] + 0.7*atr, "color": "#FF4B4B", "msg": "超买触碰布林上轨，博取回踩均线，建议 5% 仓位"}

    return plan, is_trend, v_ratio, atr, rsi

# ==================== 3. 实时可视化渲染 ====================
raw1, raw5 = fetch_okx_v5("market/candles", "&bar=1m&limit=100"), fetch_okx_v5("market/candles", "&bar=5m&limit=100")
trades = fetch_okx_v5("market/trades", "&limit=100")

if raw1 and raw5 and trades:
    df1 = pd.DataFrame(raw1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(raw5, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(trades)
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    plan, it_mode, vr, av, rv = ultimate_engine_v230(df1, df5, net_f, buy_r)

    st.title("🛡️ ETH 战神 V230 最终裁决终端")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("净流 (1m)", f"{net_f:+.2f} ETH", delta=f"{buy_r:.1f}%")
    c2.metric("模式识别", "🌊 趋势进攻" if it_mode else "⚖️ 震荡高频")
    c3.metric("RSI (1m)", f"{rv:.1f}")
    c4.metric("量比 (1m)", f"{vr:.2f}x")

    st.markdown("---")
    l_box, r_box = st.columns([1, 2.3])

    with l_box:
        st.markdown(f"""<div style="border:4px solid {plan['color']}; padding:20px; border-radius:20px; background:rgba(0,0,0,0.4)">
            <h1 style="color:{plan['color']}; margin:0">{plan['action']}</h1><hr>
            <p style="font-size:20px">📍 <b>入场参考:</b> {f'${plan["entry"]:.2f}' if plan['entry'] else '等待信号'}</p>
            <p style="font-size:20px; color:#00FFCC">💰 <b>止盈目标:</b> {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
            <p style="font-size:20px; color:#FF4B4B">❌ <b>风险止损:</b> {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
            <hr><p style="color:#eee; font-size:16px">💡 <b>裁决建议:</b> {plan['msg']}</p></div>""", unsafe_allow_html=True)
    
    with r_box:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1M K")])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1.5), name="EMA20"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='gray', width=1, dash='dot'), name="BB_Up"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='gray', width=1, dash='dot'), name="BB_Down"))
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"🚀 V230 零度裁决中 | 波动率: {av:.2f} | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.error("数据链路建立失败，请检查网络环境或 API 访问频率。")
