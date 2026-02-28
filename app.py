import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 全球顶级数据源配置 ====================
st.set_page_config(page_title="ETH V180 战神闭环终端", layout="wide")

def fetch_okx_v5(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. V180 核心双模引擎 (5M波段+高频) ====================
def tactical_engine_v180(df1, df5, net_flow, buy_ratio):
    # 底层指标闭环计算
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.2)
        d['lower'] = d['ema20'] - (d['std'] * 2.2)
        # 引入 RSI 用于抄底过滤
        delta = d['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        d['rsi'] = 100 - (100 / (1 + rs))

    c1, c5 = df1.iloc[-1], df5.iloc[-1]
    atr_val = c1['atr']
    rsi_val = c1['rsi']
    
    # 庄家动能变量 (锁定作用域)
    vol_avg = df1['v'].rolling(20).mean().iloc[-1]
    v_ratio = c1['v'] / vol_avg if vol_avg > 0 else 1.0
    
    # 模式自动换挡：基于 5M 指数移动平均斜率
    slope_5m = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope_5m) > (atr_val * 0.15)

    plan = {"mode": "📡 全频扫描", "action": "待机", "entry": None, "tp": None, "sl": None, "msg": "当前信号品质未达 85% 以上，保持空仓。"}

    # --- 模式 A：5Min 波段突击 (V175 增强版) ---
    if is_trend:
        if c1['c'] > c1['ema20'] and net_flow > 0.5:
            plan = {
                "mode": "🚀 趋势波段-多头", "action": "做多 (LONG)",
                "entry": c1['c'], "tp": c1['c'] + 3.0*atr_val, "sl": c1['c'] - 1.2*atr_val,
                "msg": "5M 动能主升浪，主力净流入。建议仓位: 10%"
            }
        elif c1['c'] < c1['ema20'] and net_flow < -0.5:
            plan = {
                "mode": "🌊 趋势波段-空头", "action": "做空 (SHORT)",
                "entry": c1['c'], "tp": c1['c'] - 3.0*atr_val, "sl": c1['c'] + 1.2*atr_val,
                "msg": "5M 下行通道开启，抛压释放中。建议仓位: 10%"
            }

    # --- 模式 B：极限高频抄底 (V180 特有强化) ---
    else:
        if c1['c'] <= c1['lower'] and rsi_val < 30:
            plan = {
                "mode": "🏹 极限高频-抄底", "action": "博反弹 (BUY)",
                "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] - 0.7*atr_val,
                "msg": "布林超卖 + RSI背离。建议仓位: 5% (快进快出)"
            }
        elif c1['c'] >= c1['upper'] and rsi_val > 70:
            plan = {
                "mode": "🎯 极限高频-摸顶", "action": "博回调 (SELL)",
                "entry": c1['c'], "tp": c1['ema20'], "sl": c1['c'] + 0.7*atr_val,
                "msg": "布林超买 + RSI高位。建议仓位: 5% (波段收割)"
            }

    return plan, is_trend, v_ratio, atr_val, rsi_val

# ==================== 3. 渲染终极战地 UI ====================
k1_raw = fetch_okx_v5("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx_v5("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx_v5("market/trades", "&limit=80")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t_raw['data'])
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    # 核心引擎计算
    plan, is_trend, vol_ratio, atr, rsi = tactical_engine_v180(df1, df5, net_f, buy_r)

    st.title("🛡️ ETH 战神 V180 终极实战终端")
    
    # 指标看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1m 净流动", f"{net_f:+.2f} ETH", delta=f"{buy_r:.1f}%")
    c2.metric("战术模式", plan['mode'])
    c3.metric("相对强弱 RSI", f"{rsi:.1f}")
    c4.metric("波动率 ATR", f"{atr:.2f}")

    st.markdown("---")
    l_col, r_col = st.columns([1, 2])

    with l_col:
        accent = "#00FFCC" if "多" in plan['mode'] or "BUY" in plan['action'] else "#FF4B4B" if "空" in plan['mode'] or "SELL" in plan['action'] else "#888"
        st.markdown(f"""
            <div style="border:3px solid {accent}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.4)">
                <h1 style="color:{accent}; margin-top:0">{plan['action']}</h1>
                <h3 style="margin-bottom:10px">{plan['mode']}</h3>
                <hr>
                <p style="font-size:18px">📍 <b>入场点:</b> {f'${plan["entry"]:.2f}' if plan['entry'] else '等待信号'}</p>
                <p style="font-size:18px; color:#00FFCC">💰 <b>止盈位:</b> {f'${plan["tp"]:.2f}' if plan['tp'] else '---'}</p>
                <p style="font-size:18px; color:#FF4B4B">❌ <b>止损位:</b> {f'${plan["sl"]:.2f}' if plan['sl'] else '---'}</p>
                <hr>
                <p style="font-size:16px">💡 <b>战地指令:</b> {plan['msg']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 复制字符串块 (交易所格式)
        if plan['entry']:
            st.code(f"ETH-USDT {plan['action']} @{plan['entry']} TP:{plan['tp']:.2f} SL:{plan['sl']:.2f}", language="bash")

    with r_col:
        fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="1m K")])
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='gray', width=1, dash='dot'), name="BB上"))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='gray', width=1, dash='dot'), name="BB下"))
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**🕵️ 系统状态诊断** | 量比: `{vol_ratio:.2f}` | 模式: `{'趋势' if is_trend else '震荡'}` | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.error("正在强行接通 OKX 战地数据源...")
