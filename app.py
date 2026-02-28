import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 全局配置与数据流 ====================
st.set_page_config(page_title="ETH V88 终极整合版", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 核心算法库 (整合 V14-V85) ====================
def ultimate_engine_v88(df1, df5, net_flow, buy_ratio):
    # --- 指标矩阵 ---
    for d in [df1, df5]:
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'], d['lower'] = d['ema20'] + d['std']*2, d['ema20'] - d['std']*2
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    
    # A. 模式识别 (来自 V74)
    slope5 = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope5) > (df5['atr'].iloc[-1] * 0.15)
    mode = "🌊 趋势模式" if is_trend else "⚖️ 震荡模式"

    # B. 动能辨伪 (来自 V80)
    vol_mean = df1['v'].rolling(20).mean().iloc[-1]
    is_real = (curr1['v'] / vol_mean > 1.1) or (abs(curr1['c']-curr1['o'])/curr1['atr'] < 1.5)
    
    # C. 资金评分系统 (整合 V72/V75)
    is_spoofing = buy_ratio > 90 and net_flow < 2.0
    f_score = 0
    if net_flow > 5 and not is_spoofing: f_score += 4
    if net_flow < -5: f_score -= 4
    if buy_ratio > 60: f_score += 1
    if buy_ratio < 40: f_score -= 1

    # D. 决策逻辑 (终极共振)
    sig = {"type": None, "dir": None, "plan": None}
    
    # 趋势模式下看 EMA 顺势
    if is_trend:
        if curr1['c'] > curr1['ema20'] and f_score >= 4 and is_real:
            sig = {"type": "趋势点火", "dir": "LONG"}
        elif curr1['c'] < curr1['ema20'] and f_score <= -4 and is_real:
            sig = {"type": "趋势下破", "dir": "SHORT"}
    # 震荡模式下看 布林轨道反转
    else:
        if curr1['c'] <= curr1['lower'] and f_score >= 2:
            sig = {"type": "震荡低吸", "dir": "LONG"}
        elif curr1['c'] >= curr1['upper'] and f_score <= -2:
            sig = {"type": "震荡高抛", "dir": "SHORT"}

    return sig, mode, f_score, is_real, is_spoofing

# ==================== 3. 渲染层 (整合 V14/V65/V81 UI) ====================
k1 = fetch_okx("market/candles", "&bar=1m&limit=100")
k5 = fetch_okx("market/candles", "&bar=5m&limit=100")
t_data = fetch_okx("market/trades", "&limit=50")

if k1 and k5 and t_data:
    df1 = pd.DataFrame(k1['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df1[['o','h','l','c','v']] = df1[['o','h','l','c','v']].astype(float)
    df1 = df1[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df5['c'] = df5['c'].astype(float)
    
    tdf = pd.DataFrame(t_data['data'])
    buy_v, sell_v = tdf[tdf['side']=='buy']['sz'].astype(float).sum(), tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_f, buy_r = buy_v - sell_v, (buy_v/(buy_v+sell_v)*100) if (buy_v+sell_v)>0 else 50
    
    sig, mode, f_score, is_real, is_spoofing = ultimate_engine_v88(df1, df5, net_f, buy_r)
    curr_p, atr = df1.iloc[-1]['c'], df1.iloc[-1]['atr']

    # --- UI 侧边栏 (整合战术指挥) ---
    with st.sidebar:
        st.header("🎛️ V88 终极指挥部")
        st.metric("核心模式", mode)
        st.write(f"动能验证: {'✅ 真实' if is_real else '❌ 虚假'}")
        
        if sig['type']:
            color = "#00FFCC" if sig['dir'] == "LONG" else "#FF4B4B"
            tp1 = curr_p + 1.8*atr if sig['dir']=="LONG" else curr_p - 1.8*atr
            st.markdown(f"""
                <div style="border:2px solid {color}; padding:15px; border-radius:10px; background:rgba(0,0,0,0.3)">
                    <h2 style="color:{color}">{sig['type']}</h2>
                    <p><b>策略：</b>{mode}下的资金共振</p>
                    <hr>
                    <p>📍 入场：${curr_p:.2f}</p>
                    <p>💰 止盈1：${tp1:.2f}</p>
                    <p>🚨 提示：达到止盈1后移动止损至保本位。</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔎 因子扫描中... 当前盘面不符合任何整合策略。")
            if is_spoofing: st.warning("🚨 正在拦截虚假对敲诱多")

    # --- UI 主界面 ---
    st.title("🛡️ ETH 全功能整合终端 V88")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("现价", f"${curr_p}")
    c2.metric("1min 净流", f"{net_f:+.2f} ETH")
    c3.metric("实时买压", f"{buy_r:.1f}%")
    c4.metric("资金动能", f_score)

    fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'])])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1.5), name="1m EMA20"))
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
