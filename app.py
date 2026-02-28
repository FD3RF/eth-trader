import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V85 全自动战术终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. V85 决策与执行引擎 ====================
def strategy_engine_v85(df, net_flow, buy_ratio):
    # --- 核心指标 ---
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['atr'] = (df['h'] - df['l']).rolling(14).mean()
    curr = df.iloc[-1]
    atr = curr['atr']
    vol_mean = df['v'].rolling(20).mean().iloc[-1]
    
    # --- 辨伪识别 ---
    price_move = abs(curr['c'] - curr['o']) / atr
    vol_surge = curr['v'] / vol_mean
    is_real = vol_surge > 1.1 or price_move < 1.5
    is_spoofing = buy_ratio > 88 and net_flow < 1.5  # 对敲识别
    
    # --- 资金权重 ---
    f_score = 0
    if net_flow > 5 and not is_spoofing: f_score += 4
    if net_flow < -5: f_score -= 4
    if buy_ratio > 60: f_score += 1
    if buy_ratio < 40: f_score -= 1

    # --- 信号决策 ---
    sig = {"type": None, "dir": None, "score": 0, "warn": ""}
    if is_real and abs(f_score) >= 4:
        if curr['c'] > df['ema20'].iloc[-1] and f_score > 0:
            sig = {"type": "战略做多", "dir": "LONG", "score": 6 + f_score}
        elif curr['c'] < df['ema20'].iloc[-1] and f_score < 0:
            sig = {"type": "战略做空", "dir": "SHORT", "score": 6 + abs(f_score)}
    
    if is_spoofing: sig["warn"] = "🚨 捕捉到庄家对敲，切勿盲目跟单！"
    elif not is_real and price_move > 2.0: sig["warn"] = "⚠️ 无量虚涨，谨防冲高回落！"
    
    return sig, atr, f_score, is_real

# ==================== 3. 渲染层 ====================
k_data = fetch_okx_data("market/candles", "&bar=1m&limit=100")
t_data = fetch_okx_data("market/trades", "&limit=50")

if k_data and t_data:
    df = pd.DataFrame(k_data['data'], columns=['ts','o', 'h', 'l', 'c', 'v', 'volCcy', 'volCcyQ', 'confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    
    tdf = pd.DataFrame(t_data['data'])
    buy_v = tdf[tdf['side']=='buy']['sz'].astype(float).sum()
    sell_v = tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_f = buy_v - sell_v
    buy_r = (buy_v / (buy_v + sell_v)) * 100 if (buy_v + sell_v) > 0 else 50
    
    sig, atr, f_score, is_real = strategy_engine_v85(df, net_f, buy_r)
    curr_p = df.iloc[-1]['c']

    with st.sidebar:
        st.header("🎯 V85 战术指挥中心")
        if sig['warn']: st.warning(sig['warn'])
        
        if sig['type']:
            color = "#00FFCC" if sig['dir'] == "LONG" else "#FF4B4B"
            # 战术计划书
            sl = curr_p - 1.5*atr if sig['dir']=="LONG" else curr_p + 1.5*atr
            tp1 = curr_p + 1.8*atr if sig['dir']=="LONG" else curr_p - 1.8*atr
            tp2 = curr_p + 3.5*atr if sig['dir']=="LONG" else curr_p - 3.5*atr
            
            st.markdown(f"""
                <div style="background:rgba(0,255,204,0.1); padding:15px; border-radius:10px; border:2px solid {color}">
                    <h2 style="color:{color}; margin:0">🚀 {sig['type']}</h2>
                    <p style="font-size:14px">评分: {sig['score']} | 品质: {'真实' if is_real else '待定'}</p>
                    <hr>
                    <p><b>📍 入场：${curr_p:.2f}</b></p>
                    <p><b>❌ 止损：${sl:.2f}</b> (1.5倍ATR)</p>
                    <p><b>💰 止盈1：${tp1:.2f}</b> (减仓50%)</p>
                    <p><b>💎 止盈2：${tp2:.2f}</b> (博取大趋势)</p>
                    <p style="color:#FFA500">🛡️ 策略：一旦TP1达成，请将剩余止损移至保本位。</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔭 1min 高频过滤中，当前未达到开仓标准。")

    st.title("🛡️ ETH 战术自动终端 V85")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1min 净流", f"{net_f:+.2f} ETH")
    c2.metric("实时买压", f"{buy_r:.1f}%")
    c3.metric("ATR 波动", f"{atr:.2f}")
    c4.metric("资金动能", f_score)

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'])])
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
    fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
