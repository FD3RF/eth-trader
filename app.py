import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V72.0 庄家雷达版", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 庄家雷达核心引擎 ====================
def whale_radar_engine(df, net_flow, buy_ratio):
    # --- 基础技术体系 ---
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['c'].ewm(span=60, adjust=False).mean()
    df['std'] = df['c'].rolling(20).std()
    df['upper'] = df['ema20'] + (df['std'] * 2)
    df['lower'] = df['ema20'] - (df['std'] * 2)
    
    # ATR 波动
    df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    atr = df['atr'].iloc[-1]
    
    # --- 资金雷达因子 (Whale Factors) ---
    flow_score = 0
    if net_flow > 5: flow_score = 2      # 主力试探入场
    if net_flow > 20: flow_score = 4     # 主力显著扫货
    if buy_ratio > 60: flow_score += 1   # 买盘占优
    
    # --- 信号决策 (结构 + 资金共振) ---
    curr = df.iloc[-1]
    vol_mean = df['v'].rolling(20).mean().iloc[-1]
    signal = {"type": None, "dir": None, "score": 0, "whale_alert": False}
    
    # 模式识别
    mode = "资金扫描中"
    if flow_score >= 4: mode = "🟢 主力强势建仓"
    elif flow_score <= -4: mode = "🔴 主力抛售洗盘"

    # 逻辑引擎
    # 1. 庄家顺势信号 (EMA回踩 + 资金共振)
    if curr['ema20'] > curr['ema60'] and curr['l'] <= curr['ema20']:
        score = 3 + flow_score
        if score >= 7:
            signal = {"type": "雷达锁定-多", "dir": "LONG", "score": score, "whale_alert": flow_score >= 4}
    
    elif curr['ema20'] < curr['ema60'] and curr['h'] >= curr['ema20']:
        score = 3 + abs(flow_score) if flow_score < 0 else 3
        if score >= 7:
            signal = {"type": "雷达锁定-空", "dir": "SHORT", "score": score, "whale_alert": flow_score <= -4}

    # 2. 突发放量突破
    if not signal['type'] and abs(flow_score) >= 5:
        if curr['c'] > df['upper'].iloc[-2]:
            signal = {"type": "主力破位-多", "dir": "LONG", "score": 9, "whale_alert": True}
        elif curr['c'] < df['lower'].iloc[-2]:
            signal = {"type": "主力破位-空", "dir": "SHORT", "score": 9, "whale_alert": True}

    return signal, atr, mode, flow_score

# ==================== 3. 终端渲染 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=300")
t_raw = fetch_okx_data("market/trades", "&limit=100")

if k_raw and t_raw:
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    
    # 实时资金流计算
    tdf = pd.DataFrame(t_raw['data'])
    buy_vol = tdf[tdf['side']=='buy']['sz'].astype(float).sum()
    sell_vol = tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_flow = buy_vol - sell_vol
    buy_ratio = (buy_vol / (buy_vol + sell_vol)) * 100
    
    sig, atr, mode, f_score = whale_radar_engine(df, net_flow, buy_ratio)
    curr_p = df.iloc[-1]['c']
    
    with st.sidebar:
        st.header("🛰️ V72.0 庄家雷达")
        st.subheader(mode)
        st.write(f"资金评分: {f_score} | 净流: {net_flow:.2f} ETH")
        
        if sig['type']:
            color = "#FFD700" if sig['whale_alert'] else "#00FFCC"
            st.markdown(f"""
                <div style="border:4px solid {color}; padding:15px; border-radius:12px; background:rgba(255,215,0,0.1)">
                    <h2 style="color:{color}; margin:0">🎯 {sig['type']}</h2>
                    <p><b>总评分: {sig['score']} | 共振确认</b></p>
                    <hr>
                    <p>📍 进场: ${curr_p:.2f}</p>
                    <p style="color:#FF4B4B">❌ 止损: ${curr_p - 1.2*atr if sig['dir']=='LONG' else curr_p + 1.2*atr:.2f}</p>
                    <p style="color:#00FFCC">💰 止盈: ${curr_p + 2.5*atr if sig['dir']=='LONG' else curr_p - 2.5*atr:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
            if sig['whale_alert']: st.success("检测到主力同步扫货，胜率加成中！")
        else:
            st.info("雷达扫描中... 暂无资金共振信号")

    st.title("🛡️ ETH 庄家雷达终端 V72.0")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1min 净流入", f"{net_flow:+.2f} ETH")
    c2.metric("实时买压", f"{buy_ratio:.1f}%")
    c3.metric("ATR 波动", f"{atr:.2f}")
    c4.metric("资金动能", f_score)

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线")])
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], line=dict(color='yellow', width=1), name="趋势线"))
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
