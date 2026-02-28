import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V100 天眼全知版", layout="wide")

def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. V100 天眼算法引擎 ====================
def omniscient_engine_v100(df1, df5, net_flow, buy_ratio):
    # 数据清洗与指标计算
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'], d['lower'] = d['ema20'] + d['std']*2, d['ema20'] - d['std']*2

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    atr = curr1['atr']
    
    # 【变盘诊断逻辑】
    rsi = 100 - (100 / (1 + (df1['c'].diff().dropna().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / 
                             df1['c'].diff().dropna().apply(lambda x: -x if x < 0 else 0).rolling(14).mean())))
    
    # 【辨伪与模式】
    slope5 = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope5) > (atr * 0.1)
    vol_quality = curr1['v'] / df1['v'].rolling(20).mean().iloc[-1]
    is_real = vol_quality > 1.1 or abs(curr1['c'] - curr1['o'])/atr < 1.2
    is_spoofing = buy_ratio > 92 and net_flow < 1.0

    # 【综合评分】
    f_score = 0
    if net_flow > 3: f_score += 2
    if buy_ratio > 55: f_score += 1
    if net_flow < -3: f_score -= 2
    if buy_ratio < 45: f_score -= 1
    if is_spoofing: f_score = 0

    # 【执行计划】
    sig = {"type": None, "dir": None, "desc": ""}
    if is_trend and is_real:
        if curr1['c'] > curr1['ema20'] and f_score >= 3:
            sig = {"type": "⚡ 趋势点火", "dir": "LONG", "desc": "5M趋势共振，资金真实流入"}
        elif curr1['c'] < curr1['ema20'] and f_score <= -3:
            sig = {"type": "🔥 趋势崩塌", "dir": "SHORT", "desc": "5M破位，放量砸盘中"}
    elif not is_trend:
        if curr1['c'] <= curr1['lower'] and f_score >= 1:
            sig = {"type": "🏹 震荡低吸", "dir": "LONG", "desc": "触碰布林下轨，主力护盘"}
        elif curr1['c'] >= curr1['upper'] and f_score <= -1:
            sig = {"type": "🎯 震荡高抛", "dir": "SHORT", "desc": "触碰布林上轨，上方抛压重"}

    return sig, is_trend, f_score, is_real, rsi.iloc[-1], atr

# ==================== 3. 渲染层 (全功能大整合) ====================
k1_raw = fetch_okx("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx("market/trades", "&limit=50")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volCcy', 'volCcyQ','confirm'])[::-1].reset_index(drop=True)
    
    tdf = pd.DataFrame(t_raw['data'])
    buy_v = tdf[tdf['side']=='buy']['sz'].astype(float).sum()
    sell_v = tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    net_f, buy_r = buy_v - sell_v, (buy_v/(buy_v+sell_v)*100) if (buy_v+sell_v)>0 else 50
    
    sig, is_trend, f_score, is_real, rsi, atr = omniscient_engine_v100(df1, df5, net_f, buy_r)
    curr_p = float(df1.iloc[-1]['c'])

    # --- 侧边栏：战术指挥与情绪雷达 ---
    with st.sidebar:
        st.header("🛸 天眼指挥部 V100")
        st.markdown(f"**市场模式**: {'📈 趋势' if is_trend else '🚥 震荡'}")
        st.markdown(f"**情绪指标**: RSI {rsi:.1f} | {'🔥 超买' if rsi>70 else '❄️ 超卖' if rsi<30 else '⚖️ 中性'}")
        
        if sig['type']:
            color = "#00FFCC" if sig['dir'] == "LONG" else "#FF4B4B"
            tp1 = curr_p + 1.8*atr if sig['dir']=="LONG" else curr_p - 1.8*atr
            sl = curr_p - 1.5*atr if sig['dir']=="LONG" else curr_p + 1.5*atr
            st.markdown(f"""
                <div style="border:3px solid {color}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.5)">
                    <h2 style="color:{color}; margin:0">{sig['type']}</h2>
                    <p style="font-size:12px">{sig['desc']}</p>
                    <hr>
                    <p><b>入场:</b> ${curr_p:.2f}</p>
                    <p><b>止盈:</b> ${tp1:.2f}</p>
                    <p><b>止损:</b> ${sl:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔭 深度扫描中，当前动能不足，建议保持空仓。")
            if buy_r > 90: st.warning("🚨 警惕：买压极高但价格不动，疑似对敲诱多！")

    # --- 主屏：战神大屏 ---
    st.title("🛡️ ETH 天眼指挥终端 V100")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1min 净流", f"{net_f:+.2f} ETH")
    c2.metric("实时买压", f"{buy_r:.1f}%")
    c3.metric("资金动能", f_score)
    c4.metric("动能真实性", "PASS" if is_real else "FAIL")

    fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="K线")])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1.5), name="EMA20"))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="布林上"))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name="布林下"))
    
    fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("正在接入 OKX 实时全量数据流...")
