import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from datetime import datetime

# ==================== 1. 核心配置与实战引擎 ====================
OKX_CONFIG = {
    "api_key": "a2a2a452-49e6-4e76-95f3-fb54eb982e7b",
    "secret_key": "330FABB2CAD3585677716686C2BF3872",
    "passphrase": "123321aA@", 
}

def fetch_okx_data(bar='5m', limit=100):
    url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar={bar}&limit={limit}"
    try:
        with requests.Session() as s:
            s.trust_env = True # 穿透 LetsVPN
            r = s.get(url, timeout=5)
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
            df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
            return df[::-1].reset_index(drop=True)
    except: return pd.DataFrame()

# ==================== 2. 核心算法逻辑 ====================
def analyze_logic(df):
    # EMA 与 MACD 计算
    df['ema21'] = df['c'].ewm(span=21, adjust=False).mean()
    exp1 = df['c'].ewm(span=12, adjust=False).mean()
    exp2 = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['hist'] = df['macd'] - df['sig']
    
    # 底背离检测
    df['p_min'] = df.iloc[argrelextrema(df['l'].values, np.less_indicator, order=5)[0]]['l']
    df['m_min'] = df.iloc[argrelextrema(df['hist'].values, np.less_indicator, order=5)[0]]['hist']
    pts = df.dropna(subset=['p_min']).tail(2)
    mds = df.dropna(subset=['m_min']).tail(2)
    div = False
    if len(pts)==2 and len(mds)==2:
        if pts.iloc[1]['l'] < pts.iloc[0]['l'] and mds.iloc[1]['hist'] > mds.iloc[0]['hist']:
            div = True
            
    # 趋势灯状态
    curr_c = df.iloc[-1]['c']
    curr_ema = df.iloc[-1]['ema21']
    status = "🟢" if curr_c > curr_ema else "🔴"
    
    return df, status, div

# ==================== 3. 页面渲染 (全能面板) ====================
st.set_page_config(page_title="ETH V34.0 全能指挥部", layout="wide")

# 执行数据抓取
df_5m_raw = fetch_okx_data('5m')
df_15m_raw = fetch_okx_data('15m')
df_1h_raw = fetch_okx_data('1h')

if not df_5m_raw.empty:
    df_5m, s5, div5 = analyze_logic(df_5m_raw)
    _, s15, _ = analyze_logic(df_15m_raw)
    _, s1h, _ = analyze_logic(df_1h_raw)
    
    curr = df_5m.iloc[-1]
    vol_avg = df_5m['v'].rolling(20).mean().iloc[-1]
    
    # 侧边栏
    st.sidebar.title("🎮 参数调节")
    yindie_limit = st.sidebar.slider("阴跌风险禁入线", 50, 100, 76)
    
    st.title("🛡️ ETH V34.0 全能报警面板")

    # --- 第一行：核心指标与报警面板 ---
    c1, c2, c3, c4 = st.columns([1, 1, 1.5, 1])
    
    with c1:
        st.metric("ETH 实时价", f"${curr['c']:.2f}", f"{curr['c']-df_5m.iloc[-2]['c']:.2f}")
    
    with c2:
        # 阴跌评分计算 (V15.3 增强公式)
        risk = 98.9 if s5 == "🔴" and curr['v'] < vol_avg * 0.75 else 12.5
        st.metric("阴跌风险评分", f"{risk}", "高危" if risk > yindie_limit else "安全", delta_color="inverse")

    with c3:
        # 【全能面板核心】：共振灯 + 背离监测 横向集成
        st.write("🚦 综合态势监控 (5m/15m/1h | 信号)")
        status_html = f"""
        <div style="background:#161b22; padding:10px; border-radius:10px; border:1px solid #30363d; display:flex; justify-content:space-around; align-items:center;">
            <div style="text-align:center;"><span style="font-size:20px;">{s5}</span><br><small>5m</small></div>
            <div style="text-align:center;"><span style="font-size:20px;">{s15}</span><br><small>15m</small></div>
            <div style="text-align:center;"><span style="font-size:20px;">{s1h}</span><br><small>1h</small></div>
            <div style="border-left:1px solid #30363d; height:30px; margin:0 10px;"></div>
            <div style="text-align:center;">
                <span style="font-size:18px; color:{'#00ffcc' if div5 else '#888'}; font-weight:bold;">
                    {'💎 底背离' if div5 else '⚪ 无背离'}
                </span>
            </div>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)

    with c4:
        st.metric("量能比", f"{curr['v']/vol_avg:.2f}x", "缩量" if curr['v'] < vol_avg else "活跃")

    # --- 第二行：实时预警提示 ---
    if s5 == "🔴" and s15 == "🔴" and s1h == "🔴":
        st.error("🚨 警告：三周期全红！大势已去，阴跌风险极高，严禁任何抄底行为！")
    elif div5:
        st.success("🎯 发现 5m 级别 MACD 底背离！如果 15m 灯转绿，可能是极佳的入场反弹点。")

    # --- 第三行：K 线主图 ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_5m.index, open=df_5m['o'], high=df_5m['h'], low=df_5m['l'], close=df_5m['c'], name="5m K线"))
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema21'], line=dict(color='cyan', width=1.5), name="EMA21压力线"))
    
    # 标记局部低点
    low_pts = df_5m.dropna(subset=['p_min'])
    fig.add_trace(go.Scatter(x=low_pts.index, y=low_pts['l'], mode='markers', marker=dict(color='yellow', size=10, symbol='triangle-up'), name="波谷支撑"))
    
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=5,r=5,t=5,b=5))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"🚀 Sniper V34.0 | 节点: {OKX_CONFIG['api_key'][:8]}... | 更新: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.warning("🔄 正在穿透 LetsVPN 隧道... 请确保 VPN 软件处于‘已连接’状态。")
