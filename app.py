import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 指挥部初始化
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32001 指挥官", page_icon="⚖️")

def init_commander_state():
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0

# ==========================================
# 2. 增强型情报引擎 (修复 OKX 字段名)
# ==========================================
@st.cache_data(ttl=10)
def get_market_intel(f_ema, s_ema):
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res = requests.get(url, timeout=5).json()
        # 修正字段名匹配 OKX 实际返回
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        # EMA 与指标计算
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # 多空比同步
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        ls_res = requests.get(ls_url).json()
        if ls_res.get('code') == '0': st.session_state.ls_ratio = float(ls_res['data'][0][1])
        return df
    except: return pd.DataFrame()

# ==========================================
# 3. 终极边栏：多因子胜率算法
# ==========================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("### 🛸 量子实时控制")
        hb = st.slider("心跳频率 (秒)", 5, 60, 10)
        f_e = st.number_input("快线 EMA", 5, 30, 12)
        s_e = st.number_input("慢线 EMA", 20, 100, 26)
        
        # 升级版胜率算法 (EMA共振 + 净流 + 多空)
        prob = 50.0
        if df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1]: prob += 15
        if st.session_state.ls_ratio > 1.05: prob += 10
        if df['net_flow'].iloc[-1] > 0: prob += 5
        prob = max(min(prob, 88.0), 25.0) # 限制极值
        
        box_color = "#00ff88" if prob > 60 else "#FFD700" if prob > 45 else "#ff4b4b"
        st.markdown(f"""
            <div style="border:2px solid {box_color}; padding:15px; border-radius:12px; background:rgba(255,255,255,0.06); text-align:center;">
                <div style="color:{box_color}; font-size:0.85em; font-weight:bold;">AI 实时胜率</div>
                <div style="color:{box_color}; font-size:2.4em; font-weight:bold;">{prob:.1f}%</div>
                <div style="color:#aaa; font-size:0.75em;">建议 R/R: 1 : 1.85</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        # 统一多空逻辑
        sentiment = "看多 🔥" if st.session_state.ls_ratio > 1.0 else "看空 ❄️"
        st.info(f"🔍 AI 复盘：{'✅ 趋势量价匹配' if prob > 50 else '📉 警惕缩量虚拉'}\n\n散户情绪: {sentiment}")
        return hb, f_e, s_e

# ==========================================
# 4. 主大屏渲染 (补全 EMA 线)
# ==========================================
def main():
    init_commander_state()
    df_init = get_market_intel(12, 26)
    if df_init.empty: st.error("卫星连接断开"); return
    
    hb, f_e, s_e = render_sidebar(df_init)
    df = get_market_intel(f_e, s_e)
    
    # 顶部仪表盘
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32001) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("价格", f"${df['c'].iloc[-1]:.2f}")
    m2.metric("全网多空比", f"{st.session_state.ls_ratio:.2f}", "多头占优" if st.session_state.ls_ratio > 1 else "空头占优")
    m3.metric("ATR 波幅", f"{df['atr'].iloc[-1]:.2f}")
    m4.metric("庄家净流", f"{df['net_flow'].iloc[-1]:.0f}")

    # 图表绘制
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    
    # 【修复点】添加 EMA 快慢线可视化
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_f'], line=dict(color='#00ff88', width=2), name="EMA 12 (快)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_s'], line=dict(color='#ff4b4b', width=2), name="EMA 26 (慢)"), row=1, col=1)
    
    # 燃料区绘制
    h_max, l_min = df['h'].tail(60).max(), df['l'].tail(60).min()
    fig.add_hrect(y0=h_max*1.018, y1=h_max*1.02, fillcolor="red", opacity=0.3, annotation_text="空头燃料", row=1, col=1)
    fig.add_hrect(y0=l_min*0.98, y1=l_min*0.982, fillcolor="green", opacity=0.3, annotation_text="多头燃料", row=1, col=1)

    # 净流柱状图
    colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="资金流"), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=820, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    time.sleep(hb); st.rerun()

if __name__ == "__main__":
    main()
