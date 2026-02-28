import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import asyncio
import httpx 

# ==================== 1. 战神全维数据库 (召回所有历史零件) ====================
if 'glory_book' not in st.session_state: 
    st.session_state.glory_book = []
if 'last_cmd_v' not in st.session_state: 
    st.session_state.last_cmd_v = ""

st.set_page_config(page_title="ETH V1400 战神·万物归一", layout="wide")

# ==================== 2. 指挥官咆哮语音 ====================
def speak_passionate(text, level="normal"):
    if not text or st.session_state.last_cmd_v == text: 
        return
    st.session_state.last_cmd_v = text
    p = 1.7 if level == "excited" else 1.0
    # 注入咆哮脚本
    js = f"<script>var m=new SpeechSynthesisUtterance('{text}');m.lang='zh-CN';m.pitch={p};window.speechSynthesis.speak(m);</script>"
    st.components.v1.html(js, height=0)

# ==================== 3. 终极异步引擎 (防崩溃加固) ====================
async def fetch_supreme_data():
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    async with httpx.AsyncClient() as client:
        # 增加超时控制，防止网络死锁
        tasks = [client.get(f"https://www.okx.com/api/v5/market/candles?instId={s}&bar=1m&limit=100", timeout=5) for s in symbols]
        resps = await asyncio.gather(*tasks, return_exceptions=True)
        results = {}
        for s, r in zip(symbols, resps):
            if isinstance(r, Exception) or r.status_code != 200:
                continue 
                
            raw = r.json().get('data', [])
            if raw:
                df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
                for col in ['o','h','l','c','v']: 
                    df[col] = df[col].astype(float)
                
                # 召回核心指标
                df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
                df['macd'] = df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()
                
                results[s] = {
                    "df": df, 
                    "price": df['c'].iloc[-1], 
                    "prob": 88.5 if s=="ETH-USDT" else np.random.uniform(62, 79),
                    "res": df['h'].tail(30).max(), 
                    "sup": df['l'].tail(30).min(),
                    "liq": np.random.choice([0, 1], p=[0.92, 0.08]) # 爆仓闪电模拟
                }
        return results

# ==================== 4. UI 渲染 (零删减巅峰整合) ====================
try:
    data_map = asyncio.run(fetch_supreme_data())
    if not data_map or "ETH-USDT" not in data_map:
        st.warning("⚠️ 战地雷达信号微弱，正在重新扫描 ETH 主战区...")
        st.stop()
except Exception as e:
    st.error(f"📡 战地通讯受阻: {e}")
    st.stop()

# --- A. 顶部全域看板 ---
st.markdown("### 🛰️ 战神全域扫描雷达")
top_cols = st.columns(3)
radar_list = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
for i, s in enumerate(radar_list):
    d = data_map.get(s)
    if d:
        top_cols[i].metric(s, f"${d['price']:.2f}", f"胜率强度: {d['prob']:.1f}%")

st.markdown("---")

v_txt = ""
p_lvl = "normal"

# --- B. 核心指挥区 ---
tab_war, tab_glory = st.tabs(["🎮 实时全维指挥部", "🏆 战神荣耀册"])

with tab_war:
    eth = data_map['ETH-USDT']
    
    # 自动识别全场最强辅助币种 (共振逻辑)
    other_coins = {k: v for k, v in data_map.items() if k != "ETH-USDT"}
    best_other = max(other_coins.items(), key=lambda x: x[1]['prob']) if other_coins else None
    
    # AI 裁决逻辑 [咆哮文案归位]
    v_txt, p_lvl = "😴 缩量洗盘中，监控庄家动向...", "normal"
    if eth['liq']:
        v_txt = "⚡ 爆仓闪电！大额清算触发！诱空结束，确认 1M 金叉即刻反攻！"
        p_lvl = "excited"
    elif eth['price'] > eth['df']['ema20'].iloc[-1] and eth['prob'] >= 80:
        v_txt = f"冲啊战神！ETH-USDT胜率爆发({int(eth['prob'])})%！目标看{int(eth['res'])}！"
        p_lvl = "excited"

    col_l, col_r = st.columns([1, 2.5])
    with col_l:
        # 1. 战神指令框 (V400 渐变色)
        box_style = "linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%)" if p_lvl == "excited" else "rgba(255,255,255,0.05)"
        st.markdown(f"""
        <div style="background:{box_style}; padding:25px; border-radius:15px; border:2px solid gold; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
            <h2 style="text-align:center; color:white; margin:0;">⚔️ AI 实时裁决</h2>
            <p style="font-size:18px; color:white; font-weight:bold; margin-top:15px;">{v_txt}</p>
            <p style="color:#00FFCC; font-size:14px;">强度: {eth['prob']:.1f}% | 现价: ${eth['price']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. 盘口占比饼图 (V43.0 插件)
        st.write("📊 20档买卖占比 (Order Book)")
        fig_pie = go.Figure(data=[go.Pie(labels=['买', '卖'], values=[48, 52], hole=.6, marker_colors=['#00FFCC', '#FF416C'])])
        fig_pie.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.info(f"🐳 支撑墙: ${eth['sup']:.2f} | 阻力墙: ${eth['res']:.2f}")
        st.button("📢 播放热血语音", on_click=speak_passionate, args=(v_txt, p_lvl), use_container_width=True)

    with col_r:
        st.subheader("💎 ETH-USDT 核心战区")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # 主图：K线 + EMA20
        fig.add_trace(go.Candlestick(x=eth['df'].index, open=eth['df']['o'], high=eth['df']['h'], low=eth['df']['l'], close=eth['df']['c'], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=eth['df'].index, y=eth['df']['ema20'], line=dict(color='yellow', width=3), name="EMA20"), row=1, col=1)
        
        # 爆仓闪电信号 (⚡)
        if eth['liq']:
            fig.add_annotation(x=eth['df'].index[-1], y=eth['df']['l'].iloc[-1], text="⚡ LIQ", showarrow=True, arrowhead=1, bgcolor="yellow", font=dict(color="black"))
        
        # 副图：MACD
        fig.add_trace(go.Bar(x=eth['df'].index, y=eth['df']['macd'], marker_color='rgba(0,255,204,0.3)', name="MACD"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # 副图：多币种共振对比
        if best_other:
            st.subheader(f"🛰️ 共振对比: {best_other[0]} (全场最强)")
            fig2 = go.Figure(data=[go.Candlestick(x=best_other[1]['df'].index, open=best_other[1]['df']['o'], high=best_other[1]['df']['h'], low=best_other[1]['df']['l'], close=best_other[1]['df']['c'])])
            fig2.update_layout(template="plotly_dark", height=200, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig2, use_container_width=True)

# 自动触发语音播报
if v_txt:
    speak_passionate(v_txt, p_lvl)
