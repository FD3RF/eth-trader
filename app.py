import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 指挥部初始化与环境硬锁
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 决策指挥官", page_icon="⚖️")

def init_commander_state():
    """初始化持久化战术状态"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'sentiment_score' not in st.session_state: st.session_state.sentiment_score = 50.0
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup_ts' not in st.session_state: st.session_state.last_cleanup_ts = time.time()

init_commander_state()

# ==========================================
# 2. 核心算法：模式识别、阻力计算与 AI 裁决
# ==========================================
def auto_mode_selector(df):
    """根据过去1小时波动率自动识别交易模式"""
    if len(df) < 60: return "初始化", "⏳", "等待数据沉淀"
    recent_atr = df['atr'].tail(60).mean()
    price = df['c'].iloc[-1]
    vol_ratio = (recent_atr / price) * 100
    if vol_ratio > 0.15: return "趋势爆发模式", "🚀", "建议：放宽止盈，顺势而为"
    if vol_ratio < 0.05: return "极窄震荡模式", "🦥", "建议：持币观望，谨防洗盘"
    return "标准波段模式", "⚖️", "建议：高抛低吸，执行R/R计划"

def calculate_pivots(df):
    """自动计算最近30根K线的物理压力与支撑"""
    if df.empty: return 0, 0
    res = df['h'].tail(30).max()
    sup = df['l'].tail(30).min()
    return res, sup

def auto_memory_cleanup():
    """每4小时强制清理内存，防止长时间盯盘卡顿"""
    if time.time() - st.session_state.last_cleanup_ts > 14400:
        st.session_state.battle_logs = [f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存治理完成"]
        st.session_state.last_cleanup_ts = time.time()
        st.cache_data.clear()

# ==========================================
# 3. 数据层：多源实时情报引擎
# ==========================================
@st.cache_data(ttl=10)
def get_market_intel(fast_ema, slow_ema):
    """抓取K线、计算波动率与多空比"""
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        df['ema_f'] = df['c'].ewm(span=fast_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=slow_ema, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        ls_res = requests.get(ls_url, timeout=5).json()
        if ls_res.get('code') == '0': st.session_state.ls_ratio = float(ls_res['data'][0][1])
        return df
    except: return pd.DataFrame()

# ==========================================
# 4. 渲染层：指挥大屏显示
# ==========================================
def render_commander():
    auto_memory_cleanup()
    st.sidebar.markdown("### 🛸 量子实时控制")
    refresh_rate = st.sidebar.slider("心跳频率 (秒)", 5, 60, 10)
    f_ema = st.sidebar.number_input("快线 EMA", 5, 30, 12)
    s_ema = st.sidebar.number_input("慢线 EMA", 20, 100, 26)
    
    df = get_market_intel(f_ema, s_ema)
    if df.empty: st.error("卫星数据同步失败，检查网络连接"); return

    # 指标计算
    res_px, sup_px = calculate_pivots(df)
    m_name, m_icon, m_tips = auto_mode_selector(df)
    last_p = df['c'].iloc[-1]
    prob = 50.0 + (10 if df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1] else -5) + (15 if st.session_state.ls_ratio < 0.95 else -10) + (st.session_state.sentiment_score - 50) * 0.35
    prob = max(min(prob, 99.0), 1.0)
    rr = abs(res_px - last_p) / max(abs(last_p - sup_px), 0.1)

    # UI 渲染
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI 进攻胜率", f"{prob:.1f}%")
    c2.metric("建议盈亏比 (R/R)", f"1 : {rr:.2f}")
    c3.metric("全网多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    c4.metric("庄家情绪", f"{st.session_state.sentiment_score:.0f}")

    st.divider()

    col_l, col_r = st.columns([1, 4])
    with col_l:
        # --- AI 裁决生成器 ---
        decision_box_color = "#00ff00" if prob > 60 else "#FFD700" if prob > 45 else "#ff4b4b"
        st.markdown(f"""
            <div style="padding:15px; border:2px solid {decision_box_color}; border-radius:12px; background:rgba(0,0,0,0.4); margin-bottom:15px; box-shadow: 0 0 10px {decision_box_color}55;">
                <p style="color:{decision_box_color}; font-weight:bold; margin:0;">🤖 AI 裁决建议</p>
                <p style="font-size:1em; color:white; margin:10px 0;">
                    建议在 <span style="color:#00ff00;">${sup_px:.1f}</span> 附近轻仓做多，目标 <span style="color:#00ff00;">${res_px:.1f}</span>。<br>
                    判定：<b>{'胜率极高' if prob > 70 else '风险适中' if prob > 50 else '风险极高'}</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.info(f"{m_icon} **{m_name}**\n\n{m_tips}")
        
        st.markdown("#### 📜 实时战术日志")
        if not st.session_state.battle_logs: st.session_state.battle_logs.append(f"【{datetime.now().strftime('%H:%M:%S')}】🛰️ 卫星同步成功：物理状态 Active Sync")
        for log in st.session_state.battle_logs[:8]: st.caption(log)

    with col_r:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH"), row=1, col=1)
        
        # 压力支撑位自动标记
        fig.add_hline(y=res_px, line_dash="dash", line_color="red", annotation_text="物理压力位", row=1, col=1)
        fig.add_hline(y=sup_px, line_dash="dash", line_color="green", annotation_text="物理支撑位", row=1, col=1)
        
        flow_colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="净流"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(refresh_rate); st.rerun()

if __name__ == "__main__":
    render_commander()
