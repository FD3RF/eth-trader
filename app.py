import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 指挥部初始化与环境锁
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="⚖️")

def init_commander_state():
    """初始化持久化战术状态"""
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'sentiment_score' not in st.session_state: st.session_state.sentiment_score = 50.0
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup_ts' not in st.session_state: st.session_state.last_cleanup_ts = time.time()

init_commander_state()

# ==========================================
# 2. 核心算法：模式识别、爆仓推算与 AI 复盘
# ==========================================
def calculate_liquidation_zones(df):
    """逆推 50x 杠杆生死线，识别价格磁吸区"""
    if df.empty: return []
    swing_high = df['h'].tail(120).max()
    swing_low = df['l'].tail(120).min()
    return [
        {'type': '空头爆仓(50x)', 'px': swing_high * 1.018, 'color': 'rgba(255, 0, 0, 0.4)'},
        {'type': '多头爆仓(50x)', 'px': swing_low * 0.982, 'color': 'rgba(0, 255, 0, 0.4)'}
    ]

def auto_mode_selector(df):
    """波动率模式识别与实战建议"""
    if len(df) < 60: return "初始化", "⏳", "等待数据沉淀"
    recent_atr = df['atr'].tail(60).mean()
    vol_ratio = (recent_atr / df['c'].iloc[-1]) * 100
    if vol_ratio > 0.15: return "趋势爆发模式", "🚀", "建议：放宽止盈，顺势而为"
    if vol_ratio < 0.05: return "极窄震荡模式", "🦥", "建议：持币观望，谨防洗盘"
    return "标准波段模式", "⚖️", "建议：高抛低吸，执行R/R计划"

def ai_market_recap(df, ls_ratio):
    """识别量价阴谋：缩量诱多或放量砸盘"""
    last, prev = df.iloc[-1], df.iloc[-2]
    is_vol_push = last['v'] > df['v'].tail(10).mean() * 1.5
    is_price_up = last['c'] > prev['c']
    
    if is_price_up and not is_vol_push: status = "📉 缩量诱多：庄家虚拉吸引散户，警惕随时反手。"
    elif not is_price_up and is_vol_push: status = "🚨 放量砸盘：主力抛售或大户强平，严禁接刀。"
    else: status = "✅ 趋势运行：目前量价匹配，处于标准模式。"
    
    sentiment = "看多 🔥" if ls_ratio < 0.95 else "看空 ❄️"
    return f"{status}\n\n【博弈核心】散户情绪: {sentiment} | 净流状态: {'流入' if df['net_flow'].iloc[-1]>0 else '流出'}"

def auto_memory_cleanup():
    """4小时强制内存防御治理"""
    if time.time() - st.session_state.last_cleanup_ts > 14400:
        st.cache_data.clear()
        st.session_state.last_cleanup_ts = time.time()
        st.session_state.battle_logs.append(f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存防御治理完成")

# ==========================================
# 3. 数据层：多源实时情报引擎
# ==========================================
@st.cache_data(ttl=10)
def get_market_intel(fast_ema, slow_ema):
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
    if df.empty: st.error("❌ 卫星连接失败：无法抓取市场实时情报"); return

    # 数据推算
    res_px = df['h'].tail(30).max()
    sup_px = df['l'].tail(30).min()
    m_name, m_icon, m_tips = auto_mode_selector(df)
    liq_zones = calculate_liquidation_zones(df)
    ai_recap = ai_market_recap(df, st.session_state.ls_ratio)
    
    last_p = df['c'].iloc[-1]
    prob = 50.0 + (10 if df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1] else -5) + (15 if st.session_state.ls_ratio < 0.95 else -10)
    prob = max(min(prob, 99.0), 1.0)

    # UI 顶层
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI 进攻胜率", f"{prob:.1f}%")
    c2.metric("模式识别", f"{m_icon} {m_name}")
    c3.metric("全网多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    c4.metric("实时波动 ATR", f"{df['atr'].iloc[-1]:.2f}")

    st.divider()

    col_l, col_r = st.columns([1, 4])
    with col_l:
        # AI 复盘报告
        st.markdown(f"""<div style="padding:15px; border:2px solid #00ff00; border-radius:12px; background:rgba(0,0,0,0.4); margin-bottom:15px;">
            <p style="color:#00ff00; font-weight:bold; margin:0;">🤖 AI 自动复盘报告</p>
            <p style="font-size:0.9em; color:white; margin:10px 0; line-height:1.5;">{ai_recap.replace('【', '<br>【')}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### ⚡ 爆仓地雷分布")
        for z in liq_zones: st.caption(f"{z['type']}: **${z['px']:.1f}**")
        
        st.markdown("#### 📜 实战日志")
        for log in st.session_state.battle_logs[-6:]: st.caption(log)

    with col_r:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH"), row=1, col=1)
        
        # 物理阻力与爆仓热力区
        fig.add_hline(y=res_px, line_dash="dash", line_color="red", annotation_text="物理压力", row=1, col=1)
        fig.add_hline(y=sup_px, line_dash="dash", line_color="green", annotation_text="物理支撑", row=1, col=1)
        for z in liq_zones:
            fig.add_hrect(y0=z['px']*0.9997, y1=z['px']*1.0003, fillcolor=z['color'], opacity=0.3, line_width=0, row=1, col=1)
        
        flow_colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="净流"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(refresh_rate); st.rerun()

if __name__ == "__main__":
    render_commander()
