import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 环境锁与内存治理 (V1 核心)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="⚖️")

def init_commander_state():
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup_ts' not in st.session_state: st.session_state.last_cleanup_ts = time.time()

def auto_memory_cleanup():
    """每4小时深度清理，防止挂机内存溢出"""
    if time.time() - st.session_state.last_cleanup_ts > 14400:
        st.session_state.battle_logs = [f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存治理完成。"]
        st.session_state.last_cleanup_ts = time.time()
        st.cache_data.clear()

# ==========================================
# 2. 数据情报引擎
# ==========================================
@st.cache_data(ttl=10)
def get_market_intel(f_ema, s_ema):
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        # 指标计算
        df['ema_f'] = df['c'].ewm(span=f_ema, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=s_ema, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        ls_res = requests.get(ls_url, timeout=5).json()
        if ls_res.get('code') == '0': st.session_state.ls_ratio = float(ls_res['data'][0][1])
        return df
    except: return pd.DataFrame()

# ==========================================
# 3. 终极边栏：胜率、R/R、复盘与策略
# ==========================================
def render_sidebar_intelligence(df):
    with st.sidebar:
        st.markdown("### 🛸 量子实时控制")
        hb = st.slider("心跳频率 (秒)", 5, 60, 10)
        f_e = st.number_input("快线 EMA", 5, 30, 12)
        s_e = st.number_input("慢线 EMA", 20, 100, 26)
        st.divider()

        # --- V1 盈亏比矩阵集成 ---
        last_p = df['c'].iloc[-1]
        atr = df['atr'].iloc[-1]
        prob = 50.0 + (10 if df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1] else -5) + (15 if st.session_state.ls_ratio < 0.95 else -10)
        prob = max(min(prob, 99.0), 1.0)
        
        # 动态计算建议 TP/SL 和 R/R
        tp_sugg = last_p + (atr * 2.5) if prob > 50 else last_p - (atr * 2.5)
        sl_sugg = last_p - (atr * 1.5) if prob > 50 else last_p + (atr * 1.5)
        rr_sugg = abs(tp_sugg - last_p) / abs(last_p - sl_sugg)
        
        box_color = "#ff4b4b" if prob < 45 else "#FFD700" if prob < 60 else "#00ff00"
        st.markdown(f"""
            <div style="border:1px solid {box_color}; padding:15px; border-radius:10px; background:rgba(255,255,255,0.05); text-align:center;">
                <div style="color:{box_color}; font-size:0.8em; font-weight:bold;">AI 实时胜率</div>
                <div style="color:{box_color}; font-size:2em; font-weight:bold;">{prob:.1f}%</div>
                <div style="color:#888; font-size:0.7em; margin-top:5px;">建议 R/R: 1 : {rr_sugg:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

        # --- AI 自动复盘 ---
        st.markdown("#### 🔍 AI 自动复盘")
        recap = "📉 缩量诱多：价格虚拉，警惕反手。" if (df['c'].iloc[-1] > df['c'].iloc[-2] and df['v'].iloc[-1] < df['v'].tail(10).mean()) else "✅ 趋势运行：目前量价匹配。"
        st.info(f"{recap}\n\n散户情绪: {'看多 🔥' if st.session_state.ls_ratio < 0.95 else '看空 ❄️'}")
        st.divider()

        # --- 策略执行计划 ---
        st.markdown("#### 🎯 策略执行计划")
        strats = [
            {"name": "物理位陷阱", "state": "⚪ 观察", "tp": df['h'].tail(30).max(), "sl": last_p - 10},
            {"name": "清算猎杀", "state": "🔥 进攻" if prob > 60 else "⚪ 待机", "tp": last_p + 30, "sl": last_p - 15}
        ]
        for s in strats:
            st.markdown(f"""<div style="font-size:0.85em; margin-bottom:10px;">
                <b>{s['state']} | {s['name']}</b><br/>
                <span style="color:#00ff00;">🎯 TP: ${s['tp']:.1f}</span> | <span style="color:#ff4b4b;">🛡️ SL: ${s['sl']:.1f}</span>
            </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"【{datetime.now().strftime('%H:%M:%S')}】卫星同步成功")
        return hb, f_e, s_e

# ==========================================
# 4. 主大屏渲染
# ==========================================
def main():
    init_commander_state()
    auto_memory_cleanup()
    
    df_init = get_market_intel(12, 26)
    if df_init.empty: st.error("❌ 数据同步失败"); return
    
    hb, f_e, s_e = render_sidebar_intelligence(df_init)
    df = get_market_intel(f_e, s_e)
    
    # 仪表盘
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("实时价格", f"${df['c'].iloc[-1]:.2f}")
    m2.metric("全网多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    m3.metric("ATR 动态波幅", f"{df['atr'].iloc[-1]:.2f}")
    m4.metric("庄家净流", f"{df['net_flow'].iloc[-1]:.0f}")

    # 图表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
    
    # 清算地雷区可视化
    h_max = df['h'].tail(60).max()
    l_min = df['l'].tail(60).min()
    fig.add_hrect(y0=h_max*1.018, y1=h_max*1.02, fillcolor="red", opacity=0.3, line_width=0, annotation_text="空头燃料", row=1, col=1)
    fig.add_hrect(y0=l_min*0.98, y1=l_min*0.982, fillcolor="green", opacity=0.3, line_width=0, annotation_text="多头燃料", row=1, col=1)

    colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="Flow"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    time.sleep(hb); st.rerun()

if __name__ == "__main__":
    main()
