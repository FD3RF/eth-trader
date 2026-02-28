import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 指挥部：环境硬锁与内存治理
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极指挥官", page_icon="⚖️")

def init_commander_state():
    if 'battle_logs' not in st.session_state: st.session_state.battle_logs = []
    if 'ls_ratio' not in st.session_state: st.session_state.ls_ratio = 1.0
    if 'last_cleanup_ts' not in st.session_state: st.session_state.last_cleanup_ts = time.time()

def auto_memory_cleanup():
    """V1 核心：4小时深度治理，防止挂机崩溃"""
    if time.time() - st.session_state.last_cleanup_ts > 14400:
        st.session_state.battle_logs = [f"【系统】{datetime.now().strftime('%H:%M:%S')} 内存治理完成：历史日志已归档。"]
        st.session_state.last_cleanup_ts = time.time()
        st.cache_data.clear()

# ==========================================
# 2. 数据层：多源情报同步
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
# 3. 渲染层：左侧边栏 (V1+V2 混合体)
# ==========================================
def render_sidebar_intelligence(df):
    with st.sidebar:
        st.markdown("### 🛸 量子实时控制")
        heartbeat = st.slider("心跳频率 (秒)", 5, 60, 10)
        f_ema = st.number_input("快线 EMA", 5, 30, 12)
        s_ema = st.number_input("慢线 EMA", 20, 100, 26)
        
        st.divider()
        
        # --- V1 盈亏比计算核心 ---
        last_p = df['c'].iloc[-1]
        atr = df['atr'].iloc[-1]
        prob = 50.0 + (10 if df['ema_f'].iloc[-1] > df['ema_s'].iloc[-1] else -5) + (15 if st.session_state.ls_ratio < 0.95 else -10)
        prob = max(min(prob, 99.0), 1.0)
        
        tp_px = last_p + (atr * (2.5 if prob > 60 else 1.5))
        sl_px = last_p - (atr * 1.5)
        rr_ratio = abs(tp_px - last_p) / abs(last_p - sl_px)
        
        box_color = "#00ff00" if (prob > 60 and rr_ratio > 1.8) else "#FFD700" if prob > 50 else "#ff4b4b"
        st.markdown(f"""
            <div style="border:1px solid {box_color}; padding:15px; border-radius:10px; background:rgba(0,0,0,0.3); text-align:center;">
                <div style="color:{box_color}; font-size:0.8em; font-weight:bold;">AI 实时胜率</div>
                <div style="color:{box_color}; font-size:1.8em; font-weight:bold;">{prob:.1f}%</div>
                <div style="color:#888; font-size:0.7em; margin-top:5px;">建议 R/R: 1 : {rr_ratio:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- V2 AI复盘报告 ---
        st.markdown("#### 🔍 AI 自动复盘")
        is_vol_push = df['v'].iloc[-1] > df['v'].tail(10).mean() * 1.5
        recap = "📉 缩量诱多：价格虚拉，警惕反手。" if (df['c'].iloc[-1] > df['c'].iloc[-2] and not is_vol_push) else "✅ 趋势运行：量价匹配。"
        st.info(f"{recap}\n\n散户情绪: {'看多 🔥' if st.session_state.ls_ratio < 0.95 else '看空 ❄️'}")
        
        st.divider()
        
        # --- 策略执行卡片 ---
        st.markdown("#### 🎯 策略执行计划")
        strats = [
            {"name": "物理位陷阱", "state": "✅ 激活" if abs(last_p - df['l'].min()) < 5 else "⚪ 观察", "tp": df['h'].max(), "sl": last_p - (atr*1.5)},
            {"name": "清算猎杀", "state": "🔥 进攻" if prob > 65 else "⚪ 待机", "tp": df['h'].max()*1.02, "sl": last_p - (atr*2)}
        ]
        for s in strats:
            color = "#00ff00" if "✅" in s['state'] or "🔥" in s['state'] else "#888"
            st.markdown(f"""<div style="border-left:3px solid {color}; padding-left:8px; margin-bottom:10px;">
                <div style="font-size:0.85em; color:{color}; font-weight:bold;">{s['state']} | {s['name']}</div>
                <div style="font-size:0.75em; color:#00ff00;">🎯 TP: ${s['tp']:.1f} | <span style="color:#ff4b4b;">🛡️ SL: ${s['sl']:.1f}</span></div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📜 战术日志")
        if not st.session_state.battle_logs: st.session_state.battle_logs.append(f"【{datetime.now().strftime('%H:%M:%S')}】卫星同步成功")
        for log in st.session_state.battle_logs[-5:]: st.caption(log)
        
        return heartbeat, f_ema, s_ema

# ==========================================
# 4. 主界面：全屏图表区
# ==========================================
def main():
    init_commander_state()
    auto_memory_cleanup()
    
    # 获取边栏参数并渲染
    f_ema_dummy = 12; s_ema_dummy = 26 # 占位符
    df_preview = get_market_intel(f_ema_dummy, s_ema_dummy)
    if df_preview.empty: st.error("❌ 卫星同步失败"); return
    
    hb, f_e, s_e = render_sidebar_intelligence(df_preview)
    df = get_market_intel(f_e, s_e) # 最终带参数抓取
    
    # 顶部仪表盘
    st.markdown(f"### 🛰️ ETH 量子决策指挥官 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("实时价格", f"${df['c'].iloc[-1]:.2f}")
    m2.metric("全网多空比", f"{st.session_state.ls_ratio}", "散户看空" if st.session_state.ls_ratio < 1 else "散户看多")
    m3.metric("ATR 动态波幅", f"{df['atr'].iloc[-1]:.2f}")
    m4.metric("庄家净流", f"{df['net_flow'].iloc[-1]:.0f}")

    # 绘图区
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
    
    # 爆仓带绘制 (逆推 50x 杠杆)
    h_max, l_min = df['h'].tail(60).max(), df['l'].tail(60).min()
    fig.add_hrect(y0=h_max*1.018, y1=h_max*1.02, fillcolor="red", opacity=0.3, line_width=0, annotation_text="空头燃料", row=1, col=1)
    fig.add_hrect(y0=l_min*0.98, y1=l_min*0.982, fillcolor="green", opacity=0.3, line_width=0, annotation_text="多头燃料", row=1, col=1)
    
    # 资金流
    colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
    fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="Flow"), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    time.sleep(hb); st.rerun()

if __name__ == "__main__":
    main()
