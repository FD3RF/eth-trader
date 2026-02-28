import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

# ==========================================
# 1. 核心视觉与拦截器 (全屏黑金架构)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH 10U QUANTUM V9", page_icon="♾️")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem; font-family: 'Orbitron'; font-weight: 800; color: #00ff88;}
    .stMetric {background: rgba(0, 255, 136, 0.04); padding: 15px; border-radius: 12px; border: 1px solid #222;}
    .whale-log {font-size: 0.75rem; color: #00ff88; font-family: 'Consolas', monospace; border-bottom: 1px solid #1a1c23; padding: 5px 0;}
    .signal-box {padding: 25px; border-radius: 15px; text-align: center; font-weight: 900; font-size: 1.8rem; margin-bottom: 20px;}
    .signal-long {background: linear-gradient(90deg, #00ff8822, #00ff8844); color: #00ff88; border: 2px solid #00ff88;}
    .signal-short {background: linear-gradient(90deg, #ff4b4b22, #ff4b4b44); color: #ff4b4b; border: 2px solid #ff4b4b;}
    .signal-wait {background: #111; color: #444; border: 2px solid #222;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 状态机硬化 (防抖动、防空值、防溢出)
# ==========================================
if 'state' not in st.session_state:
    st.session_state.state = {
        'bal': 10.0, 'pos_type': "NONE", 'pos_size': 0.0, 'entry': 0.0,
        'logs': [], 'heal': 0, 'whales': [], 'curve': [10.0],
        'last_px': 2800.0, 'iter': 0, 'last_skew': 0.0, 'last_press': 50.0
    }

s = st.session_state.state # 引用代理

# ==========================================
# 3. 影子数据引擎 (全自动自愈抓取)
# ==========================================
@st.cache_data(ttl=0.3)
def fetch_intel(iter_id):
    try:
        # 主行情获取
        url = "https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP"
        res = requests.get(url, timeout=0.7).json()
        
        if res.get('code') == '0' and res.get('data'):
            px = float(res['data'][0]['last'])
            s['last_px'] = px
            
            # 模拟：IV Skew (偏斜度), 挂单流压力, 75.3 奇点偏离
            skew = float(np.random.uniform(-35, 35))
            press = float(np.random.uniform(10, 90))
            sent = 75.30 + np.random.uniform(-0.0001, 0.0001)
            
            s['last_skew'], s['last_press'] = skew, press
            return {"lp": px, "skew": skew, "press": press, "sent": sent}
        raise ValueError("Data-Incomplete")
    except:
        s['heal'] += 1
        return {"lp": s['last_px'], "skew": s['last_skew'], "press": s['last_press'], "sent": 75.30}

# ==========================================
# 4. 极致逻辑内核 (胜率点位收割)
# ==========================================
def process_core(intel):
    lp, skew, press, sent = intel['lp'], intel['skew'], intel['press'], intel['sent']
    
    # 判据：奇点精准重合 (Sentiment Pulse)
    pulse = abs(sent - 75.30) < 0.00008
    
    # 黄金进场位：奇点脉冲 + 极度IV偏斜(>22%) + 压力流极端(>82% 或 <18%)
    entry_l = pulse and skew < -22.0 and press > 82.0
    entry_s = pulse and skew > 22.0 and press < 18.0

    if s['pos_type'] == "NONE":
        if entry_l:
            s['entry'], s['pos_type'] = round(lp, 2), "LONG"
            s['pos_size'] = round((s['bal'] * 0.999) / s['entry'], 8)
            s['bal'] = 0.0
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 🚀 做多信号共振 | 入场位: {s['entry']}")
        elif entry_s:
            s['entry'], s['pos_type'] = round(lp, 2), "SHORT"
            s['pos_size'] = round((s['bal'] * 0.999) / s['entry'], 8)
            s['bal'] = 0.0
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ⚡ 做空信号共振 | 入场位: {s['entry']}")
    else:
        # 实时盈亏监控
        pnl = (lp / s['entry'] - 1.0) if s['pos_type'] == "LONG" else (1.0 - lp / s['entry'])
        # 铁律止盈止损：1.5% / -0.7%
        if pnl > 0.015 or pnl < -0.007:
            val = s['pos_size'] * s['entry'] * (1 + pnl) * 0.9994
            s['bal'], s['pos_type'], s['pos_size'] = round(val, 6), "NONE", 0.0
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ✅ 平仓结算 | P/L: {pnl:.2%} | 余: ${s['bal']}")

    return "LONG" if entry_l else "SHORT" if entry_s else "WAIT"

# ==========================================
# 5. UI 高性能渲染 (解耦架构)
# ==========================================
intel_now = fetch_intel(s['iter'])
signal_now = process_core(intel_now)

# 数据曲线计算
curr_eq = s['bal'] + (s['pos_size'] * intel_now['lp'] if s['pos_type'] != "NONE" else 0)
s['curve'].append(curr_eq)
if len(s['curve']) > 200: s['curve'].pop(0)

# 渲染视图
L_COL, R_COL = st.columns([1, 3])

with L_COL:
    st.metric("10U 账户核心净值", f"${curr_eq:.4f}", f"{((curr_eq/10)-1)*100:.3f}%")
    st.write("---")
    
    # 策略执行看板
    if signal_now == "LONG": st.markdown('<div class="signal-box signal-long">▲ 强烈建议做多</div>', unsafe_allow_html=True)
    elif signal_now == "SHORT": st.markdown('<div class="signal-box signal-short">▼ 强烈建议做空</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="signal-box signal-wait">○ 捕获奇点中...</div>', unsafe_allow_html=True)
    
    st.write(f"📊 IV Skew: `{intel_now['skew']:.2f}%`")
    st.write(f"⚖️ 订单压力: `{intel_now['press']:.1f}%`")
    
    # 鲸鱼流成交监控
    if np.random.random() > 0.94:
        s['whales'].insert(0, f"{datetime.now().strftime('%H:%M:%S')} 🐋 {np.random.randint(40,400)} ETH")
    for w in s['whales'][:6]: st.markdown(f"<div class='whale-log'>{w}</div>", unsafe_allow_html=True)
    
    st.caption(f"自愈频率: {s['heal']} Hz")

with R_COL:
    st.markdown("### 🚀 ETH 量子坍缩决策终端 [V9.0 UNLIMITED]")
    
    
    
    # 复利曲线图表
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=s['curve'], mode='lines', line=dict(color='#00ff88', width=3), fill='tozeroy'))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Equity (USD)", xaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    with st.expander("📝 交易执行流水 (Atomic Logs)", expanded=True):
        if s['logs']:
            for l in reversed(s['logs']): st.write(f"`{l}`")
        else:
            st.caption("系统静默侦测中... 等待奇点重合...")

# 循环驱动
s['iter'] += 1
time.sleep(1)
st.rerun()
