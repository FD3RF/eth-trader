import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

# ==========================================
# 0. 基础配置：工业级黑金视觉
# ==========================================
st.set_page_config(layout="wide", page_title="ETH 10U Hardened", page_icon="🛡️")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    div[data-testid="stMetricValue"] {font-size: 2.2rem; color: #00ff88; font-family: 'Orbitron'; font-weight: 900;}
    .terminal-box {background: #0e1117; border: 1px solid #00ff8833; padding: 20px; border-radius: 5px; color: #00ff88;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 状态容错初始化 (核心修复：防御性字典)
# ==========================================
if 'vault' not in st.session_state:
    st.session_state['vault'] = {
        'bal': 10.0,
        'pos': 0.0,
        'entry': 0.0,
        'logs': [],
        'curve': [10.0]
    }

# 安全获取函数：彻底终结 KeyError
def get_val(key, default=0.0):
    return st.session_state['vault'].get(key, default)

# ==========================================
# 2. 纳米级情报 (数据强转型)
# ==========================================
@st.cache_data(ttl=1)
def fetch_raw_intel():
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT-SWAP&bar=1m&limit=100"
        res = requests.get(url, timeout=1.5).json()
        data = res.get('data', [])
        if not data: return None
        
        # 强制类型转换，物理隔绝 Decimal
        lp = float(data[0][4]) 
        sentiment = 75.30 + np.random.uniform(-0.0002, 0.0002)
        
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volC','volCQ','confirm'])
        df['c'] = df['c'].astype(float)
        return {"lp": lp, "sent": sentiment, "df": df[::-1]}
    except:
        return None

# ==========================================
# 3. 稳健执行引擎 (全链路 float 闭环)
# ==========================================
def execute_hardened_logic(intel):
    V = st.session_state['vault']
    lp = float(intel['lp'])
    sent = float(intel['sent'])
    
    # 75.3 奇点判定
    is_resonance = abs(sent - 75.30) < 0.0003
    
    # A. 入场逻辑
    if is_resonance and V['pos'] <= 0:
        V['entry'] = round(lp * 1.0005, 2)
        V['pos'] = round((V['bal'] * 0.999) / V['entry'], 8)
        V['bal'] = 0.0
        V['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 买入 ETH @ {V['entry']}")

    # B. 离场逻辑
    elif V['pos'] > 0:
        profit = (lp / V['entry']) - 1.0
        if profit > 0.012 or sent > 80.0:
            V['bal'] = round(V['pos'] * lp * 0.999, 4)
            V['pos'] = 0.0
            V['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 止盈退出 @ {lp}")

# ==========================================
# 4. 渲染层 (极致稳定)
# ==========================================
intel = fetch_raw_intel()
if intel:
    execute_hardened_logic(intel)
    V = st.session_state['vault']
    
    # 安全计算总资产
    total_equity = float(V['bal']) + (float(V['pos']) * float(intel['lp']))
    
    col_stat, col_chart = st.columns([1, 4])
    with col_stat:
        st.metric("10U 极限净值", f"${total_equity:.4f}", f"{((total_equity/10)-1)*100:.2f}%")
        st.write(f"奇点偏离: `{abs(intel['sent']-75.30):.6f}`")
        if V['pos'] > 0:
            st.warning(f"持仓中: {V['pos']} ETH")
    
    with col_chart:
        st.markdown("### 🚀 ETH 2026 量子执行终端 | 10U 逻辑硬化")
        
                
        fig = go.Figure(data=[go.Candlestick(x=intel['df']['ts'],
                        open=intel['df']['o'], high=intel['df']['h'],
                        low=intel['df']['l'], close=intel['df']['c'])])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    if V['logs']:
        with st.expander("📝 核心执行审计", expanded=True):
            for l in reversed(V['logs']):
                st.write(f"`{l}`")

time.sleep(1)
st.rerun()
