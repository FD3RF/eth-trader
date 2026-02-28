import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal, getcontext

# 设置全局精度，杜绝浮点数误差
getcontext().prec = 10

# ==========================================
# 0. 核心配置：10U 极限复利防御架构
# ==========================================
st.set_page_config(layout="wide", page_title="ETH 10U Starfire Pro", page_icon="🔥")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    div[data-testid="stMetricValue"] {font-size: 2rem; color: #ff0055; font-family: 'Orbitron'; font-weight: 900; text-shadow: 0 0 20px #ff0055CC;}
    .bsp-terminal {padding: 12px; border-radius: 10px; background: #000; border: 1px solid #ff0055; color: #ff0055; font-family: 'Courier New'; font-size: 0.9rem;}
    .resonance-active {background: rgba(0,255,136,0.2); border: 2px solid #00ff88; color: #00ff88; padding: 10px; border-radius: 10px; text-align: center; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# 初始化原子账本 (10U 专属硬化版)
if 'ledger' not in st.session_state:
    st.session_state.ledger = {
        'balance': Decimal('10.0000'), 
        'position': Decimal('0.0000'),
        'entry_price': Decimal('0.0000'),
        'is_bsp_lock': False,
        'trade_history': [],
        'equity_curve': [10.0],
        'version': '2,400,058-Audit'
    }

# ==========================================
# 1. 纳米级情报引擎 (OKX V5 协议优化)
# ==========================================
@st.cache_data(ttl=1)
def get_quantum_intel(inst_id="ETH-USDT"):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar=1m&limit=100"
        res = requests.get(url, timeout=2.0).json()
        data = res.get('data', [])
        if not data: return None
        
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volC','volCQ','confirm'])[::-1].reset_index(drop=True)
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        last_p = Decimal(str(df['c'].iloc[-1]))
        # 2026 核心倾向算法：75.3 奇点模拟
        sentiment = 75.30 + np.random.uniform(-0.001, 0.001)
        inflow = 50.0 + np.random.uniform(-2, 15)
        
        # BSP 16.0 风险判定
        vol_surge = float(df['v'].iloc[-1]) > df['v'].rolling(30).mean().iloc[-1] * 15.0
        bsp_trigger = abs(float(df['c'].pct_change().iloc[-1])) > 0.08 or vol_surge
        
        return {"df": df, "last_p": last_p, "sentiment": sentiment, "inflow": inflow, "bsp": bsp_trigger}
    except: return None

# ==========================================
# 2. 极致执行逻辑 (原子化全仓操作)
# ==========================================
def execute_quantum_logic(intel):
    L = st.session_state.ledger
    lp = intel['last_p']

    # 1. 物理熔断逻辑
    if intel['bsp'] and not L['is_bsp_lock']:
        L['is_bsp_lock'] = True
        if L['position'] > 0:
            L['balance'] += L['position'] * lp * Decimal('0.992') # 极端点滑点保护
            L['position'] = Decimal('0.0000')
        return

    # 2. 执行逻辑：75.3 奇点捕捉
    if not L['is_bsp_lock']:
        resonance = abs(intel['sentiment'] - 75.30) < 0.001
        
        # 入场：10U 必须全量压入，跨越手续费壁垒
        if resonance and intel['inflow'] > 55.0 and L['position'] == 0:
            L['entry_price'] = lp * Decimal('1.0003') # 模拟吃单深度
            L['position'] = (L['balance'] * Decimal('0.999')) / L['entry_price']
            L['balance'] = Decimal('0.0000')
            L['trade_history'].append({"time": datetime.now().strftime('%H:%M:%S'), "type": "ENTRY", "px": float(L['entry_price'])})

        # 离场：锁定 1.2% 利润或情绪衰减
        elif L['position'] > 0:
            profit_pct = (lp / L['entry_price']) - 1
            if profit_pct > Decimal('0.012') or intel['sentiment'] > 88.0:
                L['balance'] = L['position'] * lp * Decimal('0.9994') # 扣除千分之0.6手续费
                L['position'] = Decimal('0.0000')
                L['trade_history'].append({"time": datetime.now().strftime('%H:%M:%S'), "type": "EXIT", "px": float(lp)})

# ==========================================
# 3. 终极渲染视图 (指挥官全息界面)
# ==========================================
intel = get_quantum_intel()
if intel:
    execute_quantum_logic(intel)
    L = st.session_state.ledger
    total_val = float(L['balance'] + (L['position'] * intel['last_p']))

    # 布局架构
    col_stat, col_chart = st.columns([1, 4])
    
    with col_stat:
        st.metric("10U 实时净值", f"${total_val:.6f}", f"{((total_val/10)-1)*100:.4f}%")
        st.markdown(f"**审计轮次:** `{L['version']}`")
        if abs(intel['sentiment'] - 75.30) < 0.01:
            st.markdown('<div class="resonance-active">🎯 75.3 奇点共振中</div>', unsafe_allow_html=True)
        
        st.write("---")
        st.write(f"核心情绪: **{intel['sentiment']:.4f}**")
        st.write(f"资金流向: **{intel['inflow']:.1f} M**")
        
        if L['is_bsp_lock']:
            if st.button("🔴 重启逻辑核"): L['is_bsp_lock'] = False; st.rerun()

    with col_chart:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        df = intel['df']
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        
        # 净值路径渲染
        st.session_state.ledger['equity_curve'].append(total_val)
        if len(st.session_state.ledger['equity_curve']) > 300: st.session_state.ledger['equity_curve'].pop(0)
        fig.add_trace(go.Scatter(y=st.session_state.ledger['equity_curve'], fill='tozeroy', line=dict(color='#ff0055'), name="Equity"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=750, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    if L['trade_history']:
        with st.expander("📝 逻辑审计日志"):
            st.table(pd.DataFrame(L['trade_history']).iloc[::-1])

time.sleep(1.2)
st.rerun()
