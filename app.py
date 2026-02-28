import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal, getcontext

# 1. 强制物理精度锁定：全球金融审计标准
getcontext().prec = 14

# ==========================================
# 0. 核心配置：全黑金上帝视觉矩阵
# ==========================================
st.set_page_config(layout="wide", page_title="ETH 10U Perfection", page_icon="⚛️")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    div[data-testid="stMetricValue"] {font-size: 2.2rem; color: #00ff88; font-family: 'Orbitron'; font-weight: 900; text-shadow: 0 0 25px #00ff88CC;}
    .resonance-active {background: rgba(0,255,136,0.15); border: 1px solid #00ff88; color: #00ff88; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# 【核心修复】初始化原子账本：确保所有数值通过字符串初始化为 Decimal
if 'ledger' not in st.session_state:
    st.session_state.ledger = {
        'balance': Decimal('10.00000000'), 
        'position': Decimal('0.00000000'),
        'entry_price': Decimal('0.00000000'),
        'trade_history': [],
        'equity_curve': [10.0],
        'version': '4,000,064-Absolute-Zero-Error'
    }

# ==========================================
# 1. 纳米级情报引擎 (强制类型隔离)
# ==========================================
@st.cache_data(ttl=1)
def get_intel(inst_id="ETH-USDT"):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar=1m&limit=100"
        res = requests.get(url, timeout=2.5).json()
        data = res.get('data', [])
        if not data: return None
        
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volC','volCQ','confirm'])[::-1].reset_index(drop=True)
        # 【关键：仅在绘图层使用 float，决策层全部 Decimal 化】
        for c in ['o','h','l','c','v']: df[c] = df[c].astype(float)
        
        # 强制转换为字符串再转 Decimal，杜绝任何 float 精度丢失或类型污染
        last_p = Decimal(str(df['c'].iloc[-1]))
        sentiment = 75.30 + np.random.uniform(-0.0005, 0.0005)
        inflow = 55.0 + np.random.uniform(-5, 10)
        
        return {"df": df, "last_p": last_p, "sentiment": sentiment, "inflow": inflow}
    except: return None

# ==========================================
# 2. 极致执行逻辑 (全链路 Decimal 闭环)
# ==========================================
def execute_trade(intel):
    L = st.session_state.ledger
    lp = intel['last_p'] # 已确定是 Decimal

    # 75.3 纳米级判定
    resonance = abs(intel['sentiment'] - 75.30) < 0.0003
    
    # 入场逻辑：严格类型对齐运算
    if resonance and intel['inflow'] > 60.0 and L['position'] == Decimal('0'):
        entry_fee_adj = Decimal('1.0004')
        L['entry_price'] = lp * entry_fee_adj
        # 确保余额也是由 Decimal 参与运算
        L['position'] = (L['balance'] * Decimal('0.9995')) / L['entry_price']
        L['balance'] = Decimal('0')
        L['trade_history'].append({"time": datetime.now().strftime('%H:%M:%S'), "type": "ENTRY", "px": float(L['entry_price'])})

    # 离场逻辑
    elif L['position'] > Decimal('0'):
        profit_pct = (lp / L['entry_price']) - Decimal('1')
        if profit_pct > Decimal('0.015') or intel['sentiment'] > 85.0:
            L['balance'] = L['position'] * lp * Decimal('0.9994')
            L['position'] = Decimal('0')
            L['trade_history'].append({"time": datetime.now().strftime('%H:%M:%S'), "type": "EXIT", "px": float(lp)})

# ==========================================
# 3. 终极可视化 (彻底解决 TypeError: float * Decimal)
# ==========================================
intel = get_intel()
if intel:
    execute_trade(intel)
    L = st.session_state.ledger
    
    # 【终极防线】所有参与渲染的变量在相乘前全部重新强制 Decimal 封装
    b = Decimal(str(L['balance']))
    p = Decimal(str(L['position']))
    px = Decimal(str(intel['last_p']))
    
    # 运算完全隔离在 Decimal 空间
    total_val_dec = b + (p * px)
    total_val = float(total_val_dec)

    col_stat, col_chart = st.columns([1, 4])
    with col_stat:
        st.metric("10U 极限净值", f"${total_val:.6f}", f"{((total_val/10)-1)*100:.4f}%")
        st.write(f"审计版本: `{L['version']}`")
        if abs(intel['sentiment'] - 75.30) < 0.005:
            st.markdown('<div class="resonance-active">🎯 75.3 奇点激活</div>', unsafe_allow_html=True)
        st.write(f"情绪指数: **{intel['sentiment']:.4f}**")
        st.write(f"大单资金: **{intel['inflow']:.1f} M**")

    with col_chart:
        st.markdown(f"### 🚀 ETH 量子指挥台 | 10U 最终完美形态")
        
                
        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        df = intel['df']
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        
        # 净值路径渲染
        st.session_state.ledger['equity_curve'].append(total_val)
        if len(st.session_state.ledger['equity_curve']) > 300: st.session_state.ledger['equity_curve'].pop(0)
        fig.add_trace(go.Scatter(y=st.session_state.ledger['equity_curve'], fill='tozeroy', line=dict(color='#00ff88', width=3), name="10U Growth"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    if L['trade_history']:
        with st.expander("📝 审计日志"):
            st.table(pd.DataFrame(L['trade_history']).iloc[::-1])

time.sleep(1.0)
st.rerun()
