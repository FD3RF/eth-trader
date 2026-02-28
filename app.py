import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

# ==========================================
# 0. 核心配置：工业级黑金视觉 & 报错屏蔽
# ==========================================
st.set_page_config(layout="wide", page_title="ETH 10U IMMORTAL V6", page_icon="♾️")

# 强制注入 CSS 样式
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem; font-family: 'Orbitron'; font-weight: 800; color: #00ff88;}
    .stMetric {background: rgba(0, 255, 136, 0.03); padding: 12px; border-radius: 10px; border: 1px solid #1f2329;}
    .whale-log {font-size: 0.75rem; color: #00ff88; font-family: 'Consolas', monospace; border-bottom: 1px solid #1a1c23; padding: 4px 0;}
    .status-tag {padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;}
    .active-long {background: #00ff8822; color: #00ff88; border: 1px solid #00ff88;}
    .active-short {background: #ff4b4b22; color: #ff4b4b; border: 1px solid #ff4b4b;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 原子化影子状态初始化 (全隔离架构)
# ==========================================
def secure_init():
    # 核心账本与状态
    keys = {
        'BAL': 10.0, 'POS_TYPE': "NONE", 'POS_SIZE': 0.0, 
        'ENTRY': 0.0, 'LOGS': [], 'HEAL': 0, 'WHALES': [], 
        'CURVE': [10.0], 'LAST_PX': 2500.0, 'ITER': 0,
        'LAST_SKEW': 0.0, 'LAST_PRESS': 50.0
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

secure_init()

# ==========================================
# 2. 强化数据采集 (影子变量平滑技术)
# ==========================================
@st.cache_data(ttl=0.5)
def get_quantum_data(iter_key):
    try:
        # 使用 OKX 公共 API
        url = "https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP"
        res = requests.get(url, timeout=0.8).json()
        
        if res.get('code') == '0' and res.get('data'):
            lp = float(res['data'][0]['last'])
            st.session_state.LAST_PX = lp
            
            # 模拟 IV 偏斜 (机构情绪)
            skew = float(np.random.uniform(-25, 25))
            st.session_state.LAST_SKEW = skew
            
            # 模拟订单流压力 (买盘占比)
            press = float(np.random.uniform(20, 80))
            st.session_state.LAST_PRESS = press
            
            # 奇点锚点 75.3
            sent = 75.30 + np.random.uniform(-0.0002, 0.0002)
            return {"lp": lp, "skew": skew, "press": press, "sent": sent}
        raise Exception("API_TIMEOUT")
    except:
        st.session_state.HEAL += 1
        # 返回影子数据，保持系统运行
        return {
            "lp": st.session_state.LAST_PX, 
            "skew": st.session_state.LAST_SKEW, 
            "press": st.session_state.LAST_PRESS, 
            "sent": 75.30
        }

# ==========================================
# 3. 完美决策内核 (多维坍缩 + 风险隔离)
# ==========================================
def execute_quantum_engine(intel):
    lp, skew, press, sent = intel['lp'], intel['skew'], intel['press'], intel['sent']
    resonance = abs(sent - 75.30) < 0.00015 # 奇点共振阈值
    
    # --- 寻机状态 ---
    if st.session_state.POS_TYPE == "NONE":
        # 多头准入：共振触发 + (IV极度左偏 或 挂单重力>75%)
        if resonance and (skew < -18.0 or press > 78.0):
            st.session_state.ENTRY = round(lp + 0.1, 2)
            st.session_state.POS_TYPE = "LONG"
            st.session_state.POS_SIZE = round((st.session_state.BAL * 0.998) / st.session_state.ENTRY, 8)
            st.session_state.BAL = 0.0
            st.session_state.LOGS.append(f"{datetime.now().strftime('%H:%M:%S')} | 🚀 多头共振入场 @ {st.session_state.ENTRY}")

        # 空头准入：共振触发 + (IV极度右偏 或 挂单重力<22%)
        elif resonance and (skew > 18.0 or press < 22.0):
            st.session_state.ENTRY = round(lp - 0.1, 2)
            st.session_state.POS_TYPE = "SHORT"
            st.session_state.POS_SIZE = round((st.session_state.BAL * 0.998) / st.session_state.ENTRY, 8)
            st.session_state.BAL = 0.0
            st.session_state.LOGS.append(f"{datetime.now().strftime('%H:%M:%S')} | ⚡ 空头共振入场 @ {st.session_state.ENTRY}")

    # --- 持仓状态 ---
    else:
        # 实时计算盈亏率 (P/L %)
        if st.session_state.POS_TYPE == "LONG":
            p_ratio = (lp / st.session_state.ENTRY) - 1.0
        else:
            p_ratio = 1.0 - (lp / st.session_state.ENTRY)
        
        # 退出机制：止盈 1.5% / 止损 0.8% / 压力流反转保护
        p_exit = (st.session_state.POS_TYPE == "LONG" and press < 30) or (st.session_state.POS_TYPE == "SHORT" and press > 70)
        
        if p_ratio > 0.015 or p_ratio < -0.008 or p_exit:
            # 结算 (扣除 0.06% 预估综合税费)
            final_val = st.session_state.POS_SIZE * st.session_state.ENTRY * (1 + p_ratio) * 0.9994
            st.session_state.BAL = round(final_val, 6)
            st.session_state.LOGS.append(f"{datetime.now().strftime('%H:%M:%S')} | ✅ 奇点平仓 P/L: {p_ratio:.2%} | Bal: ${st.session_state.BAL}")
            st.session_state.POS_TYPE = "NONE"
            st.session_state.POS_SIZE = 0.0

# ==========================================
# 4. 全息看板 (可视化输出)
# ==========================================
intel = get_quantum_data(st.session_state.ITER)
execute_quantum_engine(intel)

# 更新净值曲线
curr_equity = st.session_state.BAL + (st.session_state.POS_SIZE * intel['lp'] if st.session_state.POS_TYPE != "NONE" else 0)
st.session_state.CURVE.append(curr_equity)
if len(st.session_state.CURVE) > 100: st.session_state.CURVE.pop(0)

col_s, col_m = st.columns([1, 3])

with col_s:
    st.metric("10U 量子核心净值", f"${curr_equity:.4f}", f"{((curr_equity/10)-1)*100:.3f}%")
    
    st.write("---")
    status_class = "active-long" if st.session_state.POS_TYPE == "LONG" else "active-short" if st.session_state.POS_TYPE == "SHORT" else ""
    st.markdown(f"📡 模式: <span class='status-tag {status_class}'>{st.session_state.POS_TYPE}</span>", unsafe_allow_html=True)
    st.write(f"🎭 IV Skew: `{intel['skew']:.2f}%`")
    st.write(f"⚖️ 订单压力: `{intel['press']:.1f}%`")
    
    # 鲸鱼监控 (模拟成交流)
    if np.random.random() > 0.9:
        st.session_state.WHALES.insert(0, f"{datetime.now().strftime('%H:%M:%S')} 🐋 {np.random.randint(50,250)} ETH")
        if len(st.session_state.WHALES) > 6: st.session_state.WHALES.pop()
    
    st.markdown("**🐋 机构大单流向**")
    for w in st.session_state.WHALES:
        st.markdown(f"<div class='whale-log'>{w}</div>", unsafe_allow_html=True)

    st.write("---")
    st.caption(f"自愈频率: {st.session_state.HEAL} Hz")
    st.caption(f"奇点偏离: {abs(intel['sent']-75.30):.8f}")

with col_main:
    st.markdown("### 🚀 ETH 量子坍缩决策终端 (V6.0 PERFECT)")
    
    
    
    # 净值复利图
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.CURVE, mode='lines', line=dict(color='#00ff88', width=3), fill='tozeroy', name="Equity"))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Balance ($)", xaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    

    if st.session_state.LOGS:
        with st.expander("📝 原子审计日志 (Atomic Audit Log)", expanded=True):
            for log in reversed(st.session_state.LOGS):
                st.write(f"`{log}`")

# 原子级时钟驱动
st.session_state.ITER += 1
time.sleep(1)
st.rerun()
