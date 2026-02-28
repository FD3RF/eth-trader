import streamlit as st
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

# ==========================================
# 1. 物理层配置 (全面禁用过时参数，杜绝黄字警告)
# ==========================================
st.set_page_config(layout="wide", page_title="ETH WARLOCK V12", page_icon="🔱")

# 强制注入 CSS 确保界面不崩
st.markdown("""
    <style>
    #MainMenu, footer, .stDeployButton {visibility: hidden;}
    .stMetric {background: rgba(0, 255, 136, 0.05); border: 1px solid #333; padding: 15px; border-radius: 10px;}
    div[data-testid="stMetricValue"] {color: #00ff88; font-family: 'Orbitron'; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 原子化状态保险箱 (用单一 KEY 锁死，绝不报 NameError)
# ==========================================
if 'CORE' not in st.session_state:
    st.session_state['CORE'] = {
        'bal': 10.0,      # 初始本金
        'pos': "NONE",    # 持仓状态
        'size': 0.0,      # 持仓数量
        'ent': 0.0,       # 入场价格
        'logs': [],       # 交易日志
        'curve': [10.0],  # 净值曲线
        'last_px': 2800.0 # 影子价格
    }

# 建立快捷引用（指向同一个内存地址）
c = st.session_state['CORE']

# ==========================================
# 3. 暴力实时抓取 (抛弃 Cache，每秒物理更新)
# ==========================================
def get_live_market():
    try:
        # 直接请求 OKX V5 接口
        r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP", timeout=0.8).json()
        px = float(r['data'][0]['last'])
        c['last_px'] = px
        return px
    except:
        # 失败则回滚影子价格，保证逻辑不断裂
        return c['last_px']

# ==========================================
# 4. 战神执行内核 (只捕获 75.3 奇点坍缩)
# ==========================================
current_px = get_live_market()

# 生成三因子物理噪声 (模拟 IV Skew, 订单压力, 75.3 奇点)
skew_val = float(np.random.uniform(-35, 35))
press_val = float(np.random.uniform(10, 90))
sent_val = 75.30 + np.random.uniform(-0.0001, 0.0001)

# 进场条件判定 (极简、硬核、无歧义)
pulse_active = abs(sent_val - 75.30) < 0.00008
is_go_long = pulse_active and skew_val < -22.0 and press_val > 82.0
is_go_short = pulse_active and skew_val > 22.0 and press_val < 18.0

# 交易撮合引擎
if c['pos'] == "NONE":
    if is_go_long:
        c['ent'], c['pos'] = current_px, "LONG"
        c['size'] = (c['bal'] * 0.998) / current_px # 扣除预估手续费
        c['bal'] = 0.0
        c['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 🚀 做多入场 @ {current_px}")
    elif is_go_short:
        c['ent'], c['pos'] = current_px, "SHORT"
        c['size'] = (c['bal'] * 0.998) / current_px
        c['bal'] = 0.0
        c['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ⚡ 做空入场 @ {current_px}")
else:
    # 动态盈亏计算
    pnl_ratio = (current_px / c['ent'] - 1.0) if c['pos'] == "LONG" else (1.0 - current_px / c['ent'])
    
    # 铁律：1.5% 止盈 / 0.75% 止损
    if pnl_ratio > 0.015 or pnl_ratio < -0.0075:
        # 资金回笼
        exit_val = c['size'] * c['ent'] * (1 + pnl_ratio) * 0.9994
        c['bal'] = round(exit_val, 6)
        c['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ✅ 平仓结算 | P/L: {pnl_ratio:.2%} | 余额: ${c['bal']}")
        c['pos'], c['size'], c['ent'] = "NONE", 0.0, 0.0

# ==========================================
# 5. UI 物理布局 (拒绝解构变量，直接渲染)
# ==========================================
# 更新净值曲线
current_total = c['bal'] + (c['size'] * current_px if c['pos'] != "NONE" else 0)
c['curve'].append(current_total)
if len(c['curve']) > 120: c['curve'].pop(0)

# 定义列 (不赋值给 col_main 等可能报错的变量)
side_col, main_col = st.columns([1, 3])

# 左侧：数据面板
with side_col:
    st.metric("10U 实时账户", f"${current_total:.4f}", f"{((current_total/10)-1)*100:.3f}%")
    st.write("---")
    
    # 状态指示灯
    if is_go_long: st.success("多头共振触发")
    elif is_go_short: st.error("空头共振触发")
    else: st.info("🔍 侦测 75.3 奇点中")
    
    st.write(f"📊 IV Skew: `{skew_val:.2f}%`")
    st.write(f"⚖️ 订单压力: `{press_val:.1f}%`")
    st.caption(f"当前模式: {c['pos']}")

# 右侧：可视化矩阵
with main_col:
    st.markdown("### 🚀 ETH 量子坍缩决策终端 V12.0")
    
    
    
    # 复利曲线图表 (修正过时参数)
    fig = go.Figure(go.Scatter(y=c['curve'], mode='lines', line=dict(color='#00ff88', width=3), fill='tozeroy'))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=10,b=0), xaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 审计日志
    with st.expander("📝 交易流水 (Atomic Logs)", expanded=True):
        if c['logs']:
            for l in reversed(c['logs']): st.write(f"`{l}`")
        else:
            st.caption("等待首个共振信号...")

# ==========================================
# 6. 原子重载
# ==========================================
time.sleep(1)
st.rerun()
