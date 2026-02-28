import streamlit as st
import requests
import time
from datetime import datetime

# 1. 物理层配置 - 绝不使用过时参数
st.set_page_config(page_title="ETH_SURVIVOR_V17")
st.markdown("<style>#MainMenu,footer{visibility:hidden;} .stMetric{background:#0e1117;}</style>", unsafe_allow_html=True)

# 2. 状态原子锁 - 唯一数据源
if 'SURVIVOR' not in st.session_state:
    st.session_state['SURVIVOR'] = {
        'bal': 10.0, 'pos': "NONE", 'ent': 0.0, 'size': 0.0, 
        'logs': [], 'history': [10.0]
    }

s = st.session_state['SURVIVOR']

# 3. 极简实时行情 - 杜绝假死
def get_px():
    try:
        r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP", timeout=1).json()
        return float(r['data'][0]['last'])
    except:
        return None

px = get_px()

# 4. 暴力执行内核 - 10U 活命逻辑
if px:
    # 记录上一秒价格，寻找“断崖式”入场机会
    if 'old_px' not in st.session_state: st.session_state.old_px = px
    
    # 计算瞬时跌幅 (只要那种能瞬间拉回来的深坑)
    drop = (px / st.session_state.old_px) - 1
    
    if s['pos'] == "NONE":
        # 只在瞬间跌幅超过 0.1% 的非理性时刻进场咬肉
        if drop < -0.001: 
            s['ent'] = px
            s['pos'] = "LONG"
            s['size'] = (s['bal'] * 0.999) / px # 预留手续费
            s['bal'] = 0.0
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 🚨 捕捉瞬间插针 @ {px}")
    else:
        # 持仓状态：10U 账户严禁贪婪
        pnl = (px / s['ent'] - 1)
        # 0.8% 止盈 (覆盖手续费后有肉) / 0.4% 止损 (保命)
        if pnl > 0.008 or pnl < -0.004:
            # 退出扣除 0.05% 滑点
            exit_val = s['size'] * s['ent'] * (1 + pnl) * 0.999
            s['bal'] = round(exit_val, 4)
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ✅ 闪击撤退 | P/L: {pnl:.2%} | 余: ${s['bal']}")
            s['pos'], s['size'] = "NONE", 0.0
    
    st.session_state.old_px = px

# 5. UI 渲染 - 绝对线性平铺 (杜绝 NameError)
st.title(f"💰 账户余额: ${s['bal'] if s['pos'] == 'NONE' else (s['size']*px):.4f}")
st.write(f"实时行情: `{px if px else 'SYNCING...'}`")

if s['pos'] != "NONE":
    st.warning(f"🎯 持仓中: {s['pos']} | 实时盈亏: {((px/s['ent']-1)*100):.3f}%")
else:
    st.info("🛰️ 侦测中心：正在等待极端插针信号...")

# 简易审计志
if s['logs']:
    st.divider()
    for l in reversed(s['logs']):
        st.code(l)

# 6. 原子级重载
time.sleep(1)
st.rerun()
