import streamlit as st
import requests
import time
from datetime import datetime

# 1. 物理层隔离：只留最原始的执行逻辑
st.set_page_config(page_title="ETH_GOD_V16")
st.markdown("<style>#MainMenu,footer{visibility:hidden;} .stCodeBlock{border:1px solid #ff4b4b;}</style>", unsafe_allow_html=True)

# 2. 状态原子化：锁死唯一内存地址
if 'ST' not in st.session_state:
    st.session_state.ST = {'bal': 10.0, 'pos': "NONE", 'ent': 0.0, 'size': 0.0, 'logs': []}

s = st.session_state.ST

# 3. 极简实时行情 (移除所有 Cache，杜绝假死)
def get_px():
    try:
        # 直接物理获取 OKX 永续合约实时价
        r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP", timeout=1).json()
        return float(r['data'][0]['last'])
    except:
        return None

px = get_px()

# 4. 暴力过滤逻辑：只打“必胜”的插针回归
if px:
    # 模拟极简动态波段：计算这一秒的偏离度 (不使用历史数据，减少内存负担)
    # 我们只在发生“断崖式”下跌（瞬时跌幅 > 0.1%）时进场咬反弹
    if 'prev' not in st.session_state: st.session_state.prev = px
    gap = px - st.session_state.prev
    
    if s['pos'] == "NONE":
        # 严格入场：必须瞬时暴跌才会触发，杜绝在阴跌中被磨损
        if gap < -(px * 0.0015): 
            s['ent'] = px
            s['pos'] = "LONG"
            s['size'] = (s['bal'] * 0.999) / px # 极限仓位，预留极低滑点
            s['bal'] = 0.0
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 🗡️ 瞬时暴跌捕捉 | 入场 @ {px}")
    else:
        # 持仓中：严格执行 0.8% 止盈 / 0.4% 止损
        # 10U 账户只能玩极短线，利润一到立刻撤退
        pnl = (px / s['ent'] - 1) if s['pos'] == "LONG" else (1 - px / s['ent'])
        
        if pnl > 0.008 or pnl < -0.004:
            # 扣除手续费 0.05% * 2
            val = s['size'] * s['ent'] * (1 + pnl) * 0.999
            s['bal'] = round(val, 4)
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 💰 咬肉成功 | P/L: {pnl:.2%} | 余额: ${s['bal']}")
            s['pos'], s['ent'], s['size'] = "NONE", 0.0, 0.0
            
    st.session_state.prev = px

# 5. UI 强制平铺：不使用 columns，不给 NameError 任何机会
st.title(f"💰 账户净值: ${s['bal'] if s['pos'] == 'NONE' else (s['size']*px):.4f}")
st.write(f"实时价格: `{px if px else 'LINK_LOST'}`")

if s['pos'] != "NONE":
    st.warning(f"⚠️ 战斗中 | 持仓: {s['pos']} | 盈亏: {((px/s['ent']-1)*100):.3f}%")
else:
    st.info("🎯 状态：冷静待机，等待市场失衡瞬间...")

if s['logs']:
    st.write("---")
    for log in reversed(s['logs']):
        st.code(log)

# 6. 循环
time.sleep(1)
st.rerun()
