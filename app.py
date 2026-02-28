import streamlit as st
import requests
import time
from datetime import datetime

# 1. 彻底清空环境，只留最基础的显示
st.set_page_config(page_title="ETH_DEATH_SQUAD")
st.markdown("### 💀 ETH 10U 战神执行终端 (V15.0)")

# 2. 状态锁死：哪怕天崩地裂，账户数据必须准确
if 'S' not in st.session_state:
    st.session_state['S'] = {
        'bal': 10.0, 'pos': "NONE", 'ent': 0.0, 'size': 0.0, 'logs': []
    }

s = st.session_state['S']

# 3. 物理心跳获取
def get_px():
    try:
        # 直接物理获取价格
        r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP", timeout=1).json()
        return float(r['data'][0]['last'])
    except:
        return None

px = get_px()

# 4. 暴力点位逻辑：只抓“确定性”
# 我们设置一个简单的极值回归：如果 10 秒内波动超过 0.3%，视为插针，立即反向咬一口
if 'last_px' not in st.session_state: st.session_state.last_px = px

if px:
    diff = px - st.session_state.last_px
    # 只要没仓位，就死等那个“暴跌后的反弹”点位
    if s['pos'] == "NONE":
        st.info(f"⚖️ 正在监测瞬时波动... 当前价: {px}")
        # 如果瞬时跌幅超过 0.2%，认为这是“稳赢”的反弹起点
        if diff < -(px * 0.002): 
            s['ent'] = px
            s['pos'] = "LONG"
            s['size'] = (s['bal'] * 0.995) / px # 扣除滑点预留
            s['bal'] = 0.0
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ⚡ 瞬间插针捕捉：做多 @ {px}")
    else:
        # 持仓中：只看那 1.5% 的肉
        pnl = (px / s['ent'] - 1) if s['pos'] == "LONG" else (1 - px / s['ent'])
        st.warning(f"🎯 咬肉中: {pnl:.4%}")
        
        # 严格执行：咬到 1% 就跑，或者亏 0.5% 就割。10U 容不下贪婪。
        if pnl > 0.01 or pnl < -0.005:
            final = s['size'] * s['ent'] * (1 + pnl) * 0.999
            s['bal'] = round(final, 4)
            s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ✅ 撤退结算 | P/L: {pnl:.2%} | 余: ${s['bal']}")
            s['pos'], s['ent'], s['size'] = "NONE", 0.0, 0.0
            
    st.session_state.last_px = px

# 5. 渲染：不搞多列，不搞图表，只要结果
st.divider()
st.title(f"💰 余额: ${s['bal'] if s['pos'] == 'NONE' else (s['size']*px):.4f}")

if s['logs']:
    st.write("---")
    for log in reversed(s['logs']):
        st.code(log)

# 6. 原子频率刷新 (1秒一次，不能再快了，否则 OKX 封 IP)
time.sleep(1)
st.rerun()
