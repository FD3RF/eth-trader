import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# 1. 物理环境硬化
st.set_page_config(page_title="ETH_ALGO_V14")
st.markdown("### 🛠️ ETH 10U 策略执行器 (V14.0)")

# 2. 状态存储 (唯一的全局字典)
if 'S' not in st.session_state:
    st.session_state['S'] = {
        'bal': 10.0, 'pos': "NONE", 'ent': 0.0, 'size': 0.0, 'logs': []
    }

s = st.session_state['S']

# 3. 实时价格抓取 (物理心跳)
def get_px():
    try:
        r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP", timeout=2).json()
        return float(r['data'][0]['last'])
    except:
        return None

px = get_px()

# 如果拿不到价格，直接中止，防止假死
if px is None:
    st.error("网络链路中断，正在重新挂载信号...")
    time.sleep(1)
    st.rerun()

# 4. 暴力执行逻辑 (无花里胡哨指标，只抓确定性点位)
# 进场逻辑：价格跌破 24 小时极值或特定偏移 (此处模拟最稳共振点)
trigger = (int(time.time()) % 60 == 0) # 模拟每分钟一次的奇点扫描

if s['pos'] == "NONE":
    st.write(f"🔍 正在扫描奇点... 当前价格: `{px}`")
    # 最稳入场位判定 (此处为示例逻辑，核心在于 1.5% 止盈)
    if trigger and px > 0: 
        s['ent'] = px
        s['pos'] = "LONG"
        s['size'] = (s['bal'] * 0.99) / px
        s['bal'] = 0.0
        s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 🚀 进场做多 @ {px}")
else:
    pnl = (px / s['ent'] - 1) if s['pos'] == "LONG" else (1 - px / s['ent'])
    st.warning(f"⚠️ 当前持仓: {s['pos']} | 盈亏: {pnl:.4%}")
    
    # 硬性止盈止损线 (1.5% / -0.8%)
    if pnl > 0.015 or pnl < -0.008:
        final = s['size'] * s['ent'] * (1 + pnl) * 0.999
        s['bal'] = round(final, 4)
        s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ✅ 平仓结算 | P/L: {pnl:.2%} | 余额: ${s['bal']}")
        s['pos'], s['ent'], s['size'] = "NONE", 0.0, 0.0

# 5. 极简渲染 (彻底杜绝 NameError)
st.divider()
st.subheader(f"账户余额: ${s['bal'] if s['pos'] == 'NONE' else (s['size']*px):.4f}")

if s['logs']:
    with st.expander("交易审计日志", expanded=True):
        for log in reversed(s['logs']):
            st.code(log)

# 6. 强制原子刷新
time.sleep(1)
st.rerun()
