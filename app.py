import streamlit as st
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

# 1. 基础配置：物理隔离，禁止任何旧参数
st.set_page_config(layout="wide", page_title="FINAL_ETH_V13")

# 2. 状态机：锁死在唯一字典中，拒绝任何外部变量引用
if 'S' not in st.session_state:
    st.session_state['S'] = {
        'bal': 10.0, 'pos': "NONE", 'size': 0.0, 'ent': 0.0,
        'logs': [], 'curve': [10.0], 'px': 2500.0
    }

# 3. 极简抓取：不加缓存，实时物理连接
def get_px():
    try:
        # 直接拉取，失败就跳过，绝不卡死
        r = requests.get("https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT-SWAP", timeout=1).json()
        px = float(r['data'][0]['last'])
        st.session_state['S']['px'] = px
        return px
    except:
        return st.session_state['S']['px']

# 4. 核心逻辑执行
px = get_px()
s = st.session_state['S']

# 模拟高胜率点位因子 (不再使用容易出错的复杂函数)
skew = np.random.uniform(-30, 30)
press = np.random.uniform(10, 90)
sent = 75.30 + np.random.uniform(-0.0001, 0.0001)

# 进场判据：奇点脉冲 + 极度失衡
pulse = abs(sent - 75.30) < 0.00008
is_l = pulse and skew < -22 and press > 82
is_s = pulse and skew > 22 and press < 18

if s['pos'] == "NONE":
    if is_l:
        s['ent'], s['pos'] = px, "LONG"
        s['size'] = (s['bal'] * 0.99) / px
        s['bal'] = 0.0
        s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | 🚀 做多入场 @ {px}")
    elif is_s:
        s['ent'], s['pos'] = px, "SHORT"
        s['size'] = (s['bal'] * 0.99) / px
        s['bal'] = 0.0
        s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ⚡ 做空入场 @ {px}")
else:
    # 盈亏计算
    pnl = (px/s['ent']-1) if s['pos']=="LONG" else (1-px/s['ent'])
    if pnl > 0.015 or pnl < -0.0075:
        # 结算
        val = s['size'] * s['ent'] * (1 + pnl) * 0.999 # 扣除滑点
        s['bal'] = round(val, 4)
        s['logs'].append(f"{datetime.now().strftime('%H:%M:%S')} | ✅ 平仓 P/L: {pnl:.2%}")
        s['pos'], s['size'] = "NONE", 0.0

# 5. UI 布局：完全弃用变量解构，使用物理列表索引 (解决图1的报错)
cols = st.columns([1, 3])

# 左边栏：数据监控
cols[0].metric("10U 账户余额", f"${(s['bal'] + (s['size']*px if s['pos']!='NONE' else 0)):.4f}")
cols[0].write(f"当前价格: `{px}`")
cols[0].write(f"IV Skew: `{skew:.2f}%`")

if is_l: cols[0].success("多头共振")
elif is_s: cols[0].error("空头共振")
else: cols[0].info("正在扫描奇点...")

# 右侧栏：图表与日志
s['curve'].append(s['bal'] + (s['size']*px if s['pos']!='NONE' else 0))
if len(s['curve']) > 100: s['curve'].pop(0)

fig = go.Figure(go.Scatter(y=s['curve'], line=dict(color='#00ff88', width=2), fill='tozeroy'))
fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0))
cols[1].plotly_chart(fig, use_container_width=True)

with cols[1].expander("交易流水", expanded=True):
    for l in reversed(s['logs']): st.write(f"`{l}`")

# 6. 强制原子刷新
time.sleep(1)
st.rerun()
