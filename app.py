import streamlit as st
import pandas as pd
import numpy as np
import httpx
import asyncio
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 模拟账户初始化 ====================
if 'balance' not in st.session_state:
    st.session_state.balance = 10000.0  # 初始资金 10,000 U
    st.session_state.equity_history = [10000.0] # 资产曲线历史
    st.session_state.trades = [] # 历史成交记录
    st.session_state.position = None # 当前持仓状态 {entry, type, sl, tp}

# ==================== 2. 模拟撮合引擎 ====================
def update_mock_account(current_price, signal_plan):
    pos = st.session_state.position
    # --- 判定平仓 (止盈或止损) ---
    if pos:
        is_close = False
        pnl = 0
        if pos['type'] == 'LONG':
            if current_price >= pos['tp'] or current_price <= pos['sl']:
                pnl = (current_price - pos['entry']) / pos['entry'] * st.session_state.balance * 0.1
                is_close = True
        elif pos['type'] == 'SHORT':
            if current_price <= pos['tp'] or current_price >= pos['sl']:
                pnl = (pos['entry'] - current_price) / pos['entry'] * st.session_state.balance * 0.1
                is_close = True
        
        if is_close:
            st.session_state.balance += pnl
            st.session_state.equity_history.append(st.session_state.balance)
            st.session_state.trades.append({"time": datetime.now(), "pnl": pnl, "final": st.session_state.balance})
            st.session_state.position = None
            st.toast(f"✅ 模拟平仓！收益: {pnl:.2f}U", icon="💰")

    # --- 判定开仓 ---
    if not st.session_state.position and signal_plan['entry']:
        pos_type = "LONG" if "多" in signal_plan['act'] else "SHORT" if "空" in signal_plan['act'] else None
        if pos_type:
            st.session_state.position = {
                "entry": current_price,
                "type": pos_type,
                "tp": signal_plan['tp'],
                "sl": signal_plan['sl']
            }
            st.toast(f"🚀 模拟开仓: {pos_type} @{current_price:.2f}", icon="🔥")

# ==================== 3. 渲染 V270 终极 UI ====================
st.set_page_config(page_title="ETH V270 模拟实盘终端", layout="wide")

# (此处省略之前的 fetch_enhanced_data 等异步抓取逻辑...)
# 假设已经拿到了 df1, plan, sentiment 等

st.title("🛡️ ETH 战神 V270 · 模拟实盘终端")

# 第一排：账户核心指标
c1, c2, c3, c4 = st.columns(4)
current_equity = st.session_state.balance
initial_equity = 10000.0
total_pnl = current_equity - initial_equity
pnl_pct = (total_pnl / initial_equity) * 100

c1.metric("账户净值 (Equity)", f"${current_equity:.2f}U", f"{pnl_pct:+.2f}%")
c2.metric("当前仓位", "空仓" if not st.session_state.position else st.session_state.position['type'])
c3.metric("累计盈亏", f"${total_pnl:.2f}U")
c4.metric("交易笔数", len(st.session_state.trades))

# 自动运行账户撮合
update_mock_account(df1['c'].iloc[-1], plan)

st.markdown("---")
l, r = st.columns([1.5, 2])

with l:
    # 权益曲线绘制
    st.subheader("📈 资产净值曲线 (Equity Curve)")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        y=st.session_state.equity_history, 
        mode='lines+markers',
        line=dict(color='#00FFCC', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 204, 0.1)'
    ))
    fig_equity.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_equity, use_container_width=True)

    # 历史成交清单
    if st.session_state.trades:
        st.subheader("📜 历史成交")
        st.dataframe(pd.DataFrame(st.session_state.trades).tail(5), use_container_width=True)

with r:
    # 实时信号图表
    st.subheader("📊 实时裁决监控")
    fig_k = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'])])
    if st.session_state.position:
        fig_k.add_hline(y=st.session_state.position['entry'], line_color="white", annotation_text="入场成本")
    fig_k.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_k, use_container_width=True)

st.info(f"💡 指挥部提示：当前账户使用 10% 保证金比例进行模拟。行情波动率 ATR: {df1['atr'].iloc[-1]:.2f}")
