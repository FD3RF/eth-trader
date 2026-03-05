import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# ==========================================
# 1. 冷却系统状态管理
# ==========================================
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = []  # 记录格式: (timestamp, result_type['win', 'loss'])
if 'cooldown_until' not in st.session_state:
    st.session_state.cooldown_until = None

def check_cooldown():
    """执行口诀：如果连续止损两单，停下来观察。"""
    if st.session_state.cooldown_until and datetime.datetime.now() < st.session_state.cooldown_until:
        return True
    
    # 检查最近1小时内的最后两笔记录
    recent_losses = [log for log in st.session_state.trade_logs 
                     if log['result'] == 'loss' and 
                     (datetime.datetime.now() - log['time']).total_seconds() < 3600]
    
    if len(recent_losses) >= 2:
        st.session_state.cooldown_until = datetime.datetime.now() + datetime.timedelta(minutes=60)
        return True
    return False

# ==========================================
# 2. 核心逻辑：盈亏比雷达 (Risk-Reward Radar)
# ==========================================
def calculate_rr_ratio(side, price, p_high, p_low):
    """口诀：位置尴尬，空间不足，观望。"""
    if side == "LONG":
        risk = price - p_low
        reward = p_high - price
    else:
        risk = p_high - price
        reward = price - p_low
    
    rr = reward / risk if risk > 0 else 0
    return rr

# ==========================================
# 3. UI 渲染：熔断保护界面
# ==========================================
@st.fragment(run_every="5s")
def render_v5_5_iron_rule():
    is_locked = check_cooldown()
    
    if is_locked:
        remaining = (st.session_state.cooldown_until - datetime.datetime.now()).total_seconds()
        st.error(f"⚠️ 触发物理熔断：连续止损两单。系统锁定中，请强制执行口诀「停下来观察」。")
        st.metric("冷却倒计时", f"{int(remaining//60)}分{int(remaining%60)}秒")
        st.sidebar.warning("🚫 交易指令已锁定")
        # 即使锁定，行情依然更新，但操作面板消失
    
    # ... (承接 V5.4 的 Core 数据获取与逻辑) ...
    # 假设此处已获取 df, res (包含 p_high, p_low, sig)
    
    # 动态盈亏比校验
    if not is_locked and res['sig']:
        rr = calculate_rr_ratio(res['sig'], curr['c'], res['p_high'], res['p_low'])
        
        with st.container(border=True):
            st.markdown(f"### 🛡️ 铁律校验 (盈亏比: {rr:.2f})")
            if rr >= 1.5:
                st.success(f"✅ 空间充足（>{1.5}），准许执行口诀：{res['msg']}")
                if st.button("🚀 确认开仓并同步止损红线", use_container_width=True):
                    # 模拟记录
                    st.session_state.trade_logs.append({'time': datetime.datetime.now(), 'result': 'loss'}) # 演示用
            else:
                st.warning(f"❌ 空间不足（<{1.5}），拒绝执行。口诀：位置尴尬，只看不动。")

    # --- K线图增加止损红线 ---
    # 在 Plotly 中，根据当前计划动态画出三色激光线
    # (此处省略重复的绘图代码，重点在于 add_hline 的颜色区分)
