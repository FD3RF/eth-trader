import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 核心逻辑优化：缩量判定与动态止损
# ==========================================
def apply_warrior_sniper_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)

    # --- 优化1：缩量观察条件 (Pre-Signal Filter) ---
    # 检查前3根K线是否出现量能萎缩 (缩量是提醒)
    df['is_shrinking'] = (df['v'].shift(1) < df['v'].shift(2)) & (df['v'].shift(2) < df['v'].shift(3))
    
    # 信号触发：放量突破 + 前置缩量确认
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']

    # 局部锚点锁定
    window = df.tail(30)
    local_down = window[window['c'] < window['o']].nlargest(1, 'v')
    local_up = window[window['c'] > window['o']].nlargest(1, 'v')
    
    anchors = {
        'upper': local_down['h'].values[0] if not local_down.empty else window['h'].max(),
        'lower': local_up['l'].values[0] if not local_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 2. 动态进场计划与移动止损建议
# ==========================================
def render_sniper_plan(curr, anchors, p):
    is_buy = curr['buy_sig']
    entry_price = curr['c']
    initial_sl = anchors['lower'] if is_buy else anchors['upper']
    
    # 计算盈亏比区间
    risk = abs(entry_price - initial_sl)
    target_1_1 = entry_price + risk if is_buy else entry_price - risk
    final_tp = entry_price + risk * p['rr_ratio'] if is_buy else entry_price - risk * p['rr_ratio']

    # --- 优化2：移动止损逻辑提示 (Trailing Logic) ---
    # 当价格触及 1:1 盈亏比时，建议保本
    st.markdown(f"""<div style='background:#11141c; border:1px solid #444; padding:15px; border-radius:10px;'>
        <h3 style='color:{"#26a69a" if is_buy else "#ef5350"};'>🎯 狙击手计划执行中</h3>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:16px;'>
            <div><b>初始止损:</b> ${initial_sl:.2f}</div>
            <div><b>保本触发点 (1:1):</b> ${target_1_1:.2f}</div>
            <div><b>最终目标:</b> ${final_tp:.2f}</div>
        </div>
        <p style='color:#ffa500; margin-top:10px; font-weight:bold;'>⚠️ 移动止损策略：价格触及 ${target_1_1:.2f} 后，请立即将止损移至开仓价 ${entry_price:.2f} 锁定风险！</p>
    </div>""", unsafe_allow_html=True)
