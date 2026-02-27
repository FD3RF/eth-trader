import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==================== 1. 初始化配置与数据引擎 ====================
st.set_page_config(page_title="ETH V59.0 复盘终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 战绩回溯与买卖点标记逻辑 ====================
def get_backtest_results(df):
    """回溯过去 24 小时的模拟买卖点"""
    lookback = 288
    recent_df = df.tail(lookback).copy()
    
    # 计算内部 RSI 用于回测
    delta = recent_df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    recent_df['rsi_sim'] = 100 - (100 / (1 + gain/loss))
    
    buy_signals = []
    sell_signals = []
    success, total = 0, 0
    
    for i in range(20, len(recent_df) - 6):
        # 信号逻辑：RSI超卖 (震荡抄底逻辑)
        if recent_df['rsi_sim'].iloc[i] < 35:
            total += 1
            entry_idx = recent_df.index[i]
            entry_p = recent_df['c'].iloc[i]
            
            # 寻找后续 6 根 K 线内的最高点
            future_segment = recent_df.iloc[i+1 : i+7]
            max_p = future_segment['h'].max()
            max_idx = future_segment['h'].idxmax()
            
            if max_p > entry_p * 1.005: # 0.5% 止盈空间
                success += 1
                buy_signals.append((entry_idx, entry_p))
                sell_signals.append((max_idx, max_p))
                
    win_rate = (success / total * 100) if total > 0 else 0
    return total, win_rate, buy_signals, sell_signals

# ==================== 3. 指令生成引擎 ====================
def get_trade_plan(curr_p, sup, res, mode, net_flow):
    if net_flow < -15:
        action, color = "🟡 观望 (大单抛售)", "#ffd700"
    elif "震荡" in mode and curr_p < (sup + 2 if sup else curr_p):
        action, color = "🟢 逢低买入", "#00ffcc"
    elif "趋势" in mode and net_flow > 25:
        action, color = "🚀 顺势追多", "#00ffcc"
    else:
        action, color = "⚪ 待机中", "#ffffff"
    return {"action": action, "color": color, "entry": curr_p, "sl": sup-2 if sup else curr_p-10, "tp": res if res else curr_p+15}

# ==================== 4. 主逻辑渲染 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=400")
d_raw = fetch_okx_data("market/books", "&sz=20")
t_raw = fetch_okx_data("market/trades", "&limit=100")

if k_raw and d_raw:
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    curr_p = df.iloc[-1]['c']
    
    # 模式识别
    tr = (df['h']-df['l']).rolling(12).mean().iloc[-1]
    s_mode = "🌀 震荡模式" if tr < 8.5 else "📊 趋势模式"
    m_factor = 1.6 if "震荡" in s_mode else 3.0
    
    # 战绩与信号
    total_s, wr, buys, sells = get_backtest_results(df)
    
    with st.sidebar:
        st.header("📈 战绩与复盘")
        st.metric("24H 胜率", f"{wr:.1f}%", delta=f"{total_s}次信号")
        
        # --- 一键复盘开关 ---
        show_replay = st.toggle("🔍 开启一键复盘", value=False, help="在图表上显示过去24小时成功的买卖点")
        
        st.divider()
        st.header("🎯 实时指令")
        
        # 盘口计算
        asks = pd.DataFrame(d_raw['data'][0]['asks']).iloc[:, :2].astype(float)
        bids = pd.DataFrame(d_raw['data'][0]['bids']).iloc[:, :2].astype(float)
        res_p = asks[asks[1] > asks[1].mean() * m_factor].iloc[0, 0] if not asks.empty else None
        sup_p = bids[bids[1] > bids[1].mean() * m_factor].iloc[0, 0] if not bids.empty else None
        tdf = pd.DataFrame(t_raw['data'])
        net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
        
        plan = get_trade_plan(curr_p, sup_p, res_p, s_mode, net_f)
        st.markdown(f"""<div style="border:2px solid {plan['color']}; padding:15px; border-radius:10px; background:rgba(0,0,0,0.2)">
            <h3 style="margin:0; color:{plan['color']}">{plan['action']}</h3>
            <p style="margin:5px 0">进场: ${plan['entry']:.2f} | 止损: ${plan['sl']:.2f}</p>
            <p style="margin:5px 0; color:#00ffcc">目标止盈: ${plan['tp']:.2f}</p>
        </div>""", unsafe_allow_html=True)

    st.title(f"🛡️ ETH {s_mode} 终端 V59.0")
    
    # K 线图渲染
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price")])
    
    # 如果开启复盘，绘制买卖点
    if show_replay:
        if buys:
            b_idx, b_val = zip(*buys)
            fig.add_trace(go.Scatter(x=b_idx, y=[v*0.998 for v in b_val], mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ffcc'), name="模拟买入"))
        if sells:
            s_idx, s_val = zip(*s_signals) if 's_signals' in locals() else zip(*sells)
            fig.add_trace(go.Scatter(x=s_idx, y=[v*1.002 for v in s_val], mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff00ff'), name="模拟止盈"))

    if res_p: fig.add_hline(y=res_p, line_dash="dash", line_color="red")
    if sup_p: fig.add_hline(y=sup_p, line_dash="dash", line_color="green")
    
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("数据引擎唤醒中...")
