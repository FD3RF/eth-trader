import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V62.0 终极全功能终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 计算引擎：战绩/模式/转向 ====================
def master_engine(df):
    # A. 计算基础指标
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['atr'] = (df['h'] - df['l']).rolling(12).mean()
    
    # B. 24H 模拟回测逻辑 (用于计算胜率和复盘标记)
    lookback = 288
    recent_df = df.tail(lookback).copy()
    success, total = 0, 0
    replay_data = {'buy': [], 'sell': []}
    
    for i in range(20, len(recent_df) - 6):
        if recent_df['rsi'].iloc[i] < 35: # 抄底逻辑
            total += 1
            entry_p = recent_df['c'].iloc[i]
            future = recent_df.iloc[i+1 : i+7]
            if future['h'].max() > entry_p * 1.005:
                success += 1
                replay_data['buy'].append((recent_df.index[i], entry_p))
                replay_data['sell'].append((future['h'].idxmax(), future['h'].max()))
                
    win_rate = (success / total * 100) if total > 0 else 0
    
    # C. 模式识别
    current_atr = df['atr'].iloc[-1]
    market_mode = "🌀 震荡模式" if current_atr < 8.5 else "📊 趋势模式"
    
    return win_rate, total, market_mode, replay_data

# ==================== 3. 智能决策中心 (带熔断与转向) ====================
def generate_smart_plan(curr_p, sup, res, win_rate, net_flow):
    # 策略常量
    PANIC_ZONE = 35.0  # 触发转向的胜率阈值
    
    if win_rate < PANIC_ZONE:
        # --- 空头转向逻辑 ---
        return {
            "title": "🔴 空头转向模式 (Trend Flip)",
            "color": "#ff4b4b",
            "reason": f"多头胜率过低({win_rate:.1f}%)，当前阴跌动能极强，已自动反转为『逢高做空』。",
            "entry": res if res else curr_p * 1.002,
            "sl": (res if res else curr_p) + 8,
            "tp": sup if sup else curr_p - 15,
            "type": "SHORT"
        }
    elif net_flow < -15:
        # --- 资金流保护熔断 ---
        return {"title": "🟡 避险锁定 (资金流出)", "color": "#ffd700", "reason": "资金正大幅流出，抄底风险极高，请待机。", "entry":0,"sl":0,"tp":0, "type":"WAIT"}
    else:
        # --- 标准多头逻辑 ---
        return {
            "title": "🟢 逢低买入 (标准模式)",
            "color": "#00ffcc",
            "reason": "胜率与资金流健康，建议在支撑位附近布局波段多单。",
            "entry": curr_p if curr_p < (sup+2 if sup else curr_p) else sup,
            "sl": (sup if sup else curr_p) - 8,
            "tp": res if res else curr_p + 15,
            "type": "LONG"
        }

# ==================== 4. UI 与 渲染层 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=400")
d_raw = fetch_okx_data("market/books", "&sz=20")
t_raw = fetch_okx_data("market/trades", "&limit=100")

if k_raw and d_raw:
    # 数据格式化
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    curr_p = df.iloc[-1]['c']
    
    # 引擎运算
    wr, total_signals, mode, replay = master_engine(df)
    
    # 侧边栏
    with st.sidebar:
        st.header("📊 战绩与策略引擎")
        st.metric("24H 模拟胜率", f"{wr:.1f}%", delta=f"{total_signals}个信号")
        show_replay = st.toggle("🔍 开启一键复盘标记", value=False)
        st.divider()
        
        # 计算辅助指标
        asks = pd.DataFrame(d_raw['data'][0]['asks']).iloc[:, :2].astype(float)
        bids = pd.DataFrame(d_raw['data'][0]['bids']).iloc[:, :2].astype(float)
        res_p = asks[asks[1] > asks[1].mean() * 1.8].iloc[0, 0] if not asks.empty else None
        sup_p = bids[bids[1] > bids[1].mean() * 1.8].iloc[0, 0] if not bids.empty else None
        tdf = pd.DataFrame(t_raw['data'])
        net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
        
        # 生成决策计划
        plan = generate_smart_plan(curr_p, sup_p, res_p, wr, net_f)
        
        # 决策卡片渲染
        st.markdown(f"""
            <div style="border:3px solid {plan['color']}; padding:15px; border-radius:10px; background:rgba(0,0,0,0.3)">
                <h2 style="margin:0; color:{plan['color']}">{plan['title']}</h2>
                <p style="margin:10px 0; font-size:14px; line-height:1.4">{plan['reason']}</p>
                {f'<hr><p><b>计划入场：</b>${plan["entry"]:.2f}</p><p><b>强制止损：</b>${plan["sl"]:.2f}</p><p style="color:#00ffcc"><b>目标盈利：</b>${plan["tp"]:.2f}</p>' if plan['entry'] > 0 else ''}
            </div>
        """, unsafe_allow_html=True)

    # 主界面
    st.title(f"🛡️ ETH {mode} 终端 V62.0")
    col1, col2, col3 = st.columns(3)
    col1.metric("当前现价", f"${curr_p}")
    col2.metric("1min 净流入", f"{net_f:+.2f} ETH")
    col3.metric("买压占比", f"{(bids[1].sum()/(asks[1].sum()+bids[1].sum())*100):.1f}%")

    # 绘图层
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线")])
    
    if show_replay:
        if replay['buy']:
            b_idx, b_val = zip(*replay['buy'])
            fig.add_trace(go.Scatter(x=b_idx, y=[v*0.998 for v in b_val], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00ffcc'), name="回测买入"))
        if replay['sell']:
            s_idx, s_val = zip(*replay['sell'])
            fig.add_trace(go.Scatter(x=s_idx, y=[v*1.002 for v in s_val], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff00ff'), name="回测止盈"))

    if res_p: fig.add_hline(y=res_p, line_dash="dash", line_color="red", annotation_text="压力/做空区")
    if sup_p: fig.add_hline(y=sup_p, line_dash="dash", line_color="green", annotation_text="支撑/做多区")
    
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("正在集成所有模块... 请确保 LetsVPN 已连接。")
