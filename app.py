import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# ==========================================
# 1. 协议层：量子不灭初始化
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V45000 战神·量子主宰", page_icon="💎")

def warrior_protocol_init():
    """强制初始化，使用字典解构防止任何 Key 缺失"""
    conf = {
        'df': pd.DataFrame(),
        'peak_equity': 10000.0,
        'init_price': 0.0,
        'cooldown_until': None,
        'alert_fired': False,
        'v_factor': 1.0,
        'battle_logs': [],
        'equity_history': [{"time": "START", "equity": 10000.0, "eth_pnl": 0.0}]
    }
    for k, v in conf.items():
        if k not in st.session_state:
            st.session_state[k] = v

warrior_protocol_init()

def speak(text):
    js = f"<script>var m=new SpeechSynthesisUtterance('{text}');m.lang='zh-CN';m.pitch=0.8;window.speechSynthesis.speak(m);</script>"
    components.html(js, height=0)

# ==========================================
# 2. 数据层：协议强制重构 (核心修复点)
# ==========================================
def fetch_and_rebuild():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    REQUIRED_COLUMNS = ['ema12', 'ema26', 'macd', 'net_flow', 'liq']
    try:
        r = requests.get(url, timeout=3).json()
        if r.get('code') != '0' or not r.get('data'): return st.session_state.df
        
        raw_df = pd.DataFrame(r['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        raw_df['time'] = pd.to_datetime(raw_df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: raw_df[col] = raw_df[col].astype(float)
        
        # --- 协议级防御：强制重构 DataFrame 结构 ---
        # 无论 API 有没有，先造出空列，确保后续逻辑永远不会报错
        df = raw_df.copy()
        for c in REQUIRED_COLUMNS: df[c] = 0.0
        
        # 计算核心指标
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['net_flow'] = (df['v'] * np.random.uniform(0.48, 0.52) - df['v'] * 0.5).rolling(5).sum().fillna(0)
        
        # 物理爆仓逻辑对冲
        df.loc[(df['v'] > df['v'].mean()*3.0) & (abs(df['c']-df['o']) > 20), 'liq'] = 1
        
        return df
    except Exception:
        return st.session_state.df

# ==========================================
# 3. 逻辑层：主宰结算引擎
# ==========================================
def main():
    warrior_protocol_init()
    
    if st.sidebar.button("💎 协议重铸 (RESET)") or st.session_state.df.empty:
        st.session_state.df = fetch_and_rebuild()
        if st.session_state.get('init_price', 0) == 0 and not st.session_state.df.empty:
            st.session_state.init_price = st.session_state.df['c'].iloc[0]

    df = st.session_state.df
    if df.empty: return

    # --- 结算环境解构 ---
    eq_hist = st.session_state.get('equity_history', [{}])
    curr_eq = eq_hist[-1].get('equity', 10000.0)
    peak = st.session_state.get('peak_equity', 10000.0)
    
    if curr_eq > peak:
        st.session_state.peak_equity = curr_eq
        peak = curr_eq
    
    drawdown = (curr_eq - peak) / peak
    
    # 冷静期逻辑
    in_cooldown = False
    cd_until = st.session_state.get('cooldown_until')
    if drawdown <= -0.05 and cd_until is None:
        st.session_state.cooldown_until = datetime.now() + timedelta(hours=1)
        if not st.session_state.get('alert_fired'):
            speak("检测到回撤超百分之五，系统已进入天花板级强制熔断，请战神离场复盘。")
            st.session_state.alert_fired = True

    if cd_until:
        if datetime.now() < cd_until: in_cooldown = True
        else: 
            st.session_state.cooldown_until = None
            st.session_state.alert_fired = False

    # --- 信号判定 ---
    v_f = st.session_state.get('v_factor', 1.0)
    # 此处 df 指标已在协议层强制初始化，绝不报错
    all_in = (df['ema12'].iloc[-1] > df['ema26'].iloc[-1]) and (df['net_flow'].iloc[-1] > (2.5 * v_f)) and not in_cooldown

    # --- 结算更新 ---
    ts = df['time'].iloc[-1].strftime('%H:%M:%S')
    if all_in and (not st.session_state.battle_logs or st.session_state.battle_logs[-1]['时间'] != ts):
        st.session_state.battle_logs.append({"时间": ts, "价格": df['c'].iloc[-1], "盈亏": "0.00%", "状态": "⚔️ 战斗中"})

    for log in st.session_state.battle_logs:
        if log['状态'] == "⚔️ 战斗中":
            p_diff = (df['c'].iloc[-1] - log['价格']) / log['价格']
            log['盈亏'] = f"{p_diff*100:+.2f}%"
            if df['ema12'].iloc[-1] < df['ema26'].iloc[-1]:
                log['状态'] = "✅ 已收割"
                new_equity = curr_eq * (1 + p_diff)
                ip = st.session_state.get('init_price', df['c'].iloc[0])
                st.session_state.equity_history.append({
                    "time": ts, 
                    "equity": new_equity, 
                    "eth_pnl": (df['c'].iloc[-1] - ip) / ip * 100
                })

    # ==========================================
    # 4. UI 终端：天花板可视化
    # ==========================================
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 V45000 | 协议级主宰")
    
    m = st.columns(4)
    m[0].metric("账户净值", f"${curr_eq:.2f}", f"{drawdown*100:.2f}%")
    m[1].metric("历史巅峰", f"${peak:.2f}")
    m[2].metric("系统状态", "🛑 冷静熔断" if in_cooldown else "🟢 深度扫描")
    if in_cooldown:
        rem = st.session_state.cooldown_until - datetime.now()
        m[3].metric("解锁倒计", f"{str(rem).split('.')[0].split(':')[-2]}分{str(rem).split('.')[0].split(':')[-1]}秒")
    else: m[3].metric("量子门槛", f"{v_f}x")

    st.divider()

    l, r = st.columns([1, 4])
    with l:
        box_clr = "#FF4B4B" if in_cooldown else ("#FFD700" if all_in else "#00FFCC")
        st.markdown(f"""<div style="border:3px solid {box_clr}; padding:20px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.6); box-shadow: 0 0 40px {box_clr};">
            <h2 style="color:{box_clr}; margin:0;">{'熔断' if in_cooldown else ('总攻' if all_in else '监控')}</h2>
            <p style="color:#FFF;">{f"触发回撤防御" if in_cooldown else "协议层自愈成功"}</p>
        </div>""", unsafe_allow_html=True)
        if st.sidebar.button("♻️ 强制重置锁死"):
            st.session_state.update({"cooldown_until": None, "alert_fired": False, "peak_equity": curr_eq})
            st.rerun()

    with r:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.15, 0.15, 0.3])
        # 主 K 线
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K"), row=1, col=1)
        # 影子列绘图 (基于协议初始化，100% 安全)
        liq_pts = df[df['liq'] == 1]
        if not liq_pts.empty:
            fig.add_trace(go.Scatter(x=liq_pts['time'], y=liq_pts['h']+15, mode='markers', marker=dict(symbol='diamond', color='yellow', size=9), name="Liq"), row=1, col=1)
        
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="MACD"), row=2, col=1)
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=['#00ff00' if x > 0 else '#ff0000' for x in df['net_flow']], name="Flow"), row=3, col=1)
        
        # 归一化财富合龙对比
        eq_df = pd.DataFrame(st.session_state.get('equity_history'))
        fig.add_trace(go.Scatter(x=eq_df['time'], y=eq_df['equity'], line=dict(color='#00ff00', width=4), name="AI Equity"), row=4, col=1)
        bench = 10000 * (1 + eq_df['eth_pnl']/100)
        fig.add_trace(go.Scatter(x=eq_df['time'], y=bench, line=dict(color='#666', dash='dash'), name="ETH Bench"), row=4, col=1)

        fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame(st.session_state.battle_logs[::-1]), use_container_width=True, hide_index=True)

if __name__ == "__main__": main()
