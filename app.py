import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 物理层：环境硬锁与 Session 持久化
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 终极量子雷达", page_icon="⚡")

def hard_lock_environment():
    """锁定 SessionState 变量，防止刷新丢失日志与情绪数据"""
    initial_states = {
        'df': pd.DataFrame(),
        'battle_logs': [],
        'prev_bids': pd.DataFrame(),
        'prev_asks': pd.DataFrame(),
        'sentiment_score': 50.0,
        'ls_ratio': 1.0
    }
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

hard_lock_environment()

# ==========================================
# 2. 侧边栏：量子参数控制与复盘导出
# ==========================================
st.sidebar.markdown("### 🛸 量子实时控制")
auto_refresh = st.sidebar.toggle("开启全球全维度监控", value=True)
refresh_rate = st.sidebar.slider("心跳频率 (秒)", 5, 60, 10)

st.sidebar.divider()
ema_f_val = st.sidebar.number_input("快线 EMA", 5, 30, 12)
ema_s_val = st.sidebar.number_input("慢线 EMA", 20, 100, 26)

# --- 一键复盘模块 ---
st.sidebar.divider()
st.sidebar.markdown("### 📊 战役复盘中心")
if st.sidebar.button("💾 生成今日战术报告"):
    if st.session_state.battle_logs:
        report_data = []
        for log in st.session_state.battle_logs:
            l_type = "爆仓" if "💀" in log else "撤单" if "⚠️" in log else "系统"
            report_data.append({
                "时间": log.split("】")[0].replace("【", ""),
                "类型": l_type,
                "详情": log.split("】")[-1].strip(),
                "多空比快照": st.session_state.ls_ratio,
                "情绪指数": st.session_state.sentiment_score
            })
        csv = pd.DataFrame(report_data).to_csv(index=False).encode('utf-8-sig')
        st.sidebar.download_button("📥 点击下载 CSV 报告", data=csv, file_name="ETH_Tactical_Report.csv", mime='text/csv')
    else:
        st.sidebar.error("暂无日志数据")

# ==========================================
# 3. 影子引擎：多源数据采集
# ==========================================
@st.cache_data(ttl=refresh_rate)
def get_quantum_data():
    """获取K线、多空比数据"""
    try:
        k_url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        kr = requests.get(k_url, timeout=5).json()
        df = pd.DataFrame(kr['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df = df.reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 指标计算
        df['ema_f'] = df['c'].ewm(span=ema_f_val, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=ema_s_val, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        
        # 全网多空比
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        lsr = requests.get(ls_url, timeout=5).json()
        st.session_state.ls_ratio = float(lsr['data'][0][1]) if lsr.get('code') == '0' else 1.0
        return df
    except:
        return st.session_state.df

@st.cache_data(ttl=2)
def get_live_combat():
    """盘口深度与爆仓单抓取"""
    try:
        d_url = "https://www.okx.com/api/v5/market/books?instId=ETH-USDT&sz=20"
        l_url = "https://www.okx.com/api/v5/public/liquidation-orders?instId=ETH-USDT-SWAP&mgnMode=cross"
        dr = requests.get(d_url, timeout=3).json()
        lr = requests.get(l_url, timeout=3).json()
        bids = pd.DataFrame(dr['data'][0]['bids'], columns=['px','sz','l','c']).astype(float)
        asks = pd.DataFrame(dr['data'][0]['asks'], columns=['px','sz','l','c']).astype(float)
        return bids, asks, lr['data'][0]['details'] if lr.get('code') == '0' else []
    except:
        return None, None, []

# ==========================================
# 4. 逻辑层：AI 胜率预测与战术研判
# ==========================================
def calculate_ai_win_rate(df, ratio, score):
    """基于清算、持仓、意图三位一体计算胜率"""
    last = df.iloc[-1]
    prob = 50.0
    if last['ema_f'] > last['ema_s']: prob += 10
    if ratio < 0.9: prob += 15
    elif ratio > 1.3: prob -= 15
    prob += (score - 50) * 0.4
    if any("💀" in str(l) for l in st.session_state.battle_logs[:3]): prob += 5
    return max(min(prob, 99.0), 1.0)

def process_combat_logic(bids, asks, liqs):
    now = datetime.now().strftime('%H:%M:%S')
    # 爆仓处理
    for l in liqs[:2]:
        sz = float(l['sz'])
        if sz > 30:
            side = "多头" if l['posSide'] == 'long' else "空头"
            msg = f"💀 [清算] {side}爆仓! 规模: {sz:.1f} ETH | 价位: ${l['bkPx']}"
            if msg not in st.session_state.battle_logs[:5]:
                st.session_state.battle_logs.insert(0, f"【{now}】{msg}")
                st.session_state.sentiment_score += (15 if side == "空头" else -15)
    # 撤单侦测
    if bids is not None and not st.session_state.prev_bids.empty:
        th = (bids['sz'].mean() + asks['sz'].mean()) * 2.5
        for _, p in st.session_state.prev_bids.iterrows():
            if p['sz'] > th and bids[bids['px'] == p['px']].empty:
                st.session_state.battle_logs.insert(0, f"【{now}】⚠️ 诱多撤单: ${p['px']}")
                st.session_state.sentiment_score -= 5
        for _, p in st.session_state.prev_asks.iterrows():
            if p['sz'] > th and asks[asks['px'] == p['px']].empty:
                st.session_state.battle_logs.insert(0, f"【{now}】⚠️ 诱空撤单: ${p['px']}")
                st.session_state.sentiment_score += 5
    st.session_state.prev_bids, st.session_state.prev_asks = bids, asks
    st.session_state.sentiment_score = max(min(st.session_state.sentiment_score, 100), 0)

# ==========================================
# 5. UI 渲染引擎
# ==========================================
def main():
    df = get_quantum_data()
    bids, asks, liqs = get_live_combat()
    process_combat_logic(bids, asks, liqs)
    
    score = st.session_state.sentiment_score
    ratio = st.session_state.ls_ratio
    win_prob = calculate_ai_win_rate(df, ratio, score)
    
    # 顶部状态栏
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    st.progress(score / 100)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("庄家情绪", f"{score:.1f}", f"{'看多' if score > 50 else '看空'}")
    with c2: st.metric("全网多空比", f"{ratio}", f"{'散户看多' if ratio > 1 else '散户看空'}", delta_color="inverse")
    with c3: st.metric("AI 进攻胜率", f"{win_prob:.1f}%", delta=f"{win_prob-50:+.1f}%")

    st.divider()

    # 主战场布局
    col_l, col_r = st.columns([1, 4])
    with col_l:
        last = df.iloc[-1]
        all_in = (last['ema_f'] > last['ema_s']) and (ratio < 1.1)
        box_c = "#FFD700" if all_in else "#FF00FF"
        st.markdown(f"""<div style="border:2px solid {box_c}; padding:15px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.5); box-shadow: 0 0 15px {box_c};">
            <h2 style="color:{box_c}; margin:0;">{'🔥 黄金总攻' if all_in else '🔒 AI 猎杀'}</h2>
            <p style="color:{box_c}; font-size:1.5em; font-weight:bold; margin:10px 0;">{win_prob:.1f}%</p>
            <p style="color:#888; font-size:0.7em;">共振胜率预测</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("#### 📜 战术日志")
        for log in st.session_state.battle_logs[:10]:
            if "💀" in log: st.warning(log)
            elif "⚠️" in log: st.error(log)
            else: st.caption(log)

    with col_r:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        if bids is not None:
            fig.add_trace(go.Bar(x=bids['px'], y=bids['sz'], name='买单墙', marker_color='green', opacity=0.3), row=2, col=1)
            fig.add_trace(go.Bar(x=asks['px'], y=asks['sz'], name='卖单墙', marker_color='red', opacity=0.3), row=2, col=1)
        
        flow_c = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_c, name="净流"), row=3, col=1)
        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
