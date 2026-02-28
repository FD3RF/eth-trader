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
    """确保在 Streamlit Cloud 或本地运行时，核心战术数据不会因重绘丢失"""
    if 'battle_logs' not in st.session_state:
        st.session_state.battle_logs = []
    if 'sentiment_score' not in st.session_state:
        st.session_state.sentiment_score = 50.0
    if 'prev_bids' not in st.session_state:
        st.session_state.prev_bids = pd.DataFrame()
    if 'ls_ratio' not in st.session_state:
        st.session_state.ls_ratio = 1.0

hard_lock_environment()

# ==========================================
# 2. 侧边栏：量子控制与战役复盘中心
# ==========================================
st.sidebar.markdown("### 🛸 量子实时控制")
auto_refresh = st.sidebar.toggle("开启全球全维度监控", value=True)
refresh_rate = st.sidebar.slider("心跳频率 (秒)", 5, 60, 10)

st.sidebar.divider()
ema_f_val = st.sidebar.number_input("快线 EMA", 5, 30, 12)
ema_s_val = st.sidebar.number_input("慢线 EMA", 20, 100, 26)

st.sidebar.divider()
st.sidebar.markdown("### 📊 战役复盘中心")
if st.sidebar.button("💾 生成今日战术报告"):
    if st.session_state.battle_logs:
        # 将日志解析为结构化数据用于导出
        report_list = []
        for log in st.session_state.battle_logs:
            tag = "爆仓" if "💀" in log else "撤单" if "⚠️" in log else "情报"
            report_list.append({
                "时间戳": log.split("】")[0].replace("【", ""),
                "事件类型": tag,
                "战术详情": log.split("】")[-1].strip(),
                "多空比快照": st.session_state.ls_ratio,
                "庄家情绪": st.session_state.sentiment_score
            })
        report_df = pd.DataFrame(report_list)
        csv = report_df.to_csv(index=False).encode('utf-8-sig')
        st.sidebar.download_button("📥 点击下载复盘 CSV", data=csv, file_name=f"ETH_Combat_{datetime.now().strftime('%Y%m%d')}.csv")
        st.sidebar.success("复盘引擎准备就绪")
    else:
        st.sidebar.error("日志区尚无战斗数据")

# ==========================================
# 3. 数据引擎：多源情报采集
# ==========================================
@st.cache_data(ttl=refresh_rate)
def fetch_global_intel():
    """获取 K线指标与全网多空持仓比"""
    try:
        # 1. 抓取 ETH K线
        k_url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
        kr = requests.get(k_url, timeout=5).json()
        df = pd.DataFrame(kr['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df = df.reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 指标影子计算
        df['ema_f'] = df['c'].ewm(span=ema_f_val, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=ema_s_val, adjust=False).mean()
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.15).rolling(5).sum().fillna(0)
        
        # 2. 抓取多空持仓比 (反向博弈核心)
        ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId=ETH-USDT&period=5m"
        lsr = requests.get(ls_url, timeout=5).json()
        if lsr.get('code') == '0':
            st.session_state.ls_ratio = float(lsr['data'][0][1])
            
        return df
    except:
        return st.session_state.df

@st.cache_data(ttl=2)
def fetch_depth_and_liqs():
    """高频抓取盘口墙与爆仓情报"""
    try:
        d_url = "https://www.okx.com/api/v5/market/books?instId=ETH-USDT&sz=20"
        l_url = "https://www.okx.com/api/v5/public/liquidation-orders?instId=ETH-USDT-SWAP&mgnMode=cross"
        dr = requests.get(d_url, timeout=3).json()
        lr = requests.get(l_url, timeout=3).json()
        bids = pd.DataFrame(dr['data'][0]['bids'], columns=['px','sz','l','c']).astype(float)
        asks = pd.DataFrame(dr['data'][0]['asks'], columns=['px','sz','l','c']).astype(float)
        liqs = lr['data'][0]['details'] if lr.get('code') == '0' else []
        return bids, asks, liqs
    except:
        return None, None, []

# ==========================================
# 4. 逻辑层：AI 胜率计算与撤单研判
# ==========================================
def calculate_win_rate(df, ratio, score):
    """
    进攻胜率 AI 模型 (Win-Rate AI)
    逻辑：EMA金叉(10%) + 散户看空(15%) + 庄家拉升情绪(20%) + 近期清算(5%)
    """
    last = df.iloc[-1]
    prob = 50.0
    # 趋势分
    if last['ema_f'] > last['ema_s']: prob += 10
    # 对手盘分 (Ratio < 1.0 说明散户在做空，有利于拉升)
    if ratio < 0.95: prob += 15
    elif ratio > 1.25: prob -= 15
    # 庄家分
    prob += (score - 50) * 0.4
    # 清算动力分
    if any("💀" in str(l) for l in st.session_state.battle_logs[:3]): prob += 5
    
    return max(min(prob, 99.0), 1.0)

def run_tactical_logic(bids, asks, liqs):
    now = datetime.now().strftime('%H:%M:%S')
    
    # 1. 清算研判 (血流成河)
    for l in liqs[:2]:
        sz = float(l['sz'])
        if sz > 40: # 门槛：40 ETH
            side = "多头" if l['posSide'] == 'long' else "空头"
            msg = f"💀 [血流成河] {side}大额爆仓! 规模: {sz:.1f} ETH | 价位: ${l['bkPx']}"
            if msg not in st.session_state.battle_logs[:5]:
                st.session_state.battle_logs.insert(0, f"【{now}】{msg}")
                st.session_state.sentiment_score += (12 if side == "空头" else -12)

    # 2. 撤单研判 (诱多/诱空)
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
# 5. UI 渲染：量子指挥部
# ==========================================
def main():
    df = fetch_global_intel()
    bids, asks, liqs = fetch_depth_and_liqs()
    run_tactical_logic(bids, asks, liqs)
    
    score = st.session_state.sentiment_score
    ratio = st.session_state.ls_ratio
    win_prob = calculate_win_rate(df, ratio, score)
    
    # --- 状态看板 ---
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    st.progress(score / 100)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("庄家情绪", f"{score:.1f}", f"{'看多' if score > 50 else '看空'}")
    m2.metric("全网多空比", f"{ratio}", f"{'散户看空' if ratio < 1 else '散户看多'}", delta_color="inverse")
    m3.metric("AI 进攻胜率", f"{win_prob:.1f}%", delta=f"{win_prob-50:+.1f}%")

    st.divider()

    # --- 战场双栏布局 ---
    col_l, col_r = st.columns([1, 4])
    
    with col_l:
        # AI 猎杀可视化 (匹配截图 7 风格)
        last = df.iloc[-1]
        active_signal = (last['ema_f'] > last['ema_s']) and (ratio < 1.1)
        box_c = "#00ff00" if active_signal else "#FF00FF"
        st.markdown(f"""
            <div style="border:2px solid {box_c}; padding:15px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.5); box-shadow: 0 0 15px {box_c};">
                <h2 style="color:{box_c}; margin:0;">{'🔥 黄金总攻' if active_signal else '🔒 AI 猎杀'}</h2>
                <p style="color:{box_c}; font-size:1.8em; font-weight:bold; margin:10px 0;">{win_prob:.1f}%</p>
                <p style="color:#888; font-size:0.75em;">共振胜率预判</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📜 实时战术日志")
        for log in st.session_state.battle_logs[:12]:
            if "💀" in log: st.warning(log)
            elif "⚠️" in log: st.error(log)
            else: st.caption(log)

    with col_r:
        # 物理图表引擎 (K线 + 深度墙 + 净流)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
        
        # 1. K线图层
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        
        # 2. 深度墙层 (匹配截图 2/3/4)
        if bids is not None:
            fig.add_trace(go.Bar(x=bids['px'], y=bids['sz'], name='买单墙', marker_color='green', opacity=0.3), row=2, col=1)
            fig.add_trace(go.Bar(x=asks['px'], y=asks['sz'], name='卖单墙', marker_color='red', opacity=0.3), row=2, col=1)
        
        # 3. 盘口净流层 (动态变色)
        flow_colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="净流"), row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=780, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 驱动主循环
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
