import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==========================================
# 1. 物理层：生命周期硬锁
# ==========================================
st.set_page_config(layout="wide", page_title="ETH V32000 量子终结者", page_icon="⚡")

def hard_lock_environment():
    """彻底锁定 SessionState 变量，防止云端重启导致日志和情绪丢失"""
    initial_states = {
        'df': pd.DataFrame(),
        'battle_logs': [],
        'last_signal_ts': None,
        'prev_bids': pd.DataFrame(),
        'prev_asks': pd.DataFrame(),
        'sentiment_score': 50.0,
        'wall_history': []
    }
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

hard_lock_environment()

# ==========================================
# 2. 侧边栏：量子参数控制
# ==========================================
st.sidebar.markdown("### 🛸 量子实时控制")
auto_refresh = st.sidebar.toggle("开启量子实时监控", value=True)
refresh_seconds = st.sidebar.slider("刷新频率 (秒)", 5, 60, 10)

st.sidebar.divider()
ema_f_val = st.sidebar.number_input("快线 EMA", 5, 30, 12)
ema_s_val = st.sidebar.number_input("慢线 EMA", 20, 100, 26)

# ==========================================
# 3. 数据引擎：多维数据抓取
# ==========================================
@st.cache_data(ttl=refresh_seconds)
def get_quantum_kline():
    """抓取K线并注入影子技术指标"""
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=10).json()
        if r.get('code') != '0': return st.session_state.df
        df = pd.DataFrame(r['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df = df.reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 影子列计算
        df['ema_f'] = df['c'].ewm(span=ema_f_val, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=ema_s_val, adjust=False).mean()
        df['macd'] = df['ema_f'] - df['ema_s']
        # 模拟盘口净流 (基于价格变化权重)
        df['net_flow'] = (df['c'].diff() * df['v'] * 0.1).rolling(5).sum().fillna(0)
        return df
    except:
        return st.session_state.df

@st.cache_data(ttl=2)
def get_depth_walls():
    """抓取盘口深度数据"""
    url = "https://www.okx.com/api/v5/market/books?instId=ETH-USDT&sz=20"
    try:
        r = requests.get(url, timeout=5).json()
        if r.get('code') != '0': return None, None
        data = r['data'][0]
        bids = pd.DataFrame(data['bids'], columns=['px', 'sz', 'liq', 'cnt']).astype(float)
        asks = pd.DataFrame(data['asks'], columns=['px', 'sz', 'liq', 'cnt']).astype(float)
        return bids, asks
    except:
        return None, None

# ==========================================
# 4. 逻辑层：撤单侦测与情绪建模
# ==========================================
def analyze_market_behavior(bids, asks):
    """撤单行为侦测算法 (Spoofing Detection)"""
    if bids is None or asks is None or st.session_state.prev_bids.empty:
        st.session_state.prev_bids, st.session_state.prev_asks = bids, asks
        return

    now_str = datetime.now().strftime('%H:%M:%S')
    # 动态阈值：挂单量超过盘口平均值的 3 倍定义为“巨墙”
    wall_threshold = (bids['sz'].mean() + asks['sz'].mean()) * 3.0

    # 1. 监测买方撤单 (诱多)
    for _, prev_row in st.session_state.prev_bids.iterrows():
        if prev_row['sz'] > wall_threshold:
            current_match = bids[bids['px'] == prev_row['px']]
            if current_match.empty or current_match.iloc[0]['sz'] < prev_row['sz'] * 0.2:
                st.session_state.battle_logs.insert(0, f"【{now_str}】⚠️ 诱多警报：买方巨墙消失! ${prev_row['px']}")

    # 2. 监测卖方撤单 (诱空)
    for _, prev_row in st.session_state.prev_asks.iterrows():
        if prev_row['sz'] > wall_threshold:
            current_match = asks[asks['px'] == prev_row['px']]
            if current_match.empty or current_match.iloc[0]['sz'] < prev_row['sz'] * 0.2:
                st.session_state.battle_logs.insert(0, f"【{now_str}】⚠️ 诱空警报：卖方巨墙消失! ${prev_row['px']}")

    # 更新缓存
    st.session_state.prev_bids, st.session_state.prev_asks = bids, asks

def update_whale_sentiment():
    """基于最近 10 条撤单行为计算情绪指数"""
    recent = st.session_state.battle_logs[:10]
    bull_signals = sum(1 for log in recent if "诱空警报" in log)
    bear_signals = sum(1 for log in recent if "诱多警报" in log)
    
    # 基础分数 50，诱空撤单加分(看涨)，诱多撤单扣分(看跌)
    raw_score = 50 + (bull_signals * 10) - (bear_signals * 10)
    st.session_state.sentiment_score = max(min(raw_score, 100), 0)

# ==========================================
# 5. UI 渲染引擎
# ==========================================
def main():
    df = get_quantum_kline()
    bids, asks = get_depth_walls()
    analyze_market_behavior(bids, asks)
    update_whale_sentiment()
    
    # --- 头部情绪进度条 (对应截图 3/4) ---
    score = st.session_state.sentiment_score
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    st.progress(score / 100)
    
    sentiment_color = "#00ff00" if score > 50 else "#ff4b4b"
    sentiment_emoji = "🔥 庄家拉升" if score > 50 else "💀 庄家砸盘" if score < 50 else "⚖️ 震荡博弈"
    st.markdown(f"<p style='text-align:center; color:{sentiment_color}; font-weight:bold;'>庄家情绪指数: {score:.1f} | {sentiment_emoji}</p>", unsafe_allow_html=True)

    # --- 核心指标看板 ---
    last = df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前价", f"${last['c']:.2f}", f"{last['c']-last['o']:+.2f}")
    m2.metric("量子阻力", f"${df['h'].max():.1f}")
    m3.metric("盘口净流", f"{last['net_flow']:.2f} ETH")
    m4.metric("物理状态", "Active Sync")

    st.divider()

    # --- 双栏实战布局 ---
    col_l, col_r = st.columns([1, 4])
    
    with col_l:
        # 实时信号判断
        is_gold = (last['ema_f'] > last['ema_s']) and (df.iloc[-2]['ema_f'] <= df.iloc[-2]['ema_s'])
        signal_active = is_gold and last['net_flow'] > 0
        
        status_color = "#FFD700" if signal_active else "#FF00FF"
        st.markdown(f"""
            <div style="border:2px solid {status_color}; padding:20px; border-radius:15px; text-align:center; 
            background:rgba(0,0,0,0.5); box-shadow: 0 0 20px {status_color};">
                <h2 style="color:{status_color}; margin:0;">{'🔥 黄金总攻' if signal_active else '🔒 AI 猎杀'}</h2>
                <p style="color:#888; font-size:0.8em; margin-top:10px;">支撑位: ${df['l'].min():.1f}</p>
            </div>
        """, unsafe_allow_html=True)
        if signal_active: st.balloons()
        
        st.markdown("#### 📜 战术日志")
        # 显示带颜色的实时日志
        for log in st.session_state.battle_logs[:8]:
            if "⚠️" in log:
                st.error(log) # 截图中的红色告警效果
            else:
                st.caption(log)

    with col_r:
        # 物理绘图引擎
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.3])
        
        # 1. K线
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        
        # 2. 盘口大单墙柱状图 (对应截图底部第一个副图)
        if bids is not None:
            fig.add_trace(go.Bar(x=bids['px'], y=bids['sz'], name='买单墙', marker_color='#00ff00', opacity=0.4), row=2, col=1)
            fig.add_trace(go.Bar(x=asks['px'], y=asks['sz'], name='卖单墙', marker_color='#ff4b4b', opacity=0.4), row=2, col=1)

        # 3. 盘口净流图 (对应截图底部第二个副图)
        colors = ['#00ff00' if x > 0 else '#ff4b4b' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=colors, name="净流"), row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 自动重绘逻辑
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()

if __name__ == "__main__":
    main()
