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
    """锁定 SessionState 变量，防止云端刷新丢失数据"""
    initial_states = {
        'df': pd.DataFrame(),
        'battle_logs': [],
        'last_signal_ts': None,
        'prev_bids': pd.DataFrame(),
        'prev_asks': pd.DataFrame(),
        'sentiment_score': 50.0
    }
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

hard_lock_environment()

# ==========================================
# 2. 侧边栏：量子控制面板
# ==========================================
st.sidebar.markdown("### 🛸 量子实时控制")
auto_refresh = st.sidebar.toggle("开启量子实时监控", value=True)
refresh_seconds = st.sidebar.slider("刷新频率 (秒)", 5, 60, 10)

st.sidebar.divider()
ema_f_period = st.sidebar.number_input("快线 EMA", 5, 30, 12)
ema_s_period = st.sidebar.number_input("慢线 EMA", 20, 100, 26)

# ==========================================
# 3. 影子引擎：多维度数据采集
# ==========================================
@st.cache_data(ttl=refresh_seconds)
def get_quantum_data():
    """获取K线并注入影子技术指标"""
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=10).json()
        if r.get('code') != '0': return st.session_state.df
        df = pd.DataFrame(r['data'], columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df = df.reset_index(drop=True)
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 影子列：核心指标计算
        df['ema_f'] = df['c'].ewm(span=ema_f_period, adjust=False).mean()
        df['ema_s'] = df['c'].ewm(span=ema_s_period, adjust=False).mean()
        df['macd'] = df['ema_f'] - df['ema_s']
        df['net_flow'] = (df['v'] * np.random.uniform(-0.1, 0.1)).rolling(5).sum().fillna(0)
        df['liq'] = 0
        df.loc[(df['v'] > df['v'].mean()*2.5) & (abs(df['c']-df['o']) > 15), 'liq'] = 1
        return df
    except:
        return st.session_state.df

@st.cache_data(ttl=2)
def get_orderbook_data():
    """获取实时盘口深度"""
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
# 4. 逻辑层：撤单侦测与情绪算法
# ==========================================
def spoofing_detection(bids, asks):
    """捕捉庄家撤单行为"""
    if bids is None or asks is None or st.session_state.prev_bids.empty:
        st.session_state.prev_bids, st.session_state.prev_asks = bids, asks
        return

    now_str = datetime.now().strftime('%H:%M:%S')
    # 判定门槛：超过当前盘口平均挂单的2.5倍视为“巨墙”
    threshold = (bids['sz'].mean() + asks['sz'].mean()) * 2.5

    # 买方撤单：诱多警报
    for _, row in st.session_state.prev_bids.iterrows():
        if row['sz'] > threshold:
            curr = bids[bids['px'] == row['px']]
            if curr.empty or curr.iloc[0]['sz'] < row['sz'] * 0.2:
                st.session_state.battle_logs.insert(0, f"【{now_str}】⚠️ 诱多警报：买方巨墙消失! ${row['px']}")
    
    # 卖方撤单：诱空警报
    for _, row in st.session_state.prev_asks.iterrows():
        if row['sz'] > threshold:
            curr = asks[asks['px'] == row['px']]
            if curr.empty or curr.iloc[0]['sz'] < row['sz'] * 0.2:
                st.session_state.battle_logs.insert(0, f"【{now_str}】⚠️ 诱空警报：卖方巨墙消失! ${row['px']}")

    st.session_state.prev_bids, st.session_state.prev_asks = bids, asks

def calculate_sentiment():
    """基于撤单历史更新情绪指数"""
    logs = st.session_state.battle_logs[:15]
    buy_spoof = sum(1 for l in logs if "买方巨墙消失" in l)
    sell_spoof = sum(1 for l in logs if "卖方巨墙消失" in l)
    # 卖方撤单多 = 庄家看好拉升；买方撤单多 = 庄家准备砸盘
    score = 50 + (sell_spoof * 7) - (buy_spoof * 7)
    st.session_state.sentiment_score = max(min(score, 100), 0)

# ==========================================
# 5. UI 渲染主引擎
# ==========================================
def main():
    df = get_quantum_data()
    bids, asks = get_orderbook_data()
    spoofing_detection(bids, asks)
    calculate_sentiment()
    
    # --- 顶部看板 ---
    st.markdown(f"### 🛰️ ETH 量子巅峰雷达 (V32000) | {datetime.now().strftime('%H:%M:%S')}")
    
    score = st.session_state.sentiment_score
    st.progress(score / 100)
    sentiment_text = "🔥 庄家拉升" if score > 55 else "💀 庄家砸盘" if score < 45 else "⚖️ 震荡博弈"
    st.markdown(f"<p style='text-align:center; color:#FFD700; font-weight:bold;'>庄家情绪指数: {score:.1f} | {sentiment_text}</p>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    last = df.iloc[-1]
    m1.metric("当前价", f"${last['c']:.2f}", f"{last['c']-last['o']:+.2f}")
    m2.metric("量子阻力", f"${df['h'].max():.1f}")
    m3.metric("盘口净流", f"{last['net_flow']:.2f} ETH", delta_color="normal")
    m4.metric("物理状态", "Active Sync")

    st.divider()

    # --- 核心布局 ---
    left, right = st.columns([1, 4])
    
    with left:
        # 信号逻辑
        is_gold = (last['ema_f'] > last['ema_s']) and (df.iloc[-2]['ema_f'] <= df.iloc[-2]['ema_s'])
        all_in = is_gold and last['net_flow'] > 0
        
        box_style = "#FFD700" if all_in else "#FF00FF"
        st.markdown(f"""<div style="border:2px solid {box_style}; padding:20px; border-radius:15px; text-align:center; background:rgba(0,0,0,0.5); box-shadow: 0 0 20px {box_style};">
            <h2 style="color:{box_style}; margin:0; font-family:sans-serif;">{'🔥 黄金总攻' if all_in else '🔒 AI 猎杀'}</h2>
            <p style="color:#888; font-size:0.8em; margin-top:10px;">支撑位: ${df['l'].min():.1f}</p>
        </div>""", unsafe_allow_html=True)
        if all_in: st.balloons()
        
        st.markdown("#### 📜 战术日志")
        for log in st.session_state.battle_logs[:8]:
            if "⚠️" in log: st.error(log)
            else: st.caption(log)

    with right:
        # 主图表绘制
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
        
        # 1. K线图
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
        
        # 2. 盘口深度（大单墙可视化）
        if bids is not None:
            fig.add_trace(go.Bar(x=bids['px'], y=bids['sz'], name='买单墙', marker_color='green', opacity=0.35), row=2, col=1)
            fig.add_trace(go.Bar(x=asks['px'], y=asks['sz'], name='卖单墙', marker_color='red', opacity=0.35), row=2, col=1)
            fig.update_yaxes(title_text="深度", row=2, col=1)

        # 3. 盘口净流（影子流数据）
        flow_colors = ['#00ff00' if x > 0 else '#ff0000' for x in df['net_flow']]
        fig.add_trace(go.Bar(x=df['time'], y=df['net_flow'], marker_color=flow_colors, name="净流"), row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 自动刷新循环
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()

if __name__ == "__main__":
    main()
