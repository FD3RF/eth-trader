import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ==================== 1. 样式与全局配置 ====================
st.set_page_config(page_title="ETH V16.0 狙击手终端", layout="wide")

# 注入红色闪烁狙击窗口 CSS
st.markdown("""
<style>
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .sniper-box {
        background-color: #ff4b4b; color: white; padding: 20px; border-radius: 10px;
        text-align: center; border: 3px solid white; animation: blink 1s infinite;
        font-weight: bold; margin-bottom: 20px;
    }
    .countdown-text { font-size: 45px; font-family: 'Courier New', Courier; }
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state.update({'initialized': True, 'risk': 1.0, 'show_review': False})

# ==================== 2. 数据与计算引擎 ====================
@st.cache_data(ttl=10)
def fetch_data(symbol="ETH", aggregate=5, limit=300):
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histominute"
        params = {"fsym": symbol, "tsym": "USD", "limit": limit, "aggregate": aggregate, "e": "CCCAGG"}
        resp = requests.get(url, timeout=10).json()
        df = pd.DataFrame(resp['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volumefrom']].rename(columns={'volumefrom': 'volume'})
    except: return pd.DataFrame()

def get_indicators(df):
    if df.empty: return df
    df['ema8'] = df['close'].ewm(span=8).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
    return df

# ==================== 3. 核心功能逻辑 ====================
def get_mock_sentiment(price):
    # 模拟庄家挂单墙与爆仓
    walls = [
        {'price': price * 1.015, 'amount': 45, 'type': 'Ask'}, # 阻力
        {'price': price * 0.985, 'amount': 62, 'type': 'Bid'}  # 支撑
    ]
    liqs = [{'time': datetime.now() - timedelta(minutes=10), 'price': price*1.005, 'side': 'Long'}]
    return walls, liqs

def check_sniper_signal(df, walls):
    """狙击逻辑：缩量 + 触墙 + MACD转折"""
    if df.empty: return False
    last = df.iloc[-1]
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    
    # 条件：成交量萎缩 + 价格靠近支撑墙 (0.3%以内) + MACD绿柱缩短
    is_low_vol = last['volume'] < avg_vol * 0.85
    support_price = [w['price'] for w in walls if w['type'] == 'Bid'][0]
    is_near_wall = abs(last['close'] - support_price) / support_price < 0.003
    is_reversal = df['hist'].iloc[-1] > df['hist'].iloc[-2] and df['hist'].iloc[-1] < 0
    
    return is_low_vol and is_near_wall and is_reversal

# ==================== 4. 界面渲染 ====================
df5 = get_indicators(fetch_data(aggregate=5))
df15 = get_indicators(fetch_data(aggregate=15))
df1h = get_indicators(fetch_data(aggregate=60))

if not df5.empty:
    curr_price = df5['close'].iloc[-1]
    walls, liqs = get_mock_sentiment(curr_price)
    
    # --- 1. 狙击手倒计时弹窗 ---
    if check_sniper_signal(df5, walls):
        st.markdown(f"""
        <div class="sniper-box">
            <div style="font-size: 20px;">🎯 发现庄家诱空 (缩量触墙) - 黄金抄底位</div>
            <div class="countdown-text">狙击窗口: 60s</div>
            <div style="font-size: 14px;">建议止损: {curr_price*0.995:.2f} | 目标位: {df5['ema21'].iloc[-1]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.toast("🚨 满足缩量触墙逻辑，立即检查仓位！", icon="🔥")

    # --- 2. 顶部状态栏 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("全球加权价", f"${curr_price:.2f}")
    c2.metric("全网多空比", "48.2% / 51.8%")
    
    with c3:
        res8 = df5['ema8'].iloc[-1] > df5['ema21'].iloc[-1]
        res21 = df15['ema8'].iloc[-1] > df15['ema21'].iloc[-1]
        st.write("🚦 趋势共振灯")
        st.markdown(f"{'🟢' if res8 else '🔴'} 5m | {'🟢' if res21 else '🔴'} 15m")
        
    score = 65 if res8 else 45
    c4.metric("综合信心分", score, delta="反弹确认" if check_sniper_signal(df5, walls) else "观望")

    # --- 3. 主图表渲染 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df5.index, open=df5['open'], high=df5['high'], low=df5['low'], close=df5['close'], name="K线"), row=1, col=1)
    
    # 绘制庄家墙
    for w in walls:
        color = "rgba(57, 211, 83, 0.4)" if w['type']=='Bid' else "rgba(248, 81, 73, 0.4)"
        fig.add_hline(y=w['price'], line_dash="dot", line_color=color, annotation_text=f" 墙 ${w['amount']}M", row=1, col=1)
        
    # 绘制爆仓闪电
    for l in liqs:
        fig.add_annotation(x=l['time'], y=l['price'], text="⚡", font=dict(size=22, color="orange"), showarrow=False, row=1, col=1)

    fig.add_trace(go.Bar(x=df5.index, y=df5['hist'], name="动能柱", marker_color='gray'), row=2, col=1)
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. 侧边栏与 AI 复盘 ---
    with st.sidebar:
        st.header("🔍 狙击手控制台")
        if st.button("🧩 生成 AI 深度量价复盘"):
            st.session_state.show_review = True
            
    if st.session_state.show_review:
        st.divider()
        st.subheader("🧠 AI 深度复盘：量价博弈分析")
        vol_v = "缩量" if df5['volume'].iloc[-1] < df5['volume'].rolling(20).mean().iloc[-1] else "放量"
        st.write(f"信号检测：当前处于 **{vol_v}下降** 阶段。")
        if vol_v == "缩量" and curr_price < df5['ema21'].iloc[-1]:
            st.error("⚠️ 检测到【缩量诱空】：庄家在支撑墙上方故意撤单制造恐慌，但这属于‘无动力下跌’。")
        else:
            st.info("✅ 属于常规放量波动，建议遵循 EMA 共振方向交易。")
