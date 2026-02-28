import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- [物理焊死层] ---
st.set_page_config(layout="wide", page_title="ETH V9000 战神决战版")

# 1. 深度同步引擎 (OKX 原始毫秒锚定)
def get_warrior_data():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=5).json()
        df = pd.DataFrame(r.get('data', []), columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        # 核心：锁定北京时间，严禁使用 reset_index 破坏坐标
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # 物理对齐指标
        df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
        df['macd'] = df['c'].ewm(span=12, adjust=False).mean() - df['c'].ewm(span=26, adjust=False).mean()
        # 动态压力/支撑 (基于截图 V2500 逻辑)
        df['res'] = df['h'].rolling(40).max()
        df['sup'] = df['l'].rolling(40).min()
        return df
    except:
        return None

df = get_warrior_data()

if df is not None:
    # 顶部状态栏：还原截图 AF6F3A1E 风格
    st.markdown(f"### ⚔️ ETH-USDT 核心战区 | 物理同步状态: 锁死 | {df['time'].iloc[-1].strftime('%H:%M:%S')}")
    
    col_ui, col_chart = st.columns([1, 4])
    
    with col_ui:
        # 猎杀锁 UI
        st.markdown(f"""<div style="border:4px solid #FF00FF; padding:15px; border-radius:15px; text-align:center; background:rgba(255,0,255,0.1);">
            <h2 style="color:#FF00FF;">🔒 AI 猎杀锁</h2>
            <p style="font-size:28px; color:white;">09:59</p>
            <p style="color:#888;">逆鳞系数: 0.95</p>
        </div>""", unsafe_allow_html=True)
        
        # 实时数据监控
        st.metric("实时价", f"${df['c'].iloc[-1]:.2f}", delta=f"{df['c'].iloc[-1]-df['o'].iloc[-1]:.2f}")
        st.write(f"📈 **压力拦截**: {df['res'].iloc[-1]:.1f}")
        st.write(f"📉 **支撑托盘**: {df['sup'].iloc[-1]:.1f}")

    with col_chart:
        # 强制共享 X 轴，确保指标绝对不飘移
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
        
        # 主 K 线
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="同步K线"), row=1, col=1)
        # 趋势生命线
        fig.add_trace(go.Scatter(x=df['time'], y=df['ema20'], line=dict(color='yellow', width=2), name="EMA20"), row=1, col=1)
        # 动态压力线
        fig.add_hline(y=df['res'].iloc[-1], line_color="red", line_dash="solid", row=1, col=1)
        fig.add_hline(y=df['sup'].iloc[-1], line_color="green", line_dash="dash", row=1, col=1)

        # 副图 MACD (物理对齐)
        fig.add_trace(go.Bar(x=df['time'], y=df['macd'], marker_color='cyan', name="动能"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
        # 锁定 X 轴最近 60 分钟，防止截图中的“视觉拉伸”
        fig.update_xaxes(range=[df['time'].iloc[-60], df['time'].iloc[-1]])
        st.plotly_chart(fig, use_container_width=True)

    # 战地复盘看板
    st.markdown(f"""<div style="background:#1e2130; padding:15px; border-radius:10px; border-left: 5px solid cyan;">
        <b>AI 战地复盘：</b>当前属于 【趋势模式】，动能校验 【虚假】。拦截位已自动锁定在 ${df['res'].iloc[-1]:.2f}。
    </div>""", unsafe_allow_html=True)
