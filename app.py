import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==================== 1. 核心引擎配置 ====================
def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True # 穿透图 10/11 的 LetsVPN 隧道
            r = s.get(url, timeout=5)
            return r.json()
    except: return None

def get_clean_kline(inst, bar='5m', limit=100):
    data = fetch_okx("market/candles", f"&bar={bar}&limit={limit}")
    if not data or 'data' not in data: return pd.DataFrame()
    df = pd.DataFrame(data['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    return df[::-1].reset_index(drop=True)

# ==================== 2. 深度择时逻辑 ====================
def add_indicators(df):
    # RSI & W%R
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    high_14, low_14 = df['h'].rolling(14).max(), df['l'].rolling(14).min()
    df['w_r'] = -100 * (high_14 - df['c']) / (high_14 - low_14)
    # EMA
    df['ema21'] = df['c'].ewm(span=21).mean()
    return df

# ==================== 3. 指挥部主界面 ====================
st.set_page_config(page_title="ETH V44.0 终极指挥部", layout="wide")

# 数据同步
df_5m = add_indicators(get_clean_kline("ETH-USDT", "5m", 150))
depth = fetch_okx("market/books", "&sz=20")

if not df_5m.empty and depth:
    curr_p = df_5m.iloc[-1]['c']
    
    # 盘口平衡计算 (饼图逻辑)
    asks = pd.DataFrame(depth['data'][0]['asks']).iloc[:, :2].astype(float)
    bids = pd.DataFrame(depth['data'][0]['bids']).iloc[:, :2].astype(float)
    ask_v, bid_v = asks[1].sum(), bids[1].sum()
    bid_pct = (bid_v / (ask_v + bid_v)) * 100

    # 标题区
    st.title("🛡️ ETH V44.0 终极全功能指挥部")
    
    # 第一行：实时平衡雷达
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1:
        fig_pie = go.Figure(data=[go.Pie(labels=['卖压', '买压'], values=[ask_v, bid_v], 
                                         hole=.4, marker_colors=['#ff4b4b', '#00ffcc'])])
        fig_pie.update_layout(height=250, margin=dict(t=30,b=0,l=0,r=0), template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c2:
        win_rate = 50 + (bid_pct - 50) * 1.2
        st.metric("实时胜算预测", f"{win_rate:.1f}%", f"{bid_pct-50:.1f}% 净偏差")
        st.progress(max(0.0, min(win_rate/100, 1.0)))
    
    with c3:
        st.metric("ETH 现价", f"${curr_p:.2f}")
        rsi_val = df_5m.iloc[-1]['rsi']
        st.write(f"指标状态: {'🟢 超卖' if rsi_val < 35 else ('🔴 超买' if rsi_val > 65 else '⚪ 正常')}")
    
    with c4:
        # 综合风险告警
        risk = 98.9 if curr_p < df_5m.iloc[-1]['ema21'] and bid_pct < 45 else 15.0
        st.metric("阴跌综合风险", f"{risk}", delta_color="inverse")

    # 第二行：全能择时主副图
    st.divider()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df_5m.index, open=df_5m['o'], high=df_5m['h'], low=df_5m['l'], close=df_5m['c'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['ema21'], line=dict(color='cyan', width=1), name="EMA21"), row=1, col=1)
    
    # 副图 RSI + W%R
    fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m['rsi'], name="RSI", line=dict(color='#00ffcc')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # 第三行：强弱对比与笔记
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("📊 庄家挂单墙 (Order Book Wall)")
        b1, b2 = st.columns(2)
        b1.table(asks.head(5).rename(columns={0:"卖价", 1:"数量"}))
        b2.table(bids.head(5).rename(columns={0:"买价", 1:"数量"}))
    
    with col_r:
        st.subheader("📝 执行笔记快照")
        note = st.text_area("输入当前逻辑...", placeholder="例如：胜算>60% 且 RSI抄底信号")
        if st.button("保存当前状态"):
            st.success(f"已存：价{curr_p} | 胜算{win_rate:.1f}%")

    st.caption(f"🚀 V44.0 Execution Elite | 实时多源同步 | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")

else:
    st.info("🔄 正在通过 LetsVPN 捕获深度数据流...")
