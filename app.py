import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime

# ---------- 1. 视觉与页面配置 ----------
st.set_page_config(layout="wide", page_title="ETH Warrior | 纯量价心法", page_icon="⚔️")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .stMetric { background: #1a1c24; border-radius: 10px; padding: 10px; border: 1px solid #3e424b; }
    </style>
""", unsafe_allow_html=True)

# ---------- 2. 侧边栏：核心策略参数 (严格对应口诀) ----------
with st.sidebar:
    st.title("⚔️ Warrior 策略中心")
    st.info("模式：只看K线 + 成交量 (5min)")
    
    # 策略阈值
    vol_ma_len = st.number_input("均量参考周期", 5, 30, 10)
    shrink_ratio = st.slider("缩量判定 (小于均量%)", 30, 80, 60) / 100
    expand_ratio = st.slider("放量判定 (大于均量%)", 120, 300, 150) / 100
    body_min = st.slider("突破实体占比", 0.0, 0.5, 0.15)
    
    st.divider()
    symbol = st.text_input("交易对", "ETH-USDT")
    refresh_sec = st.slider("秒级刷新", 5, 60, 10)
    
    # 物理冷静期逻辑
    st.warning("若连续亏损2笔，请点击下方开启物理冷静")
    if st.button("🔴 开启 1小时 冷静锁定"):
        st.session_state.cooldown = time.time() + 3600

# ---------- 3. 核心算法：量价口诀逻辑化 ----------
def apply_warrior_logic(df, s_ratio, e_ratio, v_ma_len, b_min):
    df = df.copy()
    # 1. 计算均量与波动率
    df['v_ma'] = df['volume'].rolling(v_ma_len).mean()
    df['h20'] = df['high'].rolling(20).max().shift(1)
    df['l20'] = df['low'].rolling(20).min().shift(1)
    
    # 2. 状态判定
    df['is_shrink'] = df['volume'] < (df['v_ma'] * s_ratio)
    df['is_expand'] = df['volume'] > (df['v_ma'] * e_ratio)
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_pct'] = df['body'] / df['range']
    
    df['signal'] = 0 # 1:做多, -1:做空, 0.5:多头观察, -0.5:空头观察
    
    for i in range(1, len(df)):
        # --- 做多逻辑 ---
        # A. 缩量回踩低点不破 (观察区)
        if df['is_shrink'].iloc[i] and df['low'].iloc[i] <= df['l20'].iloc[i] * 1.002:
            df.at[df.index[i], 'signal'] = 0.5
            
        # B. 放量起涨突破阴线 (执行区)
        if df['is_expand'].iloc[i] and df['close'].iloc[i] > df['open'].iloc[i]:
            if df['close'].iloc[i] > df['open'].iloc[i-1] and df['body_pct'].iloc[i] > b_min:
                df.at[df.index[i], 'signal'] = 1

        # --- 做空逻辑 ---
        # C. 缩量反弹高点不破 (观察区)
        if df['is_shrink'].iloc[i] and df['high'].iloc[i] >= df['h20'].iloc[i] * 0.998:
            df.at[df.index[i], 'signal'] = -0.5

        # D. 放量杀跌跌破阳线 (执行区)
        if df['is_expand'].iloc[i] and df['close'].iloc[i] < df['open'].iloc[i]:
            if df['close'].iloc[i] < df['open'].iloc[i-1] and df['body_pct'].iloc[i] > b_min:
                df.at[df.index[i], 'signal'] = -1
                
    return df

# ---------- 4. 数据通道 (OKX V5) ----------
def fetch_okx_data(instId):
    url = f"https://www.okx.com/api/v5/market/candles?instId={instId.upper()}&bar=5m&limit=100"
    try:
        res = requests.get(url, timeout=5).json()
        data = res.get('data', [])
        if not data: return None
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.sort_values('ts').reset_index(drop=True).rename(columns={'ts':'time','o':'open','h':'high','l':'low','c':'close','v':'volume'})
    except:
        return None

# ---------- 5. 实时渲染循环 ----------
main_placeholder = st.empty()

while True:
    df_raw = fetch_okx_data(symbol)
    
    with main_placeholder.container():
        if df_raw is not None and not df_raw.empty:
            df = apply_warrior_logic(df_raw, shrink_ratio, expand_ratio, vol_ma_len, body_min)
            last = df.iloc[-1]
            
            # --- 头部数据看板 ---
            st.header(f"⚔️ ETH Warrior 实时监控 | {symbol}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("当前价格", f"${last['close']}", f"{last['close']-df['close'].iloc[-2]:.2f}")
            m2.metric("当前量能比", f"{(last['volume']/last['v_ma']):.2f}x")
            
            # 信号实时播报
            if last['signal'] == 1:
                st.success("🚀 【放量突破】多头真钱进场，直接开多！")
            elif last['signal'] == -1:
                st.error("📉 【放量跌破】空头主动出击，直接开空！")
            elif last['signal'] == 0.5:
                st.warning("👀 【缩量不破底】进入多头观察区，等放量阳线。")
            elif last['signal'] == -0.5:
                st.warning("👀 【缩量不过顶】进入空头观察区，等放量阴线。")
            else:
                st.write("💎 当前处于震荡整理期，耐性等待缩量或放量信号...")

            # --- 专业 Plotly 图表 ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            # K线图
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
            
            # 信号标注
            buy_sig = df[df['signal'] == 1]
            sell_sig = df[df['signal'] == -1]
            fig.add_trace(go.Scatter(x=buy_sig['time'], y=buy_sig['low']*0.997, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff00'), name="多头执行"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_sig['time'], y=sell_sig['high']*1.003, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff0000'), name="空头执行"), row=1, col=1)

            # 成交量与均量线
            colors = ['red' if row['close'] < row['open'] else 'green' for i, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, opacity=0.4, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['v_ma'], line=dict(color='#ffa500', width=1.5), name="MA_Vol"), row=2, col=1)

            fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')
            
            # --- 历史信号 Log ---
            with st.expander("📝 历史进场信号日志"):
                st.table(df[df['signal'].abs() == 1][['time', 'close', 'volume', 'signal']].tail(5))
        else:
            st.error("❌ 数据链路中断，正在尝试重新连接 OKX API...")

    time.sleep(refresh_sec)
