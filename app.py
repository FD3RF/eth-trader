import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime

# ---------- 1. 页面配置 ----------
st.set_page_config(layout="wide", page_title="ETH高频量价监控")

# ---------- 2. 侧边栏参数 (基于回测 142.7 盈亏比组优化) ----------
with st.sidebar:
    st.header("⚙️ 策略执行参数")
    p_body = st.slider("最低实体占比 (body)", 0.05, 0.4, 0.15)
    p_mult = st.slider("放量倍数 (mult)", 1.1, 2.5, 1.4)
    p_vol_ma = st.slider("均量周期 (vol_ma)", 5, 30, 10)
    
    st.divider()
    symbol = st.text_input("交易对", "ETHUSDT")
    # Streamlit Cloud 通常不需要代理，如果是本地运行请填写
    proxy_url = st.text_input("代理地址 (可选)", "")
    refresh_rate = st.slider("刷新频率 (秒)", 5, 60, 15)

# ---------- 3. 核心算法逻辑 ----------
def apply_strategy(df, body_min, vol_mult, vol_ma_len):
    df = df.copy()
    df['h20'] = df['high'].rolling(20).max().shift(1)
    df['l20'] = df['low'].rolling(20).min().shift(1)
    df['v_ma'] = df['volume'].rolling(vol_ma_len).mean()
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    df['signal'] = 0
    df['is_shrink'] = df['volume'] < df['v_ma'] * 0.85 
    
    for i in range(vol_ma_len, len(df)):
        # 做多：缩量不破底 + 放量阳线
        if df['is_shrink'].iloc[i-1] and df['low'].iloc[i-1] <= df['l20'].iloc[i-1] * 1.002:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] > df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = 1
        # 做空：缩量不过高 + 放量阴线
        elif df['is_shrink'].iloc[i-1] and df['high'].iloc[i-1] >= df['h20'].iloc[i-1] * 0.998:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] < df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = -1
    return df

# ---------- 4. 数据抓取 ----------
def fetch_data(symbol, proxy):
    url = "https://api.binance.com/api/v3/klines"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        r = requests.get(url, params={"symbol": symbol.upper(), "interval": "5m", "limit": 100}, proxies=proxies, timeout=5)
        df = pd.DataFrame(r.json(), columns=['ts','o','h','l','c','v','ct','qv','nt','tb','tq','i'])
        df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.columns = ['timestamp','open','high','low','close','volume']
        return df
    except:
        return None

# ---------- 5. 原生自动刷新循环 ----------
placeholder = st.empty()

while True:
    raw_df = fetch_data(symbol, proxy_url)
    
    with placeholder.container():
        if raw_df is not None:
            df_sig = apply_strategy(raw_df, p_body, p_mult, p_vol_ma)
            last = df_sig.iloc[-1]
            
            # 状态看板
            st.subheader(f"🛡️ ETH 5min 实时监控 | {datetime.now().strftime('%H:%M:%S')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("价格", f"${last['close']}")
            c2.metric("量能比", f"{last['volume']/last['v_ma']:.2f}x")
            
            if last['signal'] == 1:
                st.success("🟢 进场信号：做多！(缩量回踩+放量突破)")
            elif last['signal'] == -1:
                st.error("🔴 进场信号：做空！(缩量触顶+放量杀跌)")

            # 图表渲染
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_sig['timestamp'], open=df_sig['open'], high=df_sig['high'], 
                                         low=df_sig['low'], close=df_sig['close'], name="ETH"), row=1, col=1)
            
            buys = df_sig[df_sig['signal'] == 1]
            sells = df_sig[df_sig['signal'] == -1]
            fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name="多"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name="空"), row=1, col=1)
            
            fig.add_trace(go.Bar(x=df_sig['timestamp'], y=df_sig['volume'], name="量", marker_color='gray'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_sig['timestamp'], y=df_sig['v_ma'], line=dict(color='orange'), name="均量"), row=2, col=1)

            fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark", margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("等待数据连接...")
    
    # 原生等待，实现高频刷新
    time.sleep(refresh_rate)
