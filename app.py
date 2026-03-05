import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime

# ---------- 页面配置 ----------
st.set_page_config(layout="wide", page_title="ETH高频量价监控")
st.title("🔥 ETH 5min 高频进场交易机器人")

# ---------- 侧边栏：回测最优高频参数 ----------
with st.sidebar:
    st.header("⚙️ 策略执行参数")
    # 预设为回测中交易次数最多的组合
    p_body = st.slider("最低实体占比 (body)", 0.05, 0.3, 0.15)
    p_mult = st.slider("放量倍数 (mult)", 1.1, 2.0, 1.4)
    p_vol_ma = st.slider("均量周期 (vol_ma)", 5, 20, 10)
    
    st.divider()
    proxy_url = st.text_input("代理地址 (本地运行必填)", "http://127.0.0.1:7890")
    refresh_rate = st.slider("自动刷新周期 (秒)", 10, 60, 30)

# ---------- 核心算法：高频进场逻辑 ----------
def apply_high_freq_strategy(df, body_min, vol_mult, vol_ma_len):
    df = df.copy()
    df['h20'] = df['high'].rolling(20).max().shift(1)
    df['l20'] = df['low'].rolling(20).min().shift(1)
    df['v_ma'] = df['volume'].rolling(vol_ma_len).mean()
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    df['signal'] = 0
    # 提高频率：缩量定义放宽至 80%
    df['is_shrink'] = df['volume'] < df['v_ma'] * 0.8
    
    for i in range(vol_ma_len, len(df)):
        # 做多：前一根缩量靠近支撑 + 本根轻微放量实体突破
        if df['is_shrink'].iloc[i-1] and df['low'].iloc[i-1] <= df['l20'].iloc[i-1] * 1.003:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] > df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = 1
        
        # 做空：前一根缩量靠近压力 + 本根轻微放量实体跌破
        elif df['is_shrink'].iloc[i-1] and df['high'].iloc[i-1] >= df['h20'].iloc[i-1] * 0.997:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] < df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = -1
    return df

# ---------- 数据抓取与 UI 渲染 ----------
def fetch_data(symbol="ETHUSDT", proxy=None):
    url = "https://api.binance.com/api/v3/klines"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        r = requests.get(url, params={"symbol": symbol, "interval": "5m", "limit": 150}, proxies=proxies, timeout=5)
        df = pd.DataFrame(r.json(), columns=['ts','o','h','l','c','v','ct','qv','nt','tb','tq','i'])
        df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.columns = ['timestamp','open','high','low','close','volume']
        return df
    except: return None

placeholder = st.empty()
while True:
    raw_df = fetch_data(proxy=proxy_url)
    if raw_df is not None:
        df_sig = apply_high_freq_strategy(raw_df, p_body, p_mult, p_vol_ma)
        last = df_sig.iloc[-1]
        
        with placeholder.container():
            # 看板
            c1, c2, c3 = st.columns(3)
            c1.metric("ETH 实时价", f"${last['close']}")
            c2.metric("当前成交量", f"{last['volume']:.0f}", delta=f"均量: {last['v_ma']:.0f}")
            
            if last['signal'] != 0:
                side = "做多 🟢" if last['signal'] == 1 else "做空 🔴"
                st.balloons()
                st.success(f"⚡ 信号触发：{side} | 入场价: {last['close']} | 实体占比: {last['body_ratio']:.2f}")
            
            # K线绘图
            fig = go.Figure(data=[go.Candlestick(x=df_sig['timestamp'], open=df_sig['open'], high=df_sig['high'], low=df_sig['low'], close=df_sig['close'])])
            buys = df_sig[df_sig['signal'] == 1]
            sells = df_sig[df_sig['signal'] == -1]
            fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name="做多"))
            fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name="做空"))
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=650, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
    time.sleep(refresh_rate)
