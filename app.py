import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime

# ---------- 1. 页面配置 ----------
st.set_page_config(layout="wide", page_title="ETH高频心法监控")

# ---------- 2. 侧边栏：回测最优参数 (0.15/10/1.4) ----------
with st.sidebar:
    st.header("⚙️ 策略参数 (90天回测高频组)")
    # 预设为回测中 14 次交易/盈亏比 142.7 的参数组合
    p_body = st.slider("最低实体占比 (body)", 0.05, 0.4, 0.15)
    p_mult = st.slider("放量倍数 (mult)", 1.1, 2.5, 1.4)
    p_vol_ma = st.slider("均量周期 (vol_ma)", 5, 30, 10)
    
    st.divider()
    symbol = st.text_input("交易对 (币安)", "ETHUSDT")
    proxy_url = st.text_input("代理地址 (国内运行必填)", "", help="例: http://127.0.0.1:7890")
    refresh_rate = st.slider("自动刷新间隔 (秒)", 5, 60, 15)

# ---------- 3. 核心算法逻辑 ----------
def apply_strategy(df, body_min, vol_mult, vol_ma_len):
    df = df.copy()
    # 动态关键位：20周期高低点
    df['h20'] = df['high'].rolling(20).max().shift(1)
    df['l20'] = df['low'].rolling(20).min().shift(1)
    df['v_ma'] = df['volume'].rolling(vol_ma_len).mean()
    # 实体占比过滤（心法精髓：过滤插针）
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    df['signal'] = 0
    # 提高频次：缩量定义放宽
    df['is_shrink'] = df['volume'] < df['v_ma'] * 0.85 
    
    for i in range(vol_ma_len, len(df)):
        # 做多：前根缩量不破底 + 本根放量阳线
        if df['is_shrink'].iloc[i-1] and df['low'].iloc[i-1] <= df['l20'].iloc[i-1] * 1.002:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] > df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = 1
        
        # 做空：前根缩量不过高 + 本根放量阴线
        elif df['is_shrink'].iloc[i-1] and df['high'].iloc[i-1] >= df['h20'].iloc[i-1] * 0.998:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] < df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = -1
    return df

# ---------- 4. 实时数据获取 ----------
def fetch_data(symbol, proxy):
    url = "https://api.binance.com/api/v3/klines"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        r = requests.get(url, params={"symbol": symbol.upper(), "interval": "5m", "limit": 120}, proxies=proxies, timeout=5)
        df = pd.DataFrame(r.json(), columns=['ts','o','h','l','c','v','ct','qv','nt','tb','tq','i'])
        df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.columns = ['timestamp','open','high','low','close','volume']
        return df
    except Exception as e:
        return None

# ---------- 5. 页面自动刷新逻辑 (无插件版) ----------
placeholder = st.empty()

# 实时刷新循环
while True:
    raw_df = fetch_data(symbol, proxy_url)
    
    with placeholder.container():
        if raw_df is not None:
            df_sig = apply_strategy(raw_df, p_body, p_mult, p_vol_ma)
            last = df_sig.iloc[-1]
            
            # 顶部仪表盘
            st.markdown(f"### ⚡ ETH 5min 量价心法监控 | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("当前价", f"${last['close']}")
            c2.metric("量能系数", f"{last['volume']/last['v_ma']:.2f}x")
            
            # 进场警报
            if last['signal'] == 1:
                st.success(f"🟢 [做多信号] 价格: {last['close']} | 理由: 缩量踩底+放量突破")
            elif last['signal'] == -1:
                st.error(f"🔴 [做空信号] 价格: {last['close']} | 理由: 缩量触顶+放量杀跌")

            # 绘制专业K线图
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            # 主图
            fig.add_trace(go.Candlestick(x=df_sig['timestamp'], open=df_sig['open'], high=df_sig['high'], 
                                         low=df_sig['low'], close=df_sig['close'], name="ETH"), row=1, col=1)
            # 信号点
            buys = df_sig[df_sig['signal'] == 1]
            sells = df_sig[df_sig['signal'] == -1]
            fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.998, mode='markers', 
                                     marker=dict(symbol='triangle-up', size=15, color='lime'), name="做多信号"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.002, mode='markers', 
                                     marker=dict(symbol='triangle-down', size=15, color='red'), name="做空信号"), row=1, col=1)
            # 副图：成交量与均线
            fig.add_trace(go.Bar(x=df_sig['timestamp'], y=df_sig['volume'], name="成交量", marker_color='gray'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_sig['timestamp'], y=df_sig['v_ma'], line=dict(color='orange', width=1), name="均量线"), row=2, col=1)

            fig.update_layout(xaxis_rangeslider_visible=False, height=700, template="plotly_dark", margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ 数据连接中断，请检查代理设置或网络状态...")
    
    time.sleep(refresh_rate)
