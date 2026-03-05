import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime

# ---------- 1. 页面配置 ----------
st.set_page_config(layout="wide", page_title="ETH高频量价监控", page_icon="📈")

# ---------- 2. 侧边栏：高频参数预设 (参考 142.7 盈亏比组) ----------
with st.sidebar:
    st.header("⚙️ 策略执行参数")
    st.info("已根据 90 天回测数据自动调优")
    p_body = st.slider("最低实体占比 (body)", 0.05, 0.4, 0.15)
    p_mult = st.slider("放量倍数 (mult)", 1.1, 2.5, 1.4)
    p_vol_ma = st.slider("均量周期 (vol_ma)", 5, 30, 10)
    
    st.divider()
    symbol = st.text_input("交易对", "ETHUSDT")
    proxy_url = st.text_input("代理地址 (若访问失败请填写)", "")
    refresh_rate = st.slider("自动刷新间隔 (秒)", 5, 60, 20)

# ---------- 3. 核心算法：高频进场逻辑 ----------
def apply_heart_logic(df, body_min, vol_mult, vol_ma_len):
    df = df.copy()
    # 动态关键支撑阻力 (20周期)
    df['h20'] = df['high'].rolling(20).max().shift(1)
    df['l20'] = df['low'].rolling(20).min().shift(1)
    # 成交量均线
    df['v_ma'] = df['volume'].rolling(vol_ma_len).mean()
    # 实体占比 (abs(close-open)/(high-low))
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    df['signal'] = 0
    # 提高频率：缩量定义放宽
    df['is_shrink'] = df['volume'] < df['v_ma'] * 0.85 
    
    for i in range(vol_ma_len, len(df)):
        # 做多：缩量靠近支撑 + 放量实体阳线突破
        if df['is_shrink'].iloc[i-1] and df['low'].iloc[i-1] <= df['l20'].iloc[i-1] * 1.002:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] > df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = 1
        
        # 做空：缩量靠近阻力 + 放量实体阴线跌破
        elif df['is_shrink'].iloc[i-1] and df['high'].iloc[i-1] >= df['h20'].iloc[i-1] * 0.998:
            if df['volume'].iloc[i] > df['v_ma'].iloc[i] * vol_mult and \
               df['close'].iloc[i] < df['open'].iloc[i] and df['body_ratio'].iloc[i] > body_min:
                df.at[df.index[i], 'signal'] = -1
    return df

# ---------- 4. 实时数据获取 ----------
def fetch_klines(symbol, proxy):
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

# ---------- 5. 实时监控界面渲染 ----------
placeholder = st.empty()

while True:
    raw_df = fetch_klines(symbol, proxy_url)
    
    with placeholder.container():
        if raw_df is not None:
            df_sig = apply_heart_logic(raw_df, p_body, p_mult, p_vol_ma)
            last = df_sig.iloc[-1]
            
            # 状态面板
            st.subheader(f"🛡️ ETH 5min 量价心法监控 | 刷新: {datetime.now().strftime('%H:%M:%S')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("当前价格", f"${last['close']}")
            c2.metric("量能比 (Vol/MA)", f"{last['volume']/last['v_ma']:.2f}x")
            
            # 信号警报
            if last['signal'] == 1:
                st.success(f"🚀 [做多进场] 价格: {last['close']} | 理由: 缩量不破底+放量突破")
            elif last['signal'] == -1:
                st.error(f"📉 [做空进场] 价格: {last['close']} | 理由: 缩量不破顶+放量杀跌")

            # 绘制专业图表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            # K线
            fig.add_trace(go.Candlestick(x=df_sig['timestamp'], open=df_sig['open'], high=df_sig['high'], 
                                         low=df_sig['low'], close=df_sig['close'], name="K线"), row=1, col=1)
            # 信号图标
            buys = df_sig[df_sig['signal'] == 1]
            sells = df_sig[df_sig['signal'] == -1]
            fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.998, mode='markers', 
                                     marker=dict(symbol='triangle-up', size=15, color='lime'), name="做多信号"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.002, mode='markers', 
                                     marker=dict(symbol='triangle-down', size=15, color='red'), name="做空信号"), row=1, col=1)
            # 成交量副图
            fig.add_trace(go.Bar(x=df_sig['timestamp'], y=df_sig['volume'], marker_color='gray', opacity=0.5, name="成交量"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_sig['timestamp'], y=df_sig['v_ma'], line=dict(color='orange', width=1), name="均量线"), row=2, col=1)

            fig.update_layout(xaxis_rangeslider_visible=False, height=700, template="plotly_dark", margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("数据抓取超时，请检查代理设置...")
            
    time.sleep(refresh_rate)
