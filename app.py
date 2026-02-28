import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ================================
# 1. 数据层（多周期）
# ================================
def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=3).json()
        return r.get('data', []) if r.get('code') == '0' else []
    except:
        return []

def get_multi():
    k1 = fetch_okx("market/candles", "&bar=1m&limit=100")
    k5 = fetch_okx("market/candles", "&bar=5m&limit=100")
    k15 = fetch_okx("market/candles", "&bar=15m&limit=100")
    t = fetch_okx("market/trades", "&limit=100")

    if not (k1 and k5 and k15 and t):
        return None

    df1 = pd.DataFrame(k1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df15 = pd.DataFrame(k15, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t)

    return df1, df5, df15, tdf


# ================================
# 2. 指标计算（审计版）
# ================================
def calc_indicators(df):
    for col in ['o','h','l','c','v']:
        df[col] = df[col].astype(float)

    df['ema20'] = df['c'].ewm(span=20).mean()
    df['ema60'] = df['c'].ewm(span=60).mean()

    tr = np.maximum(df['h'] - df['l'],
                   np.maximum(abs(df['h'] - df['c'].shift(1)),
                              abs(df['l'] - df['c'].shift(1))))
    df['atr'] = tr.rolling(14).mean()

    diff = df['c'].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = -diff.clip(upper=0).rolling(14).mean().replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))

    return df


# ================================
# 3. 多周期诊断
# ================================
def diagnose(df1, df5, df15):
    df1 = calc_indicators(df1)
    df5 = calc_indicators(df5)
    df15 = calc_indicators(df15)

    # 高频（1m）
    c1 = df1.iloc[-1]
    high_freq = {
        "rsi": c1['rsi'],
        "vol": c1['v'] / df1['v'].rolling(20).mean().iloc[-1],
        "trend": "多" if c1['ema20'] > c1['ema60'] else "空"
    }

    # 波段（5m）
    c5 = df5.iloc[-1]
    slope5 = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    swing = {
        "trend": "多" if slope5 > 0 else "空",
        "slope": slope5,
        "strength": abs(slope5) > (c5['atr'] * 0.15)
    }

    # 大趋势（15m）
    c15 = df15.iloc[-1]
    slope15 = (df15['ema20'].iloc[-1] - df15['ema20'].iloc[-5]) / 5
    trend15 = {
        "trend": "多" if slope15 > 0 else "空",
        "strength": abs(slope15) > (c15['atr'] * 0.12)
    }

    return high_freq, swing, trend15, df1


# ================================
# 4. Streamlit 面板
# ================================
st.set_page_config(page_title="多周期分析 V210", layout="wide")

data = get_multi()
if data:
    df1, df5, df15, tdf = data
    hf, sw, td, df1 = diagnose(df1, df5, df15)

    st.title("📊 多周期分析 V210（1m / 5m / 15m）")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1m 高频")
        st.metric("趋势", hf['trend'])
        st.metric("RSI", f"{hf['rsi']:.1f}")
        st.metric("量比", f"{hf['vol']:.2f}x")

    with col2:
        st.subheader("5m 波段")
        st.metric("趋势", sw['trend'])
        st.metric("斜率", f"{sw['slope']:.4f}")
        st.metric("强度", "强" if sw['strength'] else "弱")

    with col3:
        st.subheader("15m 趋势")
        st.metric("趋势", td['trend'])
        st.metric("强度", "强" if td['strength'] else "弱")

    st.write("---")
    st.subheader("趋势综合")

    st.write(f"""
    - 高频趋势：{hf['trend']}
    - 波段趋势：{sw['trend']}
    - 大趋势：{td['trend']}
    - 高频 RSI：{hf['rsi']:.1f}
    - 高频量比：{hf['vol']:.2f}
    """)

    # 图表（1m）
    fig = go.Figure(data=[go.Candlestick(
        x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c']
    )])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], name="EMA20", line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema60'], name="EMA60", line=dict(color='cyan')))
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("数据获取失败")
