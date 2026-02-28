import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import IsolationForest
import time
from datetime import datetime

# ================================
# 1. 物理同步数据获取 (OKX 原始时戳)
# ================================
def get_okx_data():
    url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100"
    try:
        r = requests.get(url, timeout=5).json()
        if r.get('code') != '0': return None
        raw = r.get('data', [])
        # 强制倒序并转换时戳
        df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['time'] = pd.to_datetime(df['ts'].astype(float), unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        for col in ['o','h','l','c','v']:
            df[col] = df[col].astype(float)
        return df.reset_index(drop=True)
    except:
        return None

# ================================
# 2. 深度学习与特征引擎
# ================================
def process_ai_logic(df):
    # 特征工程
    df['returns'] = df['c'].pct_change()
    df['label'] = (df['returns'].shift(-1) > 0).astype(int)
    
    # 异常检测 (Isolation Forest)
    iso = IsolationForest(contamination=0.03, random_state=42)
    df['anomaly'] = iso.fit_predict(df[['o','h','l','c','v']])
    
    # 构建 LSTM 序列 (Window=10 快速响应)
    window = 10
    features = df[['o','h','l','c','v']].values
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i+window])
        y.append(df['label'].iloc[i+window])
    
    X = np.array(X)
    y = np.array(y)
    
    # 构建并轻量训练模型
    model = Sequential([
        LSTM(32, input_shape=(window, 5), return_sequences=False),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # 实战建议：此处仅为演示，生产环境应加载预训练模型以保时效
    model.fit(X, y, epochs=2, batch_size=16, verbose=0) 
    
    # 推断最后一根 K 线
    last_seq = features[-window:].reshape(1, window, 5)
    prob = model.predict(last_seq)[0][0]
    
    return df, prob

# ================================
# 3. 战神全量 UI 渲染 (强制同步)
# ================================
st.set_page_config(page_title="ETH V4000 量子同步版", layout="wide")
st.title(f"⚔️ ETH V4000 战神·量子同步版 | {datetime.now().strftime('%H:%M:%S')}")

df_raw = get_okx_data()

if df_raw is not None:
    # 核心计算逻辑
    df, ai_prob = process_ai_logic(df_raw)
    
    # 顶部指标面板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前价格", f"${df['c'].iloc[-1]}")
    c2.metric("LSTM 向上概率", f"{ai_prob*100:.1f}%")
    
    advice = "看多 (STRUCTURE UP)" if ai_prob > 0.6 else "看空 (STRUCTURE DOWN)" if ai_prob < 0.4 else "观望 (NEUTRAL)"
    color = "#00FFCC" if ai_prob > 0.6 else "#FF4B4B" if ai_prob < 0.4 else "#888888"
    c3.markdown(f"**AI 建议**: <span style='color:{color}'>{advice}</span>", unsafe_allow_html=True)
    c4.metric("异常状态", "检测中" if df['anomaly'].iloc[-1] == 1 else "发现异动", delta_color="inverse")

    # ================================
    # 4. 同步绘图核心 (不同步的死穴在此修复)
    # ================================
    # 创建主副图，强制 shared_xaxes=True
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3],
        subplot_titles=("K线与异常点同步", "成交量动能")
    )

    # 绘制 K 线 (强制使用 time 序列)
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        name="ETH/USDT"
    ), row=1, col=1)

    # 叠加异常点 (物理同步在对应 K 线时间)
    anomalies = df[df['anomaly'] == -1]
    fig.add_trace(go.Scatter(
        x=anomalies['time'], y=anomalies['c'],
        mode='markers',
        marker=dict(color='#FF00FF', size=10, symbol='x'),
        name="庄家异动"
    ), row=1, col=1)

    # 绘制成交量
    fig.add_trace(go.Bar(
        x=df['time'], y=df['v'], 
        marker_color='rgba(0, 255, 204, 0.3)',
        name="成交量"
    ), row=2, col=1)

    # 布局优化
    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    # 锁定 X 轴显示范围，防止左右偏移
    fig.update_xaxes(range=[df['time'].iloc[-50], df['time'].iloc[-1]])

    st.plotly_chart(fig, use_container_width=True)

    # 复盘简报
    st.write("---")
    st.subheader("📋 战地复盘简报")
    with st.expander("点击查看深度统计"):
        st.write(df.tail(10)[['time', 'c', 'v', 'anomaly']])
        st.info("物理锁死机制：当前绘图 X 轴已强制锚定 OKX 毫秒戳，指标漂移已被物理切除。")

else:
    st.error("数据链路中断，请检查网络或 API 限制")
    if st.button("重新连接"):
        st.rerun()
