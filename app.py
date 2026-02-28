import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import IsolationForest

# ================================
# 数据获取
# ================================
def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=3).json()
        return r.get('data', []) if r.get('code') == '0' else []
    except:
        return []

def get_data():
    k1 = fetch_okx("market/candles", "&bar=1m&limit=200")
    if not k1:
        return None

    df = pd.DataFrame(k1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
    for col in ['o','h','l','c','v']:
        df[col] = df[col].astype(float)

    return df.reset_index(drop=True)


# ================================
# 特征与序列
# ================================
def make_label(df):
    df = df.copy()
    df['future'] = df['c'].pct_change().shift(-1)
    df['label'] = (df['future'] > 0).astype(int)
    return df

def build_seq(df, window=20):
    seq = []
    for i in range(len(df) - window):
        block = df[['o','h','l','c','v']].iloc[i:i+window].values
        seq.append(block)
    return np.array(seq)


# ================================
# 深度学习模型
# ================================
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ================================
# 异常检测
# ================================
def detect_anomaly(df):
    model = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = model.fit_predict(df[['o','h','l','c','v']])
    return df


# ================================
# 回测
# ================================
def backtest(df):
    df = make_label(df)
    win = df[df['label'] == 1].shape[0]
    total = df.shape[0]
    return {
        "win_rate": win / total if total > 0 else 0,
        "total": total
    }


# ================================
# AI 建议
# ================================
def ai_advice(model, seq):
    prob = model.predict(seq.reshape(1, *seq.shape))[0][0]
    if prob > 0.65:
        return "看多（结构概率优势）"
    elif prob < 0.35:
        return "看空（结构概率劣势）"
    else:
        return "观望"


# ================================
# Streamlit 面板
# ================================
st.set_page_config(page_title="AI 深度分析 V700", layout="wide")

st.title("🧠 AI 深度学习分析 V700")

df = get_data()

if df is not None:
    df = detect_anomaly(df)

    df = make_label(df)
    seq = build_seq(df)

    if len(seq) == 0:
        st.warning("序列数据不足")
        st.stop()

    # 训练模型（轻量）
    model = build_model(seq.shape[1:])
    labels = df['label'].iloc[len(df) - len(seq):].values
    model.fit(seq, labels, epochs=3, batch_size=16, verbose=0)

    # 回测
    bt = backtest(df)

    # AI 推断
    last = seq[-1]
    advice = ai_advice(model, last)

    # 面板
    col1, col2, col3 = st.columns(3)
    col1.metric("回测胜率", f"{bt['win_rate']*100:.1f}%")
    col2.metric("样本数", bt['total'])
    col3.metric("AI 建议", advice)

    st.write("---")

    # 图表
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c']
    )])

    # 异常点
    anom = df[df['anomaly'] == -1]
    fig.add_trace(go.Scatter(
        x=anom.index, y=anom['c'],
        mode='markers',
        marker=dict(size=8),
        name="异常"
    ))

    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("异常统计")
    st.write(anom[['c']].describe())

    st.write("---")
    st.info(f"AI 建议：{advice}")

else:
    st.warning("数据获取失败")
