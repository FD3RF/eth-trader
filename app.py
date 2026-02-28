import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================================
# 1. 数据获取（OKX）
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

    if not (k1 and k5 and k15):
        return None

    df1 = pd.DataFrame(k1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df15 = pd.DataFrame(k15, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)

    return df1, df5, df15


# ================================
# 2. 特征工程
# ================================
def build_features(df):
    df = df.copy()
    for col in ['o','h','l','c','v']:
        df[col] = df[col].astype(float)

    # EMA
    df['ema20'] = df['c'].ewm(span=20).mean()
    df['ema60'] = df['c'].ewm(span=60).mean()

    # RSI
    diff = df['c'].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = -diff.clip(upper=0).rolling(14).mean().replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))

    # ATR
    tr = np.maximum(df['h'] - df['l'],
                   np.maximum(abs(df['h'] - df['c'].shift(1)),
                              abs(df['l'] - df['c'].shift(1))))
    df['atr'] = tr.rolling(14).mean()

    # 标签：未来收益是否为正
    df['future'] = df['c'].pct_change().shift(-1)
    df['label'] = (df['future'] > 0).astype(int)

    features = df[['ema20','ema60','rsi','atr','v']].dropna()
    labels = df['label'].loc[features.index]

    return features, labels, df


# ================================
# 3. AI 模型
# ================================
def train_ai(df):
    X, y, _ = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = classification_report(y_test, pred)
    return model, report


def ai_infer(model, df):
    last = df.iloc[-1]

    features = pd.DataFrame([{
        "ema20": last['ema20'],
        "ema60": last['ema60'],
        "rsi": last['rsi'],
        "atr": last['atr'],
        "v": last['v']
    }])

    prob = model.predict_proba(features)[0][1]

    return {
        "confidence": prob,
        "suggestion": "看多" if prob > 0.6 else "看空" if prob < 0.4 else "观望"
    }


# ================================
# 4. 多周期评分
# ================================
def score(df1, df5, df15):
    def calc(df):
        df = build_features(df)[2]
        last = df.iloc[-1]
        score = 0
        if last['ema20'] > last['ema60']: score += 1
        if last['rsi'] < 30 or last['rsi'] > 70: score += 1
        if last['v'] > df['v'].rolling(20).mean().iloc[-1]: score += 1
        return score

    return {
        "high": calc(df1),
        "swing": calc(df5),
        "trend": calc(df15)
    }


# ================================
# 5. Streamlit 面板
# ================================
st.set_page_config(page_title="AI 分析 V600", layout="wide")

data = get_multi()
if data:
    df1, df5, df15 = data

    # AI 训练（轻量）
    model, report = train_ai(df1)

    # AI 推断
    _, _, df1b = build_features(df1)
    ai = ai_infer(model, df1b)

    # 评分
    sc = score(df1, df5, df15)

    st.title("🧠 AI 多周期分析 V600（只分析）")

    col1, col2, col3 = st.columns(3)
    col1.metric("高频评分", sc['high'])
    col2.metric("波段评分", sc['swing'])
    col3.metric("趋势评分", sc['trend'])

    st.metric("AI 置信度", f"{ai['confidence']*100:.1f}%")
    st.metric("AI 建议", ai['suggestion'])

    st.write("---")
    st.subheader("模型报告")
    st.code(report)

    # 图表
    fig = go.Figure(data=[go.Candlestick(
        x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c']
    )])
    fig.add_trace(go.Scatter(x=df1.index, y=df1b['ema20'], name="EMA20"))
    fig.add_trace(go.Scatter(x=df1.index, y=df1b['ema60'], name="EMA60"))
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("数据获取失败")
