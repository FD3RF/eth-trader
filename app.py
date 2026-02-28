import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import sqlite3
from datetime import datetime

# ================================
# 1. 数据层（实时获取）
# ================================
def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=3).json()
        return r.get('data', []) if r.get('code') == '0' else []
    except:
        return []

def get_data():
    k1 = fetch_okx("market/candles", "&bar=1m&limit=100")
    k5 = fetch_okx("market/candles", "&bar=5m&limit=100")
    t  = fetch_okx("market/trades", "&limit=100")

    if not (k1 and k5 and t):
        return None

    df1 = pd.DataFrame(k1, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t)

    return df1, df5, tdf


# ================================
# 2. 趋势诊断（高级结构）
# ================================
def diagnose_trend(df1, df5):
    for d in [df1, df5]:
        for col in ['o','h','l','c','v']:
            d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20).mean()
        d['ema60'] = d['c'].ewm(span=60).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + (
            (d['c'].diff().clip(lower=0)).rolling(14).mean() /
            (-d['c'].diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        )))

    c1 = df1.iloc[-1]
    c5 = df5.iloc[-1]

    # 趋势强度
    slope = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    trend = "多头" if slope > 0 else "空头"

    # 波动与风险
    volatility = c1['atr']
    risky = volatility > df1['atr'].rolling(50).mean().iloc[-1] * 1.3

    # RSI 状态
    rsi_state = "超买" if c1['rsi'] > 70 else "超卖" if c1['rsi'] < 30 else "中性"

    return {
        "trend": trend,
        "slope": slope,
        "rsi": c1['rsi'],
        "rsi_state": rsi_state,
        "volatility": volatility,
        "risky": risky,
        "ema20": c1['ema20'],
        "ema60": c1['ema60']
    }


# ================================
# 3. 统计闭环（只分析）
# ================================
def init_db():
    conn = sqlite3.connect('analysis.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stats
                 (ts TEXT, trend TEXT, slope REAL, rsi REAL, vol REAL, risky INTEGER)''')
    conn.commit()
    return conn

def log_stats(info):
    conn = sqlite3.connect('analysis.db')
    c = conn.cursor()
    c.execute("""INSERT INTO stats (ts, trend, slope, rsi, vol, risky)
                 VALUES (?,?,?,?,?,?)""",
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               info['trend'], info['slope'], info['rsi'],
               info['volatility'], int(info['risky'])))
    conn.commit()
    conn.close()

def stats_report():
    conn = sqlite3.connect('analysis.db')
    df = pd.read_sql_query("SELECT * FROM stats ORDER BY ts DESC", conn)
    conn.close()

    if df.empty:
        return "暂无统计"

    # 胜率式统计（趋势一致率）
    total = len(df)
    trend_ok = (df['trend'] == df['trend'].shift(1)).sum()
    stability = trend_ok / (total-1) if total > 1 else 0

    risky_rate = df['risky'].mean()

    return {
        "records": total,
        "stability": stability,
        "risky_rate": risky_rate
    }


# ================================
# 4. Streamlit 面板
# ================================
st.set_page_config(page_title="实时分析面板 V200", layout="wide")
init_db()

data = get_data()
if data:
    df1, df5, tdf = data
    info = diagnose_trend(df1, df5)
    log_stats(info)

    st.title("📊 实时分析面板 V200（只分析）")

    col1, col2, col3 = st.columns(3)
    col1.metric("趋势", info['trend'])
    col2.metric("RSI", f"{info['rsi']:.1f}", info['rsi_state'])
    col3.metric("波动率", f"{info['volatility']:.2f}")

    st.write("---")
    st.subheader("趋势诊断")

    st.write(f"""
    - 趋势方向：{info['trend']}
    - 斜率：{info['slope']:.4f}
    - RSI：{info['rsi']:.1f}（{info['rsi_state']}）
    - 风险信号：{'⚠️ 高波动' if info['risky'] else '✅ 正常'}
    """)

    # 图表
    fig = go.Figure(data=[go.Candlestick(
        x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c']
    )])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], name="EMA20", line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema60'], name="EMA60", line=dict(color='cyan')))
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 统计闭环
    st.write("---")
    st.subheader("统计闭环")

    report = stats_report()
    st.write(report)

else:
    st.warning("数据获取失败，等待重试...")
