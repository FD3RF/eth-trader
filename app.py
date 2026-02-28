import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==================== 1. 系统核心配置与数据库初始化 ====================
st.set_page_config(page_title="ETH V135 职业自感知终端", layout="wide")

def init_db():
    conn = sqlite3.connect('trading_intelligence.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT, direction TEXT, price REAL, 
                  score INTEGER, confidence REAL, rsi REAL, 
                  atr REAL, net_flow REAL, result_15m REAL DEFAULT 0)''')
    conn.commit()
    conn.close()

def log_trade(sig, p, score, conf, rsi, atr, nf):
    conn = sqlite3.connect('trading_intelligence.db')
    c = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("SELECT id FROM signals WHERE ts LIKE ? AND direction = ?", (f"{now[:16]}%", sig))
    if not c.fetchone():
        c.execute("""INSERT INTO signals (ts, direction, price, score, confidence, rsi, atr, net_flow) 
                     VALUES (?,?,?,?,?,?,?,?)""", (now, sig, p, score, conf, rsi, atr, nf))
        conn.commit()
    conn.close()

# ==================== 2. 职业审计级引擎 (V135) ====================
def quantum_engine_v135(df1, df5, net_flow, buy_ratio):
    # --- 指标预处理 (严格审计标准) ---
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        tr = np.maximum(d['h'] - d['l'], np.maximum(abs(d['h'] - d['c'].shift(1)), abs(d['l'] - d['c'].shift(1))))
        d['atr'] = tr.rolling(14).mean()
        delta = d['c'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        d['rsi'] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan))))
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()

    # 消除未来函数的回测置信度
    df1['prev_c'], df1['prev_ema'] = df1['c'].shift(1), df1['ema20'].shift(1)
    df1['signal'] = np.where((df1['prev_c'] > df1['prev_ema']), 1, -1)
    conf = np.clip(50 + (df1['signal'] * df1['c'].pct_change().shift(-1)).tail(30).sum() * 1000, 20, 95)

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    vol_ratio = curr1['v'] / df1['v'].rolling(20).mean().iloc[-1]

    # 双向评分独立 (审计修正)
    l_score, s_score = 0, 0
    t_gap = (curr5['ema20'] - curr5['ema60']) / curr5['atr']
    if t_gap > 0.8: l_score += 3
    elif t_gap < -0.8: s_score += 3
    if net_flow > 5: l_score += 2
    if net_flow < -5: s_score += 2
    
    is_real = vol_ratio > 1.2 and (abs(curr1['c']-curr1['o'])/curr1['atr'] > 0.6)
    if is_real:
        if curr1['c'] > curr1['o']: l_score += 2
        else: s_score += 2

    # 动态门槛与信号触发
    sig = {"dir": None, "score": 0, "type": "🚥 扫描中"}
    threshold = 7 if conf < 50 else 6
    
    if curr1['atr'] > df1['atr'].rolling(50).mean().iloc[-1] * 0.7:
        if l_score >= threshold: sig = {"dir": "LONG", "score": l_score, "type": "🚀 多头点火"}
        elif s_score >= threshold: sig = {"dir": "SHORT", "score": s_score, "type": "🔥 空头下破"}
    
    if sig['dir']:
        log_trade(sig['dir'], curr1['c'], sig['score'], conf, curr1['rsi'], curr1['atr'], net_flow)

    return sig, l_score, s_score, conf, is_real, curr1['rsi'], curr1['atr']

# ==================== 3. 辅助函数 ====================
def fetch_okx(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        r = requests.get(url, timeout=5).json()
        return r if r.get('code') == '0' else None
    except: return None

# ==================== 4. UI 界面布局 ====================
init_db()
k1_raw = fetch_okx("market/candles", "&bar=1m&limit=100")
k5_raw = fetch_okx("market/candles", "&bar=5m&limit=100")
t_raw = fetch_okx("market/trades", "&limit=50")

if k1_raw and k5_raw and t_raw:
    df1 = pd.DataFrame(k1_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])[::-1].reset_index(drop=True)
    df5 = pd.DataFrame(k5_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])[::-1].reset_index(drop=True)
    tdf = pd.DataFrame(t_raw['data'])
    net_f = tdf[tdf['side']=='buy']['sz'].astype(float).sum() - tdf[tdf['side']=='sell']['sz'].astype(float).sum()
    buy_r = (tdf[tdf['side']=='buy']['sz'].astype(float).sum() / tdf['sz'].astype(float).sum()) * 100
    
    sig, l_score, s_score, conf, is_real, rsi, atr = quantum_engine_v135(df1, df5, net_f, buy_r)

    with st.sidebar:
        st.header("🧠 核心自感知")
        st.metric("策略置信度", f"{conf:.1f}%")
        st.write(f"📈 多头分: `{l_score}` | 📉 空头分: `{s_score}`")
        st.divider()
        if st.button("查看黑匣子数据"):
            conn = sqlite3.connect('trading_intelligence.db')
            st.dataframe(pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 5", conn))
            conn.close()

    st.title("🛡️ ETH 职业自感知终端 V135")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1min 净流", f"{net_f:+.2f} ETH")
    c2.metric("审计 RSI", f"{rsi:.1f}")
    c3.metric("ATR 波动", f"{atr:.2f}")
    c4.metric("品质校验", "REAL" if is_real else "NOISE")

    if sig['dir']:
        color = "#00FFCC" if sig['dir'] == "LONG" else "#FF4B4B"
        st.markdown(f"<div style='background:{color}33; border:2px solid {color}; padding:20px; border-radius:10px; text-align:center;'><h2>{sig['type']} (Score: {sig['score']})</h2><p>动态准入已通过，信号已记入黑匣子</p></div>", unsafe_allow_html=True)

    fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'], name="K线")])
    fig.add_trace(go.Scatter(x=df1.index, y=df1['ema20'], line=dict(color='yellow', width=1), name="EMA20"))
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("正在同步 OKX 职业数据流...")
