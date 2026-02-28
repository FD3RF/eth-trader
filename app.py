import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 数据库日志模块 (核心新增) ====================
def init_db():
    conn = sqlite3.connect('trading_log.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT, sig_type TEXT, direction TEXT, 
                  price REAL, score INTEGER, confidence REAL, 
                  rsi REAL, atr REAL, net_flow REAL)''')
    conn.commit()
    return conn

def log_signal(sig_type, direction, price, score, conf, rsi, atr, net_flow):
    conn = sqlite3.connect('trading_log.db')
    c = conn.cursor()
    # 简单的去重逻辑：同一分钟内不重复记录同一方向信号
    now_min = datetime.now().strftime('%Y-%m-%d %H:%M')
    c.execute("SELECT * FROM signals WHERE ts LIKE ? AND direction = ?", (f"{now_min}%", direction))
    if not c.fetchone():
        c.execute("INSERT INTO signals (ts, sig_type, direction, price, score, confidence, rsi, atr, net_flow) VALUES (?,?,?,?,?,?,?,?,?)",
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sig_type, direction, price, score, conf, rsi, atr, net_flow))
        conn.commit()
    conn.close()

# ==================== 2. 职业级算法引擎 (V120 审计版) ====================
def quantum_engine_v125(df1, df5, net_flow, buy_ratio):
    # (此处承接 V120 的所有审计逻辑：ATR修正、RSI标准算法、消除未来函数)
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        tr = np.maximum(d['h'] - d['l'], np.maximum(abs(d['h'] - d['c'].shift(1)), abs(d['l'] - d['c'].shift(1))))
        d['atr'] = tr.rolling(14).mean()
        delta = d['c'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        d['rsi'] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan))))
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()

    # 真实信号回测置信度
    df1['prev_c'], df1['prev_ema'] = df1['c'].shift(1), df1['ema20'].shift(1)
    df1['signal'] = np.where((df1['prev_c'] > df1['prev_ema']), 1, -1)
    df1['real_perf'] = df1['signal'] * df1['c'].pct_change().shift(-1)
    confidence = np.clip(50 + (df1['real_perf'].tail(30).sum() * 1000), 20, 90)

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    is_low_vol = curr1['atr'] < (df1['atr'].rolling(50).mean().iloc[-1] * 0.7)

    # 双向独立评分
    l_score, s_score = 0, 0
    t_gap = (curr5['ema20'] - curr5['ema60']) / curr5['atr']
    if t_gap > 0.8: l_score += 3
    elif t_gap < -0.8: s_score += 3
    if net_flow > 4: l_score += 2
    if net_flow < -4: s_score += 2
    
    # 动能验证 (严格 AND)
    is_real = (curr1['v']/df1['v'].rolling(20).mean().iloc[-1] > 1.2) and (abs(curr1['c']-curr1['o'])/curr1['atr'] > 0.6)
    if is_real:
        if curr1['c'] > curr1['o']: l_score += 2
        else: s_score += 2

    entry_threshold = 7 if confidence < 55 else 6
    sig = {"type": "🚥 观望", "dir": None, "score": 0}

    if not is_low_vol:
        if l_score >= entry_threshold: sig = {"type": "🚀 职业多头", "dir": "LONG", "score": l_score}
        elif s_score >= entry_threshold: sig = {"type": "🔥 职业空头", "dir": "SHORT", "score": s_score}
    
    # 执行日志记录
    if sig['dir']:
        log_signal(sig['type'], sig['dir'], curr1['c'], sig['score'], confidence, curr1['rsi'], curr1['atr'], net_flow)

    return sig, l_score, s_score, confidence, curr1['rsi'], curr1['atr']

# ==================== 3. Streamlit 界面 ====================
st.set_page_config(page_title="ETH V125 日志集成版", layout="wide")
init_db() # 初始化数据库

# (此处 fetch_okx 逻辑同前)
# ... [获取数据 df1, df5, net_f, buy_r] ...

if 'df1' in locals():
    sig, l_score, s_score, conf, rsi, atr = quantum_engine_v125(df1, df5, net_f, buy_r)

    with st.sidebar:
        st.header("📊 统计数据库")
        if st.button("查看历史信号"):
            conn = sqlite3.connect('trading_log.db')
            history = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 10", conn)
            st.table(history)
            conn.close()
        st.divider()
        st.metric("实时置信度", f"{conf:.1f}%")
        st.write(f"多头: `{l_score}` | 空头: `{s_score}`")

    st.title("🛡️ ETH 职业量化终端 V125")
    # ... [渲染指标与图表] ...
