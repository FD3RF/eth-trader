import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 数据库模块 (核心闭环) ====================
def init_db():
    conn = sqlite3.connect('v160_war_room.db')
    c = conn.cursor()
    # 记录每一个计划的详细参数，方便后期复盘
    c.execute('''CREATE TABLE IF NOT EXISTS tactical_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT, action TEXT, price REAL, 
                  tp REAL, sl REAL, vol_ratio REAL, atr REAL)''')
    conn.commit()
    conn.close()

def log_tactical_plan(action, price, tp, sl, vr, atr):
    if action == "观望": return # 不记录无效等待
    conn = sqlite3.connect('v160_war_room.db')
    c = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 5分钟内不重复记录相同动作
    c.execute("SELECT id FROM tactical_logs WHERE ts > ? AND action = ?", 
              ((datetime.now() - pd.Timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'), action))
    if not c.fetchone():
        c.execute("INSERT INTO tactical_logs (ts, action, price, tp, sl, vol_ratio, atr) VALUES (?,?,?,?,?,?,?)",
                  (now, action, price, tp, sl, vr, atr))
        conn.commit()
    conn.close()

# ==================== 2. 修正后的 V160 引擎 ====================
def tactical_engine_v165(df1, df5, net_flow):
    # (此处继承 V160 截图中的所有逻辑：EMA, ATR, BB)
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['atr'] = (d['h'] - d['l']).rolling(14).mean()
        d['std'] = d['c'].rolling(20).std()
        d['upper'] = d['ema20'] + (d['std'] * 2.2)
        d['lower'] = d['ema20'] - (d['std'] * 2.2)

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    atr_val = curr1['atr']
    vol_avg = df1['v'].rolling(20).mean().iloc[-1]
    v_ratio = curr1['v'] / vol_avg if vol_avg > 0 else 1.0
    
    slope = (df5['ema20'].iloc[-1] - df5['ema20'].iloc[-5]) / 5
    is_trend = abs(slope) > (atr_val * 0.1)

    plan = {"status": "🔭 扫描中", "action": "观望", "entry": None, "tp": None, "sl": None}
    
    # 核心进攻逻辑 (截图 V160 验证成功版)
    if is_trend:
        if curr1['c'] < curr1['ema20'] and net_flow < -0.1:
            plan = {"status": "🔥 破位空头", "action": "做空 (SHORT)", 
                    "entry": curr1['c'], "tp": curr1['c'] - 2.5*atr_val, "sl": curr1['c'] + 1.2*atr_val}
        elif curr1['c'] > curr1['ema20'] and net_flow > 0.1:
            plan = {"status": "🚀 趋势多头", "action": "做多 (LONG)", 
                    "entry": curr1['c'], "tp": curr1['c'] + 2.5*atr_val, "sl": curr1['c'] - 1.2*atr_val}
    
    # 自动入库记录
    if plan['entry']:
        log_tactical_plan(plan['action'], plan['entry'], plan['tp'], plan['sl'], v_ratio, atr_val)
        
    return plan, is_trend, v_ratio, atr_val

# ==================== 3. 终端 UI (带日志侧边栏) ====================
st.set_page_config(page_title="ETH V165 指挥部", layout="wide")
init_db()

# ... [获取数据逻辑 OKX API] ...

if 'df1' in locals():
    plan, is_trend, vol_ratio, atr = tactical_engine_v165(df1, df5, net_f)

    with st.sidebar:
        st.header("📊 战地日志复盘")
        if st.button("查看历史指令单"):
            conn = sqlite3.connect('v160_war_room.db')
            history = pd.read_sql_query("SELECT ts, action, price as 入场点 FROM tactical_logs ORDER BY id DESC LIMIT 10", conn)
            st.dataframe(history)
            conn.close()
        st.divider()
        st.metric("1m 净流", f"{net_f:+.2f}")
        st.write(f"波动率 ATR: `{atr:.2f}`")

    # (此处渲染 V160 截图中的实时作战计划卡片和图表)
    # ... [参考 V160 布局代码] ...
