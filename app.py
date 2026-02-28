import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import plotly.graph_objects as go
from datetime import datetime

# ==================== 1. 职业级数据库架构 ====================
def init_db():
    conn = sqlite3.connect('trading_intelligence.db')
    c = conn.cursor()
    # 建立包含入场参数与结果跟踪的表结构
    c.execute('''CREATE TABLE IF NOT EXISTS signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT, direction TEXT, price REAL, 
                  score INTEGER, confidence REAL, rsi REAL, 
                  atr REAL, net_flow REAL, vol_ratio REAL,
                  result_15m REAL DEFAULT 0)''')
    conn.commit()
    conn.close()

def log_trade(sig, p, score, conf, rsi, atr, nf, vr):
    conn = sqlite3.connect('trading_intelligence.db')
    c = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 简单的分钟内去重
    c.execute("SELECT id FROM signals WHERE ts LIKE ? AND direction = ?", (f"{now[:16]}%", sig))
    if not c.fetchone():
        c.execute("""INSERT INTO signals (ts, direction, price, score, confidence, rsi, atr, net_flow, vol_ratio) 
                     VALUES (?,?,?,?,?,?,?,?,?)""", (now, sig, p, score, conf, rsi, atr, nf, vr))
        conn.commit()
    conn.close()

# ==================== 2. 核心分析模块 (胜率自省) ====================
def get_performance_stats():
    conn = sqlite3.connect('trading_intelligence.db')
    try:
        df = pd.read_sql_query("SELECT * FROM signals", conn)
        if len(df) < 5: return 0, 0 # 样本不足
        # 这里是一个简化的逻辑：如果后期价格同向变动，计为胜
        win_rate = (df['score'] > 7).mean() * 100 # 示例：高分单占比
        return len(df), win_rate
    except: return 0, 0
    finally: conn.close()

# ==================== 3. 修正后的审计引擎 (V130) ====================
def blackbox_engine_v130(df1, df5, net_flow, buy_ratio):
    # [此处继承 V120 的所有修正：ATR, RSI, Shift-1 消除未来函数]
    for d in [df1, df5]:
        for col in ['o', 'h', 'l', 'c', 'v']: d[col] = d[col].astype(float)
        tr = np.maximum(d['h'] - d['l'], np.maximum(abs(d['h'] - d['c'].shift(1)), abs(d['l'] - d['c'].shift(1))))
        d['atr'] = tr.rolling(14).mean()
        delta = d['c'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        d['rsi'] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan))))
        d['ema20'] = d['c'].ewm(span=20, adjust=False).mean()
        d['ema60'] = d['c'].ewm(span=60, adjust=False).mean()

    # 职业级置信度 (基于 Shift-1)
    df1['prev_c'], df1['prev_ema'] = df1['c'].shift(1), df1['ema20'].shift(1)
    df1['signal'] = np.where((df1['prev_c'] > df1['prev_ema']), 1, -1)
    conf = np.clip(50 + (df1['signal'] * df1['c'].pct_change().shift(-1)).tail(30).sum() * 1000, 20, 95)

    curr1, curr5 = df1.iloc[-1], df5.iloc[-1]
    vol_ratio = curr1['v'] / df1['v'].rolling(20).mean().iloc[-1]

    # 双向评分独立
    l_score, s_score = 0, 0
    if (curr5['ema20'] - curr5['ema60']) / curr5['atr'] > 0.8: l_score += 3
    if (curr5['ema20'] - curr5['ema60']) / curr5['atr'] < -0.8: s_score += 3
    if net_flow > 5: l_score += 2
    if net_flow < -5: s_score += 2
    
    # 动能真实性 (严格 AND)
    is_real = vol_ratio > 1.2 and (abs(curr1['c']-curr1['o'])/curr1['atr'] > 0.6)
    if is_real:
        if curr1['c'] > curr1['o']: l_score += 2
        else: s_score += 2

    # 最终信号与入库
    sig = {"dir": None, "score": 0, "type": "🚥 扫描中"}
    threshold = 7 if conf < 50 else 6
    
    # 波动率保护
    if curr1['atr'] > df1['atr'].rolling(50).mean().iloc[-1] * 0.7:
        if l_score >= threshold: sig = {"dir": "LONG", "score": l_score, "type": "🚀 多头入库"}
        elif s_score >= threshold: sig = {"dir": "SHORT", "score": s_score, "type": "🔥 空头入库"}
    
    if sig['dir']:
        log_trade(sig['dir'], curr1['c'], sig['score'], conf, curr1['rsi'], curr1['atr'], net_flow, vol_ratio)

    return sig, l_score, s_score, conf, is_real, curr1['rsi']

# ==================== 4. 终端渲染 ====================
st.set_page_config(page_title="ETH V130 黑匣子终端", layout="wide")
init_db()

# ... [获取数据逻辑 fetch_okx ...]

if 'df1' in locals():
    sig, l_score, s_score, conf, is_real, rsi = blackbox_engine_v130(df1, df5, net_f, buy_r)
    total_logs, high_score_rate = get_performance_stats()

    with st.sidebar:
        st.header("📓 智能日志分析")
        st.metric("累计信号数", total_logs)
        st.metric("高分信号占比", f"{high_score_rate:.1f}%")
        if st.button("清理过期日志"):
            # 这里的清理逻辑可以根据需要添加
            pass
        st.divider()
        st.write(f"📊 当前置信度: `{conf:.1f}%`")
        st.write(f"📈 多头分: `{l_score}` | 📉 空头分: `{s_score}`")

    st.title("🛡️ ETH 职业黑匣子 V130")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("市场信号", sig['type'], delta=f"Score: {sig['score']}")
    c2.metric("动能品质", "💎 真实" if is_real else "🌪️ 噪音")
    c3.metric("RSI 审计", f"{rsi:.1f}")
    c4.metric("策略健康度", "良好" if conf > 50 else "风险")

    # 绘制带审计轨道的图表
    fig = go.Figure(data=[go.Candlestick(x=df1.index, open=df1['o'], high=df1['h'], low=df1['l'], close=df1['c'])])
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
