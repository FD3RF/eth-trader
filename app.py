import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V70.0 顶级全能终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 指挥官核心引擎 ====================
def commander_engine_v70(df):
    # 指标计算：EMA + ATR + RSI
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['c'].ewm(span=60, adjust=False).mean()
    df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    atr = df['atr'].iloc[-1]
    
    # 偏离度与动能指标
    df['dev'] = (df['c'] - df['ema20']).abs() / atr
    delta = df['c'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()))
    
    # --- 市场模式识别 (结构化) ---
    curr, prev = df.iloc[-1], df.iloc[-2]
    hh = df['h'].iloc[-1] > df['h'].iloc[-3]
    hl = df['l'].iloc[-1] > df['l'].iloc[-3]
    ll = df['l'].iloc[-1] < df['l'].iloc[-3]
    lh = df['h'].iloc[-1] < df['h'].iloc[-3]
    
    if curr['ema20'] > curr['ema60'] and hh and hl: mode = "强势多头"
    elif curr['ema20'] < curr['ema60'] and ll and lh: mode = "强势空头"
    else: mode = "横盘震荡"

    # --- 因子评分系统 (三引擎并行) ---
    signal = {"type": None, "dir": None, "score": 0}
    vol_mean = df['v'].rolling(20).mean().iloc[-1]
    
    # 引擎 A：趋势回踩 (评分门槛 5)
    if "强势" in mode:
        score = 3 # 结构基础分
        if curr['v'] > vol_mean: score += 2
        if (mode == "强势多头" and curr['l'] <= curr['ema20']) or (mode == "强势空头" and curr['h'] >= curr['ema20']):
            if score >= 5: signal = {"type": "趋势回踩", "dir": "LONG" if "多头" in mode else "SHORT", "score": score}

    # 引擎 B：横盘突破 (针对震荡市，评分门槛 6)
    if signal['type'] is None and mode == "横盘震荡":
        if curr['c'] > df['h'].rolling(30).mean().iloc[-2] and curr['v'] > vol_mean * 1.5:
            signal = {"type": "区间突破", "dir": "LONG", "score": 7}
        elif curr['c'] < df['l'].rolling(30).mean().iloc[-2] and curr['v'] > vol_mean * 1.5:
            signal = {"type": "区间突破", "dir": "SHORT", "score": 7}

    # 引擎 C：乖离反转 (下调偏离度门槛至 1.5)
    if signal['type'] is None and curr['dev'] > 1.5:
        if curr['rsi'] < 30: signal = {"type": "极端反转", "dir": "LONG", "score": 8}
        elif curr['rsi'] > 70: signal = {"type": "极端反转", "dir": "SHORT", "score": 8}

    return signal, atr, mode, curr['dev']

# ==================== 3. 自适应执行计划 ====================
def generate_v70_plan(curr_p, sig, atr, mode):
    if not sig['type']:
        return {"title": "📡 深度扫描中", "tag": "暂无高分因子", "color": "#888", "why": "当前波动率不足或方向不明，建议继续保持空仓观察。", "entry":0,"sl":0,"tp":0}

    # 动态风控参数
    sl_coef = 1.2 if "强势" in mode else 1.5
    tp_coef = 2.5 if "强势" in mode else 1.8
    
    sl_dist, tp_dist = sl_coef * atr, tp_coef * atr
    entry = curr_p
    
    if sig['dir'] == "LONG":
        sl, tp = entry - sl_dist, entry + tp_dist
        color = "#bf94ff" if "趋势" in mode else "#00ffcc"
    else:
        sl, tp = entry + sl_dist, entry - tp_dist
        color = "#ff4b4b"

    return {
        "title": f"⚡ {sig['type']} ({sig['dir']})",
        "tag": f"因子分: {sig['score']} | {mode}",
        "color": color,
        "why": f"基于{mode}环境，已动态调整 ATR 风控系数。止盈目标设定在 {tp_coef} 倍波动区间。",
        "entry": entry, "sl": sl, "tp": tp
    }

# ==================== 4. 终端 UI 渲染 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=300")
if k_raw:
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    
    sig, atr, mode, dev = commander_engine_v70(df)
    curr_p = df.iloc[-1]['c']
    
    with st.sidebar:
        st.header("🦅 V70.0 顶级指挥部")
        st.metric("市场环境", mode)
        st.metric("偏离度", f"{dev:.2f}")
        
        plan = generate_v70_plan(curr_p, sig, atr, mode)
        st.markdown(f"""
            <div style="border:4px solid {plan['color']}; padding:15px; border-radius:12px; background:rgba(0,0,0,0.2)">
                <h2 style="color:{plan['color']}; margin:0">{plan['title']}</h2>
                <p style="color:{plan['color']}"><b>{plan['tag']}</b></p>
                <hr style="border:0.5px solid #444">
                <p style="font-size:13px">{plan['why']}</p>
                <p style="margin:5px 0">📍 <b>入场点:</b> ${plan['entry']:.2f}</p>
                <p style="margin:5px 0; color:#ffbcbc">❌ <b>止损点:</b> ${plan['sl']:.2f}</p>
                <p style="margin:5px 0; color:#bcffbc">💰 <b>止盈点:</b> ${plan['tp']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        st.write("---")
        st.caption("指挥官策略：趋势做回踩，震荡做突破，极端做反转。")

    st.title(f"🛡️ ETH 指挥官终端 V70.0")
    c1, c2, c3 = st.columns(3)
    c1.metric("当前价格", f"${curr_p}")
    c2.metric("波动率 (ATR)", f"{atr:.2f}")
    c3.metric("RSI 指数", f"{df['rsi'].iloc[-1]:.1f}")

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="行情")])
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], line=dict(color='yellow', width=1.5), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema60'], line=dict(color='cyan', width=1), name="EMA60"))
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
