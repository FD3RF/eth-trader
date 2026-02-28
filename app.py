import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==================== 1. 系统核心配置 ====================
st.set_page_config(page_title="ETH V68.0 A+顶级终端", layout="wide")

def fetch_okx_data(endpoint, params=""):
    url = f"https://www.okx.com/api/v5/{endpoint}?instId=ETH-USDT{params}"
    try:
        with requests.Session() as s:
            s.trust_env = True  
            r = s.get(url, timeout=5).json()
            return r if r.get('code') == '0' else None
    except: return None

# ==================== 2. 核心工程引擎：评分与自适应风控 ====================
def master_engine_v68(df):
    # --- 基础指标与ATR ---
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['c'].ewm(span=60, adjust=False).mean()
    df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    atr = df['atr'].iloc[-1]
    
    # RSI 与 偏离度
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['deviation'] = (df['c'] - df['ema20']).abs() / atr
    
    # --- 升级1：结构趋势确认 (HH/HL) ---
    hh = df['h'].iloc[-1] > df['h'].iloc[-3]
    hl = df['l'].iloc[-1] > df['l'].iloc[-3]
    ll = df['l'].iloc[-1] < df['l'].iloc[-3]
    lh = df['h'].iloc[-1] < df['h'].iloc[-3]
    
    # --- 升级2：市场模式识别 (基于结构) ---
    vol_mean = df['v'].rolling(20).mean().iloc[-1]
    is_strong_vol = df['v'].iloc[-1] > vol_mean
    
    if df['ema20'].iloc[-1] > df['ema60'].iloc[-1] and hh and hl:
        market_mode = "趋势多头"
    elif df['ema20'].iloc[-1] < df['ema60'].iloc[-1] and ll and lh:
        market_mode = "趋势空头"
    else:
        market_mode = "区间震荡"

    # --- 升级3：因子评分系统 (Factor Scoring) ---
    score = 0
    curr = df.iloc[-1]
    signal = {"type": None, "dir": None, "score": 0}
    
    # A. 趋势引擎因子
    if "趋势" in market_mode:
        if market_mode == "趋势多头" and curr['l'] <= curr['ema20'] and curr['c'] > curr['ema20']:
            score += 3 # 结构成立
            if is_strong_vol: score += 2 # 成交量配合
            if curr['rsi'] < 65: score += 2 # 动能未衰竭
            if score >= 5: signal = {"type": "趋势回踩", "dir": "LONG", "score": score}
            
        elif market_mode == "趋势空头" and curr['h'] >= curr['ema20'] and curr['c'] < curr['ema20']:
            score += 3
            if is_strong_vol: score += 2
            if curr['rsi'] > 35: score += 2
            if score >= 5: signal = {"type": "趋势回踩", "dir": "SHORT", "score": score}

    # B. 反转引擎因子 (加入偏离度过滤)
    else:
        if curr['deviation'] > 2.0: # 必须严重偏离均线
            if curr['rsi'] < 30 and curr['v'] < vol_mean:
                score = 6
                signal = {"type": "乖离反转", "dir": "LONG", "score": score}
            elif curr['rsi'] > 70 and curr['v'] < vol_mean:
                score = 6
                signal = {"type": "乖离反转", "dir": "SHORT", "score": score}

    return signal, atr, market_mode, curr['deviation']

# ==================== 3. 自适应风控渲染 ====================
def generate_v68_plan(curr_p, signal, atr, mode):
    if not signal['type'] or signal['score'] < 5:
        return {"title": "⚪ 扫描中", "tag": "评分未达标", "color": "#888", "why": "因子分值不足，系统强制拦截过滤。", "entry":0,"sl":0,"tp":0}

    # --- 升级4：模式自适应 ATR 风控 ---
    if "趋势" in mode:
        sl_coef, tp_coef = 1.2, 2.5  # 趋势模式：窄止损，大止盈
    else:
        sl_coef, tp_coef = 1.5, 1.8  # 震荡模式：宽止损，快止盈

    sl_dist = sl_coef * atr
    tp_dist = tp_coef * atr
    
    if signal['dir'] == "LONG":
        entry, sl, tp = curr_p, curr_p - sl_dist, curr_p + tp_dist
        color = "#bf94ff" if "趋势" in mode else "#00ffcc"
    else:
        entry, sl, tp = curr_p, curr_p + sl_dist, curr_p - tp_dist
        color = "#ff4b4b"

    return {
        "title": f"🔥 {signal['type']} ({signal['dir']})",
        "tag": f"Score: {signal['score']} | {mode}",
        "color": color,
        "why": f"偏离度确认。当前模式 {mode}，ATR 风险系数已调整为 SL:{sl_coef}/TP:{tp_coef}。",
        "entry": entry, "sl": sl, "tp": tp
    }

# ==================== 4. 主逻辑渲染 ====================
k_raw = fetch_okx_data("market/candles", "&bar=5m&limit=400")
t_raw = fetch_okx_data("market/trades", "&limit=100")

if k_raw:
    df = pd.DataFrame(k_raw['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQ','confirm'])
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df = df[::-1].reset_index(drop=True)
    
    sig, atr, mode, dev = master_engine_v68(df)
    curr_p = df.iloc[-1]['c']
    
    with st.sidebar:
        st.header("🦅 A+级顶级指挥部 V68.0")
        st.info(f"市场模式: {mode}")
        st.write(f"均线偏离度: {dev:.2f} (建议>2.0)")
        
        # 冷却机制建议 (UI提示)
        st.warning("⏱️ 信号冷却中: 10根K线内禁止重复开仓" if st.session_state.get('last_idx', 0) > len(df)-10 else "🟢 信号发射就绪")
        
        plan = generate_v68_plan(curr_p, sig, atr, mode)
        if plan['entry'] > 0:
            st.session_state['last_idx'] = len(df) # 记录最后一次信号位置
        
        st.markdown(f"""
            <div style="border:4px solid {plan['color']}; padding:15px; border-radius:12px; background:rgba(255,255,255,0.05)">
                <h1 style="margin:0; color:{plan['color']}; font-size:18px">{plan['title']}</h1>
                <p style="color:{plan['color']}; font-size:12px"><b>{plan['tag']}</b></p>
                <p style="font-size:13px;">💬 <b>老大哥点评：</b>{plan['why']}</p>
                <hr style="border:0.5px solid #444">
                <p style="margin:5px 0">📍 <b>进场：</b>${plan['entry']:.2f}</p>
                <p style="margin:5px 0; color:#ffbcbc">❌ <b>止损：</b>${plan['sl']:.2f}</p>
                <p style="margin:5px 0; color:#bcffbc">💰 <b>目标：</b>${plan['tp']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    st.title("🛡️ ETH A+ 级工程化终端 V68.0")
    col1, col2, col3 = st.columns(3)
    col1.metric("市场模式", mode)
    col2.metric("当前偏离度", f"{dev:.2f}")
    col3.metric("ATR 波动", f"{atr:.2f}")

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线")])
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], line=dict(color='yellow', width=1.5), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema60'], line=dict(color='cyan', width=1), name="EMA60"))
    fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("正在同步顶级行情数据...")
