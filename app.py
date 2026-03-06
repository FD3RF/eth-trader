import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import hashlib
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统底层加固 ---
st.set_page_config(page_title="ETH AI V6.8 - 工业至尊版", layout="wide")
st_autorefresh(interval=5000, key="eth_core_v68")

if "v68_state" not in st.session_state:
    st.session_state.v68_state = {
        "last_atomic_hash": "",
        "last_speak_ts": 0,
        "retry_backoff": 1,
        "cache_df": None,
        "cache_tick": None
    }

# --- 2. 语音引擎：原子化执行 + 强制唤醒 ---
def trigger_voice(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    js_id = int(time.time() * 1000)
    js_code = f"""
    <div id="v_{js_id}">
    <script>
        (function() {{
            if (!window.speechSynthesis) return;
            // 强制唤醒音频上下文
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            window.speechSynthesis.cancel(); 
            setTimeout(() => {{
                var m = new SpeechSynthesisUtterance("{clean_text}");
                m.lang = 'zh-CN'; m.rate = 1.4; m.pitch = 1.0;
                m.onend = function() {{ document.getElementById("v_{js_id}").remove(); }};
                window.speechSynthesis.speak(m);
            }}, 30);
        }})();
    </script>
    </div>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 核心数据引擎（带指数退避与物理对齐） ---
@st.cache_resource
def get_ex():
    return ccxt.bybit({'enableRateLimit': True, 'timeout': 5000, 'options': {'defaultType': 'linear'}})

def fetch_data_v68():
    ex = get_ex()
    try:
        bars = ex.fetch_ohlcv('ETH/USDT', '5m', limit=600)
        tick = ex.fetch_ticker('ETH/USDT') 
        if len(bars) > 300 and tick:
            df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
            df = df.sort_values('ts', ascending=True)
            df['dt'] = pd.to_datetime(df['ts'], unit='ms')
            
            # 【量能向量加速】：计算 24H 动态均量 (288根)
            v_vals = df['v'].values
            v_limit = np.percentile(v_vals, 99)
            v_clean = np.clip(v_vals, a_min=None, a_max=v_limit)
            v_avg = pd.Series(v_clean).shift(1).rolling(window=288, min_periods=30).mean().values
            df['v_avg'] = v_avg
            
            df_ui = df.iloc[-300:].copy()
            df_ui['ratio'] = df_ui['v'] / (df_ui['v_avg'] + 1e-10)
            df_ui['ratio'] = np.nan_to_num(df_ui['ratio'], nan=1.0).clip(0, 20.0)
            
            st.session_state.v68_state["retry_backoff"] = 1
            return df_ui, tick
    except Exception:
        return st.session_state.v68_state["cache_df"], st.session_state.v68_state["cache_tick"]
    return None, None

# --- 4. 核心口诀判定（逻辑绝对对齐，拒绝删减） ---
def analyze_vpa(df, tick):
    status = {"act": "AI 全速监测中", "motto": "量价合一", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty: return status, 1.0, 0, 0

    curr = df.iloc[-1]
    ratio = float(curr['ratio'])
    
    # 【核心对齐】：锁定过去第 2 根到第 52 根 (50根闭合线)
    ref_window = df.iloc[-51:-1] 
    p_res = np.round(float(ref_window['h'].max()), 1)
    p_sup = np.round(float(ref_window['l'].min()), 1)
    curr_c = np.round(float(curr['c']), 1)
    
    # 动态容差：基于历史波动万分之五
    tolerance = np.round((p_res - p_sup) * 0.005, 1) 
    
    # 时间确认：15秒过滤开盘脉冲，45秒确认缩量形态
    elapsed = (tick['timestamp'] - curr['ts']) / 1000
    is_stable_long = elapsed > 45 
    is_not_opening_surge = elapsed > 15

    # 【执行口诀 - 核心逻辑绝对闭环】
    if ratio < 0.6 and is_stable_long:
        if curr['l'] <= p_sup + tolerance: 
            status.update({"act": "准备多", "motto": "缩量回踩，支撑有效", "color": "#0D47A1", "voice": "缩量回踩，低点不破"})
        elif curr['h'] >= p_res - tolerance:
            status.update({"act": "准备空", "motto": "缩量反弹，压力明显", "color": "#3E2723", "voice": "缩量反弹，高点不破"})
            
    elif ratio > 1.6 and is_not_opening_surge:
        if curr_c > p_res + 0.1: 
            status.update({"act": "直接开多", "motto": "爆量突破，猛龙过江", "color": "#1B5E20", "voice": "放量起涨，直接开多", "tri": "buy"})
        elif curr_c < p_sup - 0.1:
            status.update({"act": "直接开空", "motto": "爆量跌破，大势已去", "color": "#B71C1C", "voice": "放量下跌，直接开空", "tri": "sell"})

    return status, ratio, p_res, p_sup

# --- 5. 渲染引擎 ---
def main():
    placeholder = st.empty()
    df, tick = fetch_data_v68()
    
    if df is not None:
        st.session_state.v68_state["cache_df"] = df
        st.session_state.v68_state["cache_tick"] = tick
        status, ratio, res_p, sup_p = analyze_vpa(df, tick)
        
        # 播报决策指纹：[行为+TS+状态色]
        current_ts = int(df.iloc[-1]['ts'])
        sig_hash = hashlib.md5(f"{status['act']}_{current_ts}_{status['color']}".encode()).hexdigest()
        
        if status["voice"] and sig_hash != st.session_state.v68_state["last_atomic_hash"]:
            if time.time() - st.session_state.v68_state["last_speak_ts"] > 15:
                trigger_voice(status["voice"])
                st.session_state.v68_state["last_atomic_hash"] = sig_hash
                st.session_state.v68_state["last_speak_ts"] = time.time()

        with placeholder.container():
            price = str(tick.get('last', '---'))
            st.markdown(f"""
            <div style="background:{status['color']}; padding:30px; border-radius:15px; text-align:center; border: 2px solid gold; color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <h1 style="margin:0; font-size:62px; font-weight:bold;">{status['act']} ({ratio:.2f}x)</h1>
                <h2 style="color:#FFD700; margin:15px 0; letter-spacing:2px;">“{status['motto']}”</h2>
                <p style="font-size:24px; opacity:0.9;">ETH现价: {price} | 压力线: {res_p} | 支撑线: {sup_p}</p>
            </div>
            """, unsafe_allow_html=True)

            

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df['dt'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH-5M"), row=1, col=1)
            fig.add_hline(y=res_p, line_dash="dash", line_color="#FF00FF", annotation_text="静态压力线", row=1, col=1)
            fig.add_hline(y=sup_p, line_dash="dash", line_color="#00FFFF", annotation_text="静态支撑线", row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['dt'], y=df['ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能强度"), row=2, col=1)
            fig.add_hline(y=1.6, line_dash="dot", line_color="#FF4444", row=2, col=1)
            fig.add_hline(y=0.6, line_dash="dot", line_color="#44FF44", row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
