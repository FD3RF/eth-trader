import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import hashlib
from streamlit_autorefresh import st_autorefresh

# --- 1. 底层加固：公网环境优化 ---
st.set_page_config(page_title="ETH AI V7.1 - 公网稳定版", layout="wide")
st_autorefresh(interval=5000, key="eth_core_v71_public")

if "v71_state" not in st.session_state:
    st.session_state.v71_state = {
        "last_atomic_hash": "",
        "last_speak_ts": 0,
        "cache_df": None,
        "cache_tick": None,
        "error_count": 0
    }

# --- 2. 语音引擎：强制唤醒 (防止浏览器静默) ---
def trigger_voice(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    js_code = f"""
    <script>
        (function() {{
            if (!window.speechSynthesis) return;
            window.speechSynthesis.cancel(); 
            var m = new SpeechSynthesisUtterance("{clean_text}");
            m.lang = 'zh-CN'; m.rate = 1.4;
            window.speechSynthesis.speak(m);
        }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 核心数据引擎（防崩断机制） ---
@st.cache_resource
def get_ex():
    # 增加超时时间到 15秒，防止公网加载慢导致报错
    return ccxt.bybit({'enableRateLimit': True, 'timeout': 15000, 'options': {'defaultType': 'linear'}})

def fetch_data_v71():
    ex = get_ex()
    try:
        bars = ex.fetch_ohlcv('ETH/USDT', '5m', limit=500)
        tick = ex.fetch_ticker('ETH/USDT') 
        if bars and len(bars) >= 55:
            df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
            df['dt'] = pd.to_datetime(df['ts'], unit='ms')
            
            # 计算 24H 动态均量 (288根)
            v_vals = df['v'].values
            v_avg = pd.Series(v_vals).shift(1).rolling(window=288, min_periods=30).mean().values
            df['v_avg'] = v_avg
            
            df_ui = df.iloc[-200:].copy()
            df_ui['ratio'] = df_ui['v'] / (df_ui['v_avg'] + 1e-10)
            df_ui['ratio'] = np.nan_to_num(df_ui['ratio'], nan=1.0).clip(0, 20.0)
            
            st.session_state.v71_state["error_count"] = 0 # 重置错误计数
            return df_ui, tick
    except Exception:
        st.session_state.v71_state["error_count"] += 1
        return st.session_state.v71_state["cache_df"], st.session_state.v71_state["cache_tick"]
    return None, None

# --- 4. 核心口诀判定（逻辑绝对对齐） ---
def analyze_vpa(df, tick):
    status = {"act": "公网监测中", "motto": "扫描量价逻辑", "color": "#121212", "voice": "", "tri": None}
    if df is None or len(df) < 55: return status, 1.0, 0, 0

    curr = df.iloc[-1]
    ratio = float(curr['ratio'])
    ref_window = df.iloc[-51:-1].copy()
    p_res = np.round(float(ref_window['h'].max()), 1)
    p_sup = np.round(float(ref_window['l'].min()), 1)
    curr_c = np.round(float(curr['c']), 1)
    
    tolerance = np.round((p_res - p_sup) * 0.005, 1) # 0.5% 容差
    
    elapsed = (tick['timestamp'] - curr['ts']) / 1000
    is_stable = elapsed > 45 

    # 口诀：缩量回踩不破/爆量突破
    if ratio < 0.6 and is_stable:
        if curr['l'] <= p_sup + tolerance: status.update({"act": "准备多", "motto": "缩量回踩支撑", "color": "#0D47A1", "voice": "缩量回踩，支撑有效"})
        elif curr['h'] >= p_res - tolerance: status.update({"act": "准备空", "motto": "缩量反弹压力", "color": "#3E2723", "voice": "缩量反弹，压力明显"})
    elif ratio > 1.6:
        if curr_c > p_res + 0.1: status.update({"act": "直接开多", "motto": "放量突破压力", "color": "#1B5E20", "voice": "爆量突破，直接开多"})
        elif curr_c < p_sup - 0.1: status.update({"act": "直接开空", "motto": "放量跌破支撑", "color": "#B71C1C", "voice": "爆量跌破，直接开空"})

    return status, ratio, p_res, p_sup

# --- 5. 渲染引擎 ---
def main():
    placeholder = st.empty()
    df, tick = fetch_data_v71()
    
    if df is not None:
        st.session_state.v71_state["cache_df"] = df
        st.session_state.v71_state["cache_tick"] = tick
        status, ratio, res_p, sup_p = analyze_vpa(df, tick)
        
        # 播报指纹锁
        sig_hash = hashlib.md5(f"{status['act']}_{int(df.iloc[-1]['ts'])}".encode()).hexdigest()
        if status["voice"] and sig_hash != st.session_state.v71_state["last_atomic_hash"]:
            trigger_voice(status["voice"])
            st.session_state.v71_state["last_atomic_hash"] = sig_hash

        with placeholder.container():
            price = str(tick.get('last', '---'))
            err_msg = f" | ⚠️ 网络不稳定 (重试 {st.session_state.v71_state['error_count']} 次)" if st.session_state.v71_state["error_count"] > 0 else ""
            
            st.markdown(f"""
            <div style="background:{status['color']}; padding:30px; border-radius:15px; text-align:center; border: 2px solid gold; color: white;">
                <h1 style="margin:0; font-size:60px; font-weight:bold;">{status['act']} ({ratio:.2f}x)</h1>
                <h2 style="color:#FFD700; margin:15px 0;">“{status['motto']}”</h2>
                <p style="font-size:18px;">ETH现价: {price} | 阻力: {res_p} | 支撑: {sup_p}{err_msg}</p>
            </div>
            """, unsafe_allow_html=True)

            

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df['dt'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH-5M"), row=1, col=1)
            fig.add_hline(y=res_p, line_dash="dash", line_color="#FF00FF", row=1, col=1)
            fig.add_hline(y=sup_p, line_dash="dash", line_color="#00FFFF", row=1, col=1)
            fig.add_trace(go.Scatter(x=df['dt'], y=df['ratio'], fill='tozeroy', line=dict(color='gold'), name="量能比"), row=2, col=1)
            fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)
            fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
