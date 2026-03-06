import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统核心配置 ---
st.set_page_config(page_title="ETH AI 终极盯盘 V3.5", layout="wide")
st_autorefresh(interval=5000, key="eth_core_refresh")

if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_action": None, "last_time": 0}

# --- 2. 语音引擎：采用强制字符保护与隔离 ---
def safe_broadcast(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    js_code = f"""
    <script>
    (function() {{
        try {{
            if (window.speechSynthesis) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{clean_text}");
                msg.lang='zh-CN'; msg.rate=1.3;
                window.speechSynthesis.speak(msg);
            }}
        }} catch(e) {{ console.error("Broadcast Error"); }}
    }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 稳健数据引擎（强化超时与采样精度） ---
@st.cache_resource
def get_exchange():
    return ccxt.okx({
        'enableRateLimit': True, 
        'timeout': 5000, # 提升至5秒容错
        'options': {'defaultType': 'swap'}
    })

def fetch_safe_data():
    ex = get_exchange()
    try:
        # 深采样 600 根，确保 24H (288根) 均线在可见区域完全真实
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        ticker = ex.fetch_ticker('ETH/USDT:USDT')
        
        if not bars or not ticker or len(bars) < 300:
            return None, None
            
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        
        # 严格 24 小时均量计算，解决初始 NaN 问题
        df['vol_24h_avg'] = df['vol'].rolling(window=288).mean()
        df_ui = df.iloc[-300:].copy()
        
        # 【加固点】：防止均量过小导致的比例爆炸
        df_ui['vol_ratio'] = df_ui['vol'] / df_ui['vol_24h_avg'].apply(lambda x: x if x > 0.01 else 1.0)
        
        return df_ui.dropna(subset=['vol_ratio']), ticker
    except Exception:
        return None, None

# --- 4. 核心口诀判定（逻辑无缝对齐） ---
def analyze_logic(df):
    # 默认状态
    status = {"action": "AI 扫描中", "motto": "量价合一，顺势而为", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty:
        return status, 1.0, 0, 0

    curr = df.iloc[-1]
    ratio = float(curr['vol_ratio'])
    
    # 锁定动态支撑压力 (50 周期分位精准定位)
    res = float(df['high'].iloc[-50:-1].quantile(0.95))
    sup = float(df['low'].iloc[-50:-1].quantile(0.05))
    
    # 【不删减口诀】：缩量不破、爆量突破
    if ratio < 0.6:
        if curr['low'] <= sup * 1.001:
            status.update({"action": "准备多", "motto": "缩量回踩，低点不破", "color": "#1A237E", "voice": "缩量回踩，低点不破"})
        elif curr['high'] >= res * 0.999:
            status.update({"action": "准备空", "motto": "缩量反弹，高点不破", "color": "#4E342E", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6:
        if curr['close'] > res:
            status.update({"action": "直接开多", "motto": "爆量起涨，突破前高", "color": "#1B5E20", "voice": "放量起涨，直接开多", "tri": "buy"})
        elif curr['close'] < sup:
            status.update({"action": "直接开空", "motto": "爆量下跌，跌破前低", "color": "#B71C1C", "voice": "放量下跌，直接开空", "tri": "sell"})

    return status, ratio, res, sup

# --- 5. 渲染引擎 ---
def main():
    df, ticker = fetch_safe_data()
    
    # 数据异常时的占位提醒
    if df is None or ticker is None:
        st.info("🔄 正在从 OKX 获取最新 ETH 5M 量价序列... 请保持网络连通。")
        return

    status, ratio, res, sup = analyze_logic(df)
    
    # 播报逻辑
    now = time.time()
    if status["voice"] and st.session_state.signal_memory["last_action"] != status["action"]:
        if now - st.session_state.signal_memory["last_time"] > 20:
            safe_broadcast(status["voice"])
            st.session_state.signal_memory["last_action"] = status["action"]
            st.session_state.signal_memory["last_time"] = now

    # 顶部状态看板
    price = ticker.get('last', '---')
    st.markdown(f"""
    <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
        <h1 style="margin:0; font-size:48px;">{status['action']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
        <p style="font-size:16px; opacity:0.8;">当前价: {price} | 压力线: {res:.1f} | 支撑线: {sup:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 绘图层
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # 主图：K线与信号
    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="#FF00FF", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="#00FFFF", row=1, col=1)

    # 副图：量能进化倍数
    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能比"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
