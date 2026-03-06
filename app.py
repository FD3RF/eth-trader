import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统核心配置 ---
st.set_page_config(page_title="ETH AI 终极盯盘 V3.3", layout="wide")
st_autorefresh(interval=5000, key="eth_stable_v33")

if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_action": None, "last_time": 0}

# --- 2. 语音引擎：强制字符化保护 ---
def safe_broadcast(text):
    if text is None: return
    # 强制转换为字符串并过滤非法字符，防止 JS 拼接报错
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    if not clean_text: return
    
    js_code = f"""
    <script>
    try {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("{clean_text}");
        msg.lang='zh-CN'; msg.rate=1.3;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{ console.log("Voice Error"); }}
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 稳健数据引擎（全方位 None 保护） ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_safe_data():
    ex = init_exchange()
    try:
        # 获取 600 根以确保 24H 均线计算
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        ticker = ex.fetch_ticker('ETH/USDT:USDT')
        
        if not bars or not ticker:
            return None, None
            
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        df['vol_24h_avg'] = df['vol'].rolling(window=288).mean()
        
        df_ui = df.iloc[-300:].copy()
        df_ui['vol_ratio'] = df_ui['vol'] / df_ui['vol_24h_avg'].replace(0, 1) 
        
        return df_ui.dropna(subset=['vol_ratio']), ticker
    except:
        return None, None

# --- 4. 核心口诀判定（逻辑严丝合缝） ---
def analyze_logic(df):
    status = {"action": "AI 扫描中", "motto": "量价合一，顺势而为", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty:
        return status, 1.0, 0.0, 0.0

    curr = df.iloc[-1]
    ratio = float(curr.get('vol_ratio', 1.0))
    
    # 动态支撑压力 (采用 50 周期样本)
    res = float(df['high'].iloc[-50:-1].quantile(0.95))
    sup = float(df['low'].iloc[-50:-1].quantile(0.05))
    
    # 判定逻辑 (不删减任何功能)
    if ratio < 0.6:
        if curr['low'] <= sup * 1.001:
            status.update({"action": "准备多", "motto": "缩量回踩，支撑有效", "color": "#0D47A1", "voice": "缩量回踩，低点不破"})
        elif curr['high'] >= res * 0.999:
            status.update({"action": "准备空", "motto": "缩量反弹，压力明显", "color": "#E65100", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6:
        if curr['close'] > res:
            status.update({"action": "直接开多", "motto": "爆量突破，猛龙过江", "color": "#1B5E20", "voice": "放量起涨，突破前高", "tri": "buy"})
        elif curr['close'] < sup:
            status.update({"action": "直接开空", "motto": "爆量跌破，大势已去", "color": "#B71C1C", "voice": "放量下跌，跌破前低", "tri": "sell"})

    return status, ratio, res, sup

# --- 5. 渲染引擎：解决 'NoneType' + 'str' 报错 ---
def run_monitor():
    df, ticker = fetch_safe_data()
    
    # 即使数据获取失败，也显示基础 UI 框架，防止黑屏崩溃
    if df is None or ticker is None:
        st.warning("🔄 正在同步 OKX 数据链路，请检查网络连接...")
        return

    status, ratio, res, sup = analyze_logic(df)
    
    # 语音调度逻辑
    now = time.time()
    voice_content = status.get("voice")
    if voice_content and st.session_state.signal_memory["last_action"] != status["action"]:
        if now - st.session_state.signal_memory["last_time"] > 20:
            safe_broadcast(voice_content)
            st.session_state.signal_memory["last_action"] = status["action"]
            st.session_state.signal_memory["last_time"] = now

    # 看板渲染：使用 f-string 前先确保所有变量均为字符串
    display_price = str(ticker.get('last', '---'))
    display_action = str(status.get('action', '扫描中'))
    display_motto = str(status.get('motto', '---'))
    display_color = str(status.get('color', '#121212'))

    st.markdown(f"""
    <div style="background:{display_color}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
        <h1 style="margin:0; font-size:45px;">{display_action} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{display_motto}”</h3>
        <p style="font-size:14px; opacity:0.8;">实时价: {display_price} | 压力: {res:.1f} | 支撑: {sup:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 绘图层 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K线主图
    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="magenta", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="cyan", row=1, col=1)

    # 量能进化曲线
    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="24H量能比"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    try:
        run_monitor()
    except Exception as e:
        st.error(f"系统运行异常: {str(e)}")
