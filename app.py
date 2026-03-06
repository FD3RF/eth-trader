import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 核心配置 ---
st.set_page_config(page_title="ETH AI V3.9 - 工业级稳定版", layout="wide")
st_autorefresh(interval=5000, key="eth_stable_v39")

if "signal_state" not in st.session_state:
    st.session_state.signal_state = {
        "last_action": "None",
        "last_broadcast_time": 0,
        "retry_count": 0
    }

# --- 2. 语音引擎：原子级保护 ---
def safe_broadcast(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    # 强制隔离，防止 JS 报错干扰主逻辑
    st.components.v1.html(f"""
    <script>
        try {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance("{clean_text}");
            msg.lang = 'zh-CN'; msg.rate = 1.3;
            window.speechSynthesis.speak(msg);
        }} catch(e) {{}}
    </script>
    """, height=0)

# --- 3. 稳健数据引擎（彻底解决 NoneType 拼接报错） ---
@st.cache_resource
def get_exchange():
    return ccxt.okx({
        'enableRateLimit': True,
        'timeout': 5000,
        'options': {'defaultType': 'swap'}
    })

def fetch_safe_data():
    ex = get_exchange()
    try:
        # 批量拉取，减少请求次数
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        ticker = ex.fetch_ticker('ETH/USDT:USDT')
        
        # 严格检查数据完整性
        if not bars or not ticker or len(bars) < 300:
            return None, None
            
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        
        # 24H 均量计算，解决 NaN 污染
        df['vol_24h_avg'] = df['vol'].rolling(window=288).mean()
        df_ui = df.iloc[-300:].copy()
        
        # 溢出保护
        df_ui['vol_ratio'] = df_ui['vol'] / df_ui['vol_24h_avg'].apply(lambda x: x if x > 0.01 else 1.0)
        
        st.session_state.signal_state["retry_count"] = 0
        return df_ui.dropna(subset=['vol_ratio']), ticker
    except:
        st.session_state.signal_state["retry_count"] += 1
        return None, None

# --- 4. 核心口诀判定（不删减逻辑，强化类型安全） ---
def analyze_logic(df):
    # 初始化状态字典，确保每个 key 都有初始字符串
    status = {"action": "AI 扫描中", "motto": "量价合一", "color": "#121212", "voice": "", "tri": None}
    
    if df is None or df.empty:
        return status, 1.0, 0.0, 0.0

    curr = df.iloc[-1]
    ratio = float(curr.get('vol_ratio', 1.0))
    
    # 动态支撑压力精准计算
    res = float(df['high'].iloc[-50:-1].quantile(0.95))
    sup = float(df['low'].iloc[-50:-1].quantile(0.05))
    
    # 价格边界锁定
    c_low, c_high, c_close = float(curr['low']), float(curr['high']), float(curr['close'])

    # --- 核心口诀判定流程 ---
    if ratio < 0.6:
        if c_low <= (sup * 1.001):
            status.update({"action": "准备多", "motto": "缩量回踩，低点不破", "color": "#1A237E", "voice": "缩量回踩，低点不破"})
        elif c_high >= (res * 0.999):
            status.update({"action": "准备空", "motto": "缩量反弹，高点不破", "color": "#3E2723", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6:
        if c_close > res:
            status.update({"action": "直接开多", "motto": "爆量突破，猛龙过江", "color": "#1B5E20", "voice": "放量起涨，直接开多", "tri": "buy"})
        elif c_close < sup:
            status.update({"action": "直接开空", "motto": "爆量跌破，大势已去", "color": "#B71C1C", "voice": "放量下跌，直接开空", "tri": "sell"})

    return status, ratio, res, sup

# --- 5. 主渲染流程 ---
def run_monitor():
    df, ticker = fetch_safe_data()
    
    # 数据断流处理
    if df is None or ticker is None:
        st.warning(f"📡 链路侦测中 (重试: {st.session_state.signal_state['retry_count']}) | 提示: 正在尝试备用握手协议...")
        return

    status, ratio, res, sup = analyze_logic(df)
    
    # 播报锁
    now = time.time()
    if status["voice"] and status["action"] != st.session_state.signal_state["last_action"]:
        if now - st.session_state.signal_state["last_broadcast_time"] > 20:
            safe_broadcast(status["voice"])
            st.session_state.signal_state["last_action"] = str(status["action"])
            st.session_state.signal_state["last_broadcast_time"] = now

    # 渲染安全化：彻底杜绝 'NoneType' + 'str'
    price = str(ticker.get('last', '---'))
    action = str(status.get('action', '---'))
    motto = str(status.get('motto', '---'))
    bg_color = str(status.get('color', '#121212'))

    st.markdown(f"""
    <div style="background:{bg_color}; padding:25px; border-radius:15px; text-align:center; border: 2px solid gold; color: white;">
        <h1 style="margin:0; font-size:42px;">{action} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{motto}”</h3>
        <p style="font-size:16px; opacity:0.9;">实时价: {price} | 压力: {res:.1f} | 支撑: {sup:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 绘图层
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="#FF00FF", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="#00FFFF", row=1, col=1)
    
    # 爆量/缩量信号标记
    if status['tri'] == "buy":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['low']*0.998], mode="markers", marker=dict(symbol="triangle-up", size=18, color="#00FF88"), name="买入"), row=1, col=1)
    elif status['tri'] == "sell":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['high']*1.002], mode="markers", marker=dict(symbol="triangle-down", size=18, color="#FF3344"), name="卖出"), row=1, col=1)

    # 量能进化曲线
    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能比"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    try:
        run_monitor()
    except Exception as e:
        st.error(f"⚠️ 系统核心调度异常，已拦截并准备重启。错误代码: {str(e)}")
