import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from datetime import datetime
import time

# --- 1. 系统核心配置 ---
st.set_page_config(page_title="ETH AI 终极播报", layout="wide")

# 状态锁：确保播报精准，不漏报不重报
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None, "last_time": 0}

def ai_voice_broadcast(text):
    """精准语音引擎：注入 JS 执行，带 15s 冷却防刷屏"""
    js = f"""<script>
    try {{
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang='zh-CN'; msg.rate=1.2;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{}}
    </script>"""
    st.components.v1.html(js, height=0)

# --- 2. 稳健数据流 ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_data():
    ex = init_exchange()
    try:
        # 获取 150 根 K 线确保 MA 与分位数计算稳定
        bars = ex.fetch_ohlcv("ETH/USDT:USDT", timeframe="5m", limit=150)
        ticker = ex.fetch_ticker("ETH/USDT:USDT")
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        return df.dropna(), ticker
    except Exception:
        return None, None

# --- 3. 精准口诀判定引擎 (完全对齐逻辑) ---
def ai_engine(df, ticker):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    # 判定缩量与放量的基准
    avg_vol = df['vol'].iloc[-50:-1].median() 
    vol_ratio = curr['vol'] / (avg_vol if avg_vol > 0 else 1)
    
    # 锁定关键位置：前高与前低
    res = df['high'].iloc[-40:-1].quantile(0.95)
    sup = df['low'].iloc[-40:-1].quantile(0.05)
    h24, l24 = ticker['high'], ticker['low']
    
    status = {"action": "AI 扫描中", "motto": "静如处子，动如脱兔", "color": "#121212", "voice": "", "tri": None}

    # --- 做多口诀判定 ---
    if vol_ratio < 0.6 and curr['low'] >= sup * 0.999 and curr['close'] > curr['low']:
        status.update({"action": "准备动手", "motto": "缩量回踩，低点不破", "color": "#0D47A1", "voice": "缩量回踩，低点不破，准备动手"})
    elif vol_ratio > 1.6 and curr['close'] > res:
        status.update({"action": "直接开多", "motto": "放量起涨，突破前高", "color": "#1B5E20", "voice": "放量起涨，突破前高，直接开多", "tri": "buy"})
    elif vol_ratio > 2.5 and curr['low'] >= sup and curr['close'] < prev['close'] * 0.98:
        status.update({"action": "这是机会", "motto": "放量急跌，底部不破", "color": "#2E7D32", "voice": "放量急跌，底部不破，这是机会"})
    
    # --- 做空口诀判定 ---
    elif vol_ratio < 0.6 and curr['high'] <= res * 1.001 and curr['close'] < curr['high']:
        status.update({"action": "准备动手", "motto": "缩量反弹，高点不破", "color": "#E65100", "voice": "缩量反弹，高点不破，准备动手"})
    elif vol_ratio > 1.6 and curr['close'] < sup:
        status.update({"action": "直接开空", "motto": "放量下跌，跌破前低", "color": "#B71C1C", "voice": "放量下跌，跌破前低，直接开空", "tri": "sell"})
    elif vol_ratio > 2.5 and curr['high'] <= res and curr['close'] > prev['close'] * 1.02:
        status.update({"action": "这是机会", "motto": "放量急涨，顶部不破", "color": "#D32F2F", "voice": "放量急涨，顶部不破，这是机会"})

    # --- 埋伏口诀 ---
    elif vol_ratio < 0.4:
        status.update({"action": "埋伏等变", "motto": "缩量横盘，观察托压", "color": "#424242", "voice": "缩量横盘，埋伏等待"})

    return status, res, sup, h24, l24

# --- 4. 终极界面与渲染 ---
def render():
    df, ticker = fetch_data()
    if df is None: return

    status, res, sup, h24, l24 = ai_engine(df, ticker)
    
    # 语音播报逻辑：精准去重
    now = time.time()
    if (st.session_state.signal_memory["last_key"] != status["action"] or now - st.session_state.signal_memory["last_time"] > 20):
        if status["voice"]:
            ai_voice_broadcast(status["voice"])
            st.session_state.signal_memory["last_key"] = status["action"]
            st.session_state.signal_memory["last_time"] = now

    # 顶端看板：零删减显示
    st.markdown(f"""
        <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 3px solid #FFD700;">
            <h1 style="color:white; font-size:48px; margin:0;">{status['action']}</h1>
            <h2 style="color:#FFD700; margin-top:10px;">“{status['motto']}”</h2>
            <div style="display: flex; justify-content: space-around; margin-top: 15px; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 10px;">
                <div><small style="color:#aaa;">5M压力</small><br><b style="color:#FF00FF;font-size:20px;">{res:.2f}</b></div>
                <div><small style="color:#aaa;">5M支撑</small><br><b style="color:#00FFFF;font-size:20px;">{sup:.2f}</b></div>
                <div><small style="color:#aaa;">24H高/低</small><br><b style="color:white;">{h24:.1f} / {l24:.1f}</b></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # K线进场三角标注
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#00ff88', decreasing_line_color='#ff3344'
    )])

    if status['tri'] == "buy":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['low']*0.999], mode="markers", marker=dict(symbol="triangle-up", size=20, color="#00ff88"), name="多头进场"))
    elif status['tri'] == "sell":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['high']*1.001], mode="markers", marker=dict(symbol="triangle-down", size=20, color="#ff3344"), name="空头进场"))

    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"🛡️ 逻辑对齐: 缩量提醒/放量信号 | 刷新时间: {datetime.now().strftime('%H:%M:%S')}")

render()
time.sleep(8)
st.rerun()
