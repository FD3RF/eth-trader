import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统基础配置 ---
st.set_page_config(page_title="ETH AI 终极盯盘 V2.0", layout="wide")
st_autorefresh(interval=5000, key="refresh")

if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None}
if "last_voice_time" not in st.session_state:
    st.session_state.last_voice_time = 0

# --- 2. 数据引擎 (增加 24H 采样深度) ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_full_data():
    ex = init_exchange()
    try:
        # 获取 300 根 K 线以计算更稳健的 24H 均值参考
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=300)
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        # 计算成交量进化指标：当前量 / 过去 24 周期中位量
        df['vol_ma'] = df['vol'].rolling(window=24).median()
        df['vol_ratio_curve'] = df['vol'] / df['vol_ma']
        return df.dropna(), ex.fetch_ticker('ETH/USDT:USDT')
    except:
        return None, None

# --- 3. 增强型 AI 判定引擎 ---
def ai_engine_v2(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    ratio = curr['vol_ratio_curve']
    
    # 动态支撑压力
    res = df['high'].iloc[-50:-1].quantile(0.95)
    sup = df['low'].iloc[-50:-1].quantile(0.05)
    
    status = {"action":"AI 扫描中","motto":"量价合一，顺势而为","color":"#121212","voice":None, "tri":None}

    # 逻辑对齐：缩量 (Ratio < 0.6) | 放量 (Ratio > 1.6)
    if ratio < 0.6:
        if curr['low'] <= sup * 1.001:
            status.update({"action":"准备多","motto":"缩量回踩，支撑有效","color":"#0D47A1","voice":"缩量回踩，低点不破"})
        elif curr['high'] >= res * 0.999:
            status.update({"action":"准备空","motto":"缩量反弹，压力明显","color":"#E65100","voice":"缩量反弹，高点不破"})
            
    elif ratio > 1.6:
        if curr['close'] > res:
            status.update({"action":"直接开多","motto":"爆量突破，猛龙过江","color":"#1B5E20","voice":"放量起涨，突破前高", "tri":"buy"})
        elif curr['close'] < sup:
            status.update({"action":"直接开空","motto":"爆量跌破，大势已去","color":"#B71C1C","voice":"放量下跌，跌破前低", "tri":"sell"})

    return status, ratio, res, sup

# --- 4. UI 渲染与“成交量进化曲线”展示 ---
def render_dashboard():
    df, ticker = fetch_full_data()
    if df is None: return

    status, vol_ratio, res, sup = ai_engine_v2(df)
    
    # 看板渲染
    st.markdown(f"""
    <div style="background:{status['color']};padding:20px;border-radius:12px;text-align:center;border:2px solid #FFD700;color:white;">
        <h1 style="margin:0;">{status['action']} ({vol_ratio:.2f}x)</h1>
        <h3 style="color:#FFD700;">{status['motto']}</h3>
    </div>
    """, unsafe_allow_html=True)

    # 创建双子图：上方 K 线，下方成交量进化曲线
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # 主图：K 线与标记
    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH价格"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="magenta", annotation_text="压力线", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="cyan", annotation_text="支撑线", row=1, col=1)

    # 副图：成交量进化曲线 
    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio_curve'], fill='tozeroy', name="量能进化倍数", line=dict(color='gold', width=2)), row=2, col=1)
    # 预警线：0.6 缩量线 与 1.6 放量线
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 语音播报调度
    now = time.time()
    if status["voice"] and st.session_state.signal_memory["last_key"] != status["action"] and now - st.session_state.last_voice_time > 20:
        js = f'<script>window.speechSynthesis.cancel(); var m=new SpeechSynthesisUtterance("{status["voice"]}"); m.lang="zh-CN"; window.speechSynthesis.speak(m);</script>'
        st.components.v1.html(js, height=0)
        st.session_state.signal_memory["last_key"] = status["action"]
        st.session_state.last_voice_time = now

render_dashboard()
