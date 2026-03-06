import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统核心配置与自动刷新 ---
st.set_page_config(page_title="ETH AI 终极盯盘", layout="wide")
# 5秒自动刷新，确保信号即时性
st_autorefresh(interval=5000, key="refresh")

# 状态锁：优化播报逻辑
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None, "last_time": 0}

def ai_voice_broadcast(text):
    """精准语音引擎：防止短时间内重复轰炸"""
    js = f"""
    <script>
    try {{
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang='zh-CN'; msg.rate=1.2;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{}}
    </script>
    """
    st.components.v1.html(js, height=0)

# --- 2. 数据获取与性能计算 ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_data():
    ex = init_exchange()
    try:
        # 增加 limit 以支持 MA50 及更高周期的计算
        bars = ex.fetch_ohlcv("ETH/USDT:USDT", timeframe="5m", limit=150)
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        return df.dropna()
    except Exception as e:
        return None

# --- 3. 核心口诀与行为检测引擎 ---
def get_indicators(df):
    # 趋势判定：MA20 与 MA50 交叉
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    curr_trend = "up" if ma20.iloc[-1] > ma50.iloc[-1] else "down"
    
    # 动态压力支撑：使用 95% 分位数过滤极端插针，更精准
    res = df['high'].iloc[-40:-1].quantile(0.95)
    sup = df['low'].iloc[-40:-1].quantile(0.05)
    
    # 趋势强度 (计算斜率与乖离率)
    slope = ma20.diff().iloc[-5:].mean()
    volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
    strength = round((abs(slope) / (volatility if volatility > 0 else 0.001)), 2)
    
    return curr_trend, res, sup, strength

def ai_engine(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    avg_vol = df['vol'].iloc[-50:-1].median()
    vol_ratio = curr['vol'] / (avg_vol if avg_vol > 0 else 1)
    
    trend_dir, res, sup, strength = get_indicators(df)
    
    status = {"action": "AI 扫描中", "motto": "等待量价共振", "color": "#121212", "voice": ""}

    # A. 极端异动判定 (高优先级)
    # 庄家拉升：量能翻倍且价格连续上涨
    is_whale = df['vol'].iloc[-3:].mean() > avg_vol * 2.5 and curr['close'] > df['close'].iloc[-3]
    # 暴力砸盘：巨量且跌幅超过 1%
    is_dump = curr['vol'] > avg_vol * 3 and curr['close'] < prev['close'] * 0.99

    # B. 核心交易口诀
    # 1. 突破判定 (放量)
    if vol_ratio > 1.6 and curr['close'] > res:
        status.update({"action": "直接开多", "motto": "放量突破压力位", "color": "#1B5E20", "voice": "放量突破，直接开多"})
    elif vol_ratio > 1.6 and curr['close'] < sup:
        status.update({"action": "直接开空", "motto": "放量跌破支撑位", "color": "#B71C1C", "voice": "放量跌破，直接开空"})
    
    # 2. 假突破判定 (无量)
    elif curr['close'] > res and vol_ratio < 0.8:
        status.update({"action": "假突破", "motto": "警惕诱多回落", "color": "#4A148C", "voice": "检测到假突破"})
    
    # 3. 缩量机会
    elif vol_ratio < 0.5 and curr['close'] <= sup * 1.001:
        status.update({"action": "准备多", "motto": "缩量测试支撑", "color": "#0D47A1", "voice": "缩量回踩完毕"})
    
    # 特殊提醒叠加
    if is_whale: status.update({"action": "庄家抢筹", "color": "#2E7D32", "voice": "检测到主力吸筹"})
    if is_dump: status.update({"action": "砸盘逃命", "color": "#D32F2F", "voice": "巨量砸盘，快跑"})

    return status, trend_dir, strength, res, sup

# --- 4. 渲染引擎 ---
def render():
    df = fetch_data()
    if df is None:
        st.error("数据链路中断，正在重新拨号...")
        return

    status, trend_dir, strength, res, sup = ai_engine(df)
    
    # 语音播报：加入 15 秒冷却防止刷屏
    now = time.time()
    if (st.session_state.signal_memory["last_key"] != status["action"] or 
        now - st.session_state.signal_memory["last_time"] > 15):
        if status["voice"]:
            ai_voice_broadcast(status["voice"])
            st.session_state.signal_memory["last_key"] = status["action"]
            st.session_state.signal_memory["last_time"] = now

    # 顶端大屏
    st.markdown(f"""
        <div style="background:{status['color']}; padding:30px; border-radius:15px; text-align:center; border: 3px solid #FFD700;">
            <h1 style="color:white; font-size:60px; margin:0;">{status['action']}</h1>
            <h3 style="color:#FFD700; margin-top:10px;">趋势: {trend_dir.upper()} | 强度: {strength}</h3>
        </div>
    """, unsafe_allow_html=True)

    # K线可视化
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close']
    )])
    fig.add_hline(y=res, line_dash="dash", line_color="#FF00FF", annotation_text="95% 压力")
    fig.add_hline(y=sup, line_dash="dash", line_color="#00FFFF", annotation_text="5% 支撑")
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"🛡️ 终极盯盘系统运行中 | 刷新: {datetime.now().strftime('%H:%M:%S')}")

render()
