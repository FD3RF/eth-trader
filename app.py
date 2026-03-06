import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 环境初始化与黑屏防御 ---
st.set_page_config(page_title="ETH AI 终极盯盘", layout="wide")

# 强制刷新器：解决 Streamlit 挂起导致的黑屏
st_autorefresh(interval=5000, key="eth_monitor_refresh")

if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_key": None, "last_time": 0}

def safe_broadcast(text):
    """采用更稳健的 IFrame 注入方式，防止 JS 阻塞主线程导致黑屏"""
    components_code = f"""
    <script>
    (function() {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang='zh-CN'; msg.rate=1.2;
        window.speechSynthesis.speak(msg);
    }})();
    </script>
    """
    st.components.v1.html(components_code, height=0)

# --- 2. 工业级数据引擎 ---
@st.cache_resource
def get_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_market_data():
    ex = get_exchange()
    try:
        # 获取 200 根以确保 24H 成交量进化曲线的平滑度
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=200)
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        # 量能进化：当前量 / 24周期中位量
        df['vol_ma'] = df['vol'].rolling(window=24).median()
        df['vol_ratio'] = df['vol'] / df['vol_ma']
        ticker = ex.fetch_ticker('ETH/USDT:USDT')
        return df.dropna(), ticker
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None, None

# --- 3. 核心口诀判定逻辑 (严格不删减) ---
def analyze_signal(df):
    curr = df.iloc[-1]
    ratio = curr['vol_ratio']
    # 动态锁定 5M 关键位置
    res = df['high'].iloc[-40:-1].quantile(0.95)
    sup = df['low'].iloc[-40:-1].quantile(0.05)
    
    status = {"action": "AI 扫描中", "motto": "量价合一，顺势而为", "color": "#121212", "voice": "", "tri": None}

    # 做多口诀：缩量回踩 vs 放量起涨
    if ratio < 0.6 and curr['low'] <= sup * 1.002:
        status.update({"action": "准备多", "motto": "缩量回踩，低点不破", "color": "#0D47A1", "voice": "缩量回踩，低点不破"})
    elif ratio > 1.6 and curr['close'] > res:
        status.update({"action": "直接开多", "motto": "放量起涨，突破前高", "color": "#1B5E20", "voice": "放量起涨，突破前高，直接开多", "tri": "buy"})
    
    # 做空口诀：缩量反弹 vs 放量跌破
    elif ratio < 0.6 and curr['high'] >= res * 0.998:
        status.update({"action": "准备空", "motto": "缩量反弹，高点不破", "color": "#E65100", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6 and curr['close'] < sup:
        status.update({"action": "直接开空", "motto": "放量下跌，跌破前低", "color": "#B71C1C", "voice": "放量下跌，跌破前低，直接开空", "tri": "sell"})

    return status, ratio, res, sup

# --- 4. 视觉渲染引擎 ---
def main():
    df, ticker = fetch_market_data()
    if df is None: return

    status, ratio, res, sup = analyze_signal(df)
    
    # 语音播报调度：增加 25 秒冷却，防止循环重播
    now = time.time()
    if status["voice"] and st.session_state.signal_memory["last_key"] != status["action"] and now - st.session_state.signal_memory["last_time"] > 25:
        safe_broadcast(status["voice"])
        st.session_state.signal_memory["last_key"] = status["action"]
        st.session_state.last_time = now

    # 修复看板：采用简洁的 Markdown 渲染，防止 HTML 溢出
    st.markdown(f"""
    <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
        <h1 style="margin:0; font-size:45px;">{status['action']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
        <p style="font-size:14px; opacity:0.7;">当前价格: {ticker['last']} | 24H高低: {ticker['high']}/{ticker['low']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 成交量进化双图 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K线图层
    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="#FF00FF", annotation_text="压力", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="#00FFFF", annotation_text="支撑", row=1, col=1)

    # 信号点标注
    if status['tri'] == "buy":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['low']*0.999], mode="markers", marker=dict(symbol="triangle-up", size=20, color="#00FF88"), name="买入"), row=1, col=1)
    elif status['tri'] == "sell":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['ts_dt']], y=[df.iloc[-1]['high']*1.001], mode="markers", marker=dict(symbol="triangle-down", size=20, color="#FF3344"), name="卖出"), row=1, col=1)

    # 进化曲线图层
    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能进化"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
