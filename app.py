import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统配置 ---
st.set_page_config(page_title="ETH AI V3.6 - 网络加固版", layout="wide")
st_autorefresh(interval=5000, key="eth_network_fix")

if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {"last_action": None, "last_time": 0}

# --- 2. 语音引擎 ---
def safe_broadcast(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    st.components.v1.html(f"""
    <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("{clean_text}");
        msg.lang='zh-CN'; msg.rate=1.4;
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# --- 3. 强化版数据引擎（支持多节点自愈） ---
@st.cache_resource
def get_exchange(node="default"):
    # 节点列表：默认节点 vs 亚马逊海外加速节点
    urls = {
        "default": "https://www.okx.com",
        "aws": "https://aws.okx.com"
    }
    return ccxt.okx({
        'enableRateLimit': True,
        'timeout': 2000, # 压缩至2秒，极速重试
        'urls': {'api': {'public': urls[node]}},
        'options': {'defaultType': 'swap'}
    })

def fetch_data_with_retry():
    # 尝试默认节点，失败则切换 AWS 节点
    for node in ["default", "aws"]:
        try:
            ex = get_exchange(node)
            bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
            ticker = ex.fetch_ticker('ETH/USDT:USDT')
            if bars and ticker:
                return bars, ticker, node
        except:
            continue
    return None, None, "error"

# --- 4. 核心逻辑（保持口诀不变，不删减） ---
def run_monitor():
    bars, ticker, active_node = fetch_data_with_retry()
    
    if active_node == "error":
        st.error("🚨 全球所有 API 节点响应超时！请检查代理或切换至更稳定的网络（推荐 5G 或 海外节点）。")
        return

    # 数据处理
    df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
    df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
    df['vol_24h_avg'] = df['vol'].rolling(window=288).mean()
    df_ui = df.iloc[-300:].copy()
    df_ui['vol_ratio'] = df_ui['vol'] / df_ui['vol_24h_avg'].replace(0, 1)
    
    curr = df_ui.iloc[-1]
    ratio, res, sup = float(curr['vol_ratio']), float(df_ui['high'].iloc[-50:-1].quantile(0.95)), float(df_ui['low'].iloc[-50:-1].quantile(0.05))
    
    # 信号逻辑
    status = {"action": "AI 扫描中", "motto": "量价合一", "color": "#121212", "voice": "", "tri": None}
    if ratio < 0.6:
        if curr['low'] <= sup * 1.001: status.update({"action":"准备多","motto":"缩量回踩，低点不破","color":"#1A237E","voice":"缩量回踩"})
        elif curr['high'] >= res * 0.999: status.update({"action":"准备空","motto":"缩量反弹，高点不破","color":"#4E342E","voice":"缩量反弹"})
    elif ratio > 1.6:
        if curr['close'] > res: status.update({"action":"直接开多","motto":"爆量突破","color":"#1B5E20","voice":"放量起涨","tri":"buy"})
        elif curr['close'] < sup: status.update({"action":"直接开空","motto":"爆量跌破","color":"#B71C1C","voice":"放量下跌","tri":"sell"})

    # 语音播报
    now = time.time()
    if status["voice"] and st.session_state.signal_memory["last_action"] != status["action"]:
        if now - st.session_state.signal_memory["last_time"] > 20:
            safe_broadcast(status["voice"])
            st.session_state.signal_memory["last_action"] = status["action"]
            st.session_state.signal_memory["last_time"] = now

    # UI 渲染
    st.markdown(f"""
    <div style="background:{status['color']}; padding:20px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
        <h1 style="margin:0;">{status['action']} ({ratio:.2f}x)</h1>
        <p style="color:#FFD700;">“{status['motto']}” | 节点: {active_node.upper()} | 价格: {ticker['last']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 图表绘制 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_ui['ts_dt'], open=df_ui['open'], high=df_ui['high'], low=df_ui['low'], close=df_ui['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="magenta", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="cyan", row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ui['ts_dt'], y=df_ui['vol_ratio'], fill='tozeroy', line=dict(color='gold'), name="量能比"), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=5,r=5,t=5,b=5))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_monitor()
