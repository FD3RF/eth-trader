import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统核心配置 ---
st.set_page_config(page_title="ETH AI V3.8 - 工业级加固版", layout="wide")
st_autorefresh(interval=5000, key="eth_industrial_v38")

# 核心状态机：管理播报唯一性与数据连续性
if "signal_state" not in st.session_state:
    st.session_state.signal_state = {
        "last_action": None,
        "last_broadcast_time": 0,
        "connection_errors": 0
    }

# --- 2. 语音引擎：单例播放锁 ---
def safe_broadcast(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    # 使用 IFrame 隔离，确保 JS 执行不阻塞 Streamlit 主进程
    js_code = f"""
    <script>
        (function() {{
            if (!window.speechSynthesis) return;
            window.speechSynthesis.cancel(); // 强行终止排队语音
            var msg = new SpeechSynthesisUtterance("{clean_text}");
            msg.lang = 'zh-CN';
            msg.rate = 1.3;
            window.speechSynthesis.speak(msg);
        }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 稳健数据引擎（带溢出保护与超时自愈） ---
@st.cache_resource
def get_exchange():
    return ccxt.okx({
        'enableRateLimit': True,
        'timeout': 5000,
        'options': {'defaultType': 'swap'},
        # 如需代理，请取消下两行注释并修改端口
        # 'proxies': {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    })

def fetch_safe_data():
    ex = get_exchange()
    try:
        # 深采样 600 根，确保 24H 均线计算窗口完整
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        ticker = ex.fetch_ticker('ETH/USDT:USDT')
        if not bars or not ticker: return None, None, "数据返回为空"
        
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        
        # 【性能计算稳定修复】：滑动窗口计算，加入 epsilon 防止除以零
        df['vol_24h_avg'] = df['vol'].rolling(window=288).mean()
        df_ui = df.iloc[-300:].copy()
        
        # 溢出保护：若均量极小则强制设为 1，防止 ratio 变成无穷大
        df_ui['vol_ratio'] = df_ui['vol'] / df_ui['vol_24h_avg'].apply(lambda x: x if x > 0.1 else 1.0)
        
        st.session_state.signal_state["connection_errors"] = 0 # 重置错误计数
        return df_ui, ticker, "OK"
    except Exception as e:
        st.session_state.signal_state["connection_errors"] += 1
        return None, None, str(e)

# --- 4. 核心口诀判定（100% 逻辑对齐） ---
def analyze_logic(df):
    status = {"action": "AI 扫描中", "motto": "量价合一，顺势而为", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty: return status, 1.0, 0, 0

    curr = df.iloc[-1]
    ratio = float(curr['vol_ratio'])
    
    # 锁定动态支撑压力 (采用 50 周期样本分位)
    res = float(df['high'].iloc[-50:-1].quantile(0.95))
    sup = float(df['low'].iloc[-50:-1].quantile(0.05))
    
    # 【不删减口诀】：严丝合缝匹配
    if ratio < 0.6:
        # 缩量逻辑
        if curr['low'] <= sup * 1.0005: # 允许 0.05% 的极其微小误差补偿
            status.update({"action": "准备多", "motto": "缩量回踩，低点不破", "color": "#1A237E", "voice": "缩量回踩，低点不破"})
        elif curr['high'] >= res * 0.9995:
            status.update({"action": "准备空", "motto": "缩量反弹，高点不破", "color": "#3E2723", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6:
        # 爆量逻辑
        if curr['close'] > res:
            status.update({"action": "直接开多", "motto": "爆量突破，猛龙过江", "color": "#1B5E20", "voice": "放量起涨，直接开多", "tri": "buy"})
        elif curr['close'] < sup:
            status.update({"action": "直接开空", "motto": "爆量跌破，大势已去", "color": "#B71C1C", "voice": "放量下跌，直接开空", "tri": "sell"})

    return status, ratio, res, sup

# --- 5. UI 与 渲染 ---
def run_monitor():
    df, ticker, msg = fetch_safe_data()
    
    if df is None:
        st.warning(f"📡 链路侦测中 (重试次数: {st.session_state.signal_state['connection_errors']})")
        st.info(f"当前状态: {msg}")
        return

    status, ratio, res, sup = analyze_logic(df)
    
    # 语音播报逻辑：状态切换锁 + 20秒强制冷却
    now = time.time()
    if status["voice"] and status["action"] != st.session_state.signal_state["last_action"]:
        if now - st.session_state.signal_state["last_broadcast_time"] > 20:
            safe_broadcast(status["voice"])
            st.session_state.signal_state["last_action"] = status["action"]
            st.session_state.signal_state["last_broadcast_time"] = now

    # 顶部状态面板
    last_price = ticker.get('last', '---')
    st.markdown(f"""
    <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid gold; color: white;">
        <h1 style="margin:0; font-size:42px;">{status['action']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
        <p style="font-size:16px; opacity:0.9;">实时价: {last_price} | 压力线: {res:.1f} | 支撑线: {sup:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 绘图层：K线图与量能比
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])

    # 主图渲染
    fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
    fig.add_hline(y=res, line_dash="dash", line_color="#FF00FF", row=1, col=1)
    fig.add_hline(y=sup, line_dash="dash", line_color="#00FFFF", row=1, col=1)

    # 副图：24H量能比 (工业级稳定曲线)
    fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能进化比"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_monitor()
