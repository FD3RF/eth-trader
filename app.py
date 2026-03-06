import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统核心配置 ---
st.set_page_config(page_title="ETH AI 终极盯盘 V3.2", layout="wide")
st_autorefresh(interval=5000, key="eth_stable_stream_v32")

# 状态锁：播报精准度核心
if "signal_memory" not in st.session_state:
    st.session_state.signal_memory = {
        "last_action": None, 
        "last_time": 0,
        "history_log": []
    }

# --- 2. 语音引擎：干净利落 ---
def safe_broadcast(text):
    if not text or str(text).strip() in ["", "None"]:
        return
    clean_text = str(text).replace('"', '').replace("'", "")
    js_code = f"""
    <script>
    try {{
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("{clean_text}");
        msg.lang='zh-CN'; msg.rate=1.3;
        window.speechSynthesis.speak(msg);
    }} catch(e) {{ console.log("Speech Error"); }}
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 稳健数据引擎（600根深采样） ---
@st.cache_resource
def init_exchange():
    return ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_safe_data():
    ex = init_exchange()
    try:
        # 获取 600 根以计算完美的 288 周期 (24H) 均线
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        if not bars or len(bars) < 300: return None, None
            
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')
        
        # 计算 24H 移动均量
        df['vol_24h_avg'] = df['vol'].rolling(window=288).mean()
        
        # 截取最后 300 根用于 UI
        df_ui = df.iloc[-300:].copy()
        df_ui['vol_ratio'] = df_ui['vol'] / df_ui['vol_24h_avg'].replace(0, 1) 
        
        ticker = ex.fetch_ticker('ETH/USDT:USDT')
        return df_ui, ticker
    except Exception as e:
        st.error(f"数据链路异常: {e}")
        return None, None

# --- 4. 核心口诀判定（逻辑强化，确保精准） ---
def analyze_logic(df):
    status = {"action": "AI 扫描中", "motto": "量价合一，顺势而为", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty: return status, 1.0, 0.0, 0.0

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    ratio = curr.get('vol_ratio', 1.0)
    
    # 动态支撑压力 (采用 50 周期样本)
    res = df['high'].iloc[-50:-1].quantile(0.95)
    sup = df['low'].iloc[-50:-1].quantile(0.05)
    
    # 精度锁处理
    c_low, c_high, c_close = round(curr['low'], 4), round(curr['high'], 4), round(curr['close'], 4)
    r_sup, r_res = round(sup, 4), round(res, 4)

    # 【口诀完全对齐 - 强化判定】
    if ratio < 0.6:
        # 允许 0.1% 的触碰误差
        if c_low <= round(r_sup * 1.001, 4):
            status.update({"action": "准备多", "motto": "缩量回踩，支撑有效", "color": "#0D47A1", "voice": "缩量回踩，低点不破"})
        elif c_high >= round(r_res * 0.999, 4):
            status.update({"action": "准备空", "motto": "缩量反弹，压力明显", "color": "#E65100", "voice": "缩量反弹，高点不破"})
            
    elif ratio > 1.6:
        if c_close > r_res:
            status.update({"action": "直接开多", "motto": "爆量突破，猛龙过江", "color": "#1B5E20", "voice": "放量起涨，突破前高", "tri": "buy"})
        elif c_close < r_sup:
            status.update({"action": "直接开空", "motto": "爆量跌破，大势已去", "color": "#B71C1C", "voice": "放量下跌，跌破前低", "tri": "sell"})

    return status, float(ratio), float(res), float(sup)

# --- 5. 渲染引擎 ---
def run_monitor():
    try:
        df, ticker = fetch_safe_data()
        if df is None: return

        status, ratio, res, sup = analyze_logic(df)
        
        # 【播报精准控制：动作变化 + 时间冷却】
        now = time.time()
        if status["voice"]:
            is_new_action = status["action"] != st.session_state.signal_memory["last_action"]
            time_passed = now - st.session_state.signal_memory["last_time"]
            
            if is_new_action and time_passed > 15: # 动作切换且过15秒
                safe_broadcast(status["voice"])
                st.session_state.signal_memory["last_action"] = status["action"]
                st.session_state.signal_memory["last_time"] = now
            elif time_passed > 180: # 即使动作没变，3分钟后再次提醒
                safe_broadcast(status["voice"])
                st.session_state.signal_memory["last_time"] = now

        # 看板渲染
        price = ticker.get('last', '---')
        st.markdown(f"""
        <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
            <h1 style="margin:0; font-size:45px;">{status['action']} ({ratio:.2f}x)</h1>
            <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
            <p style="font-size:14px; opacity:0.8;">实时价: {price} | 压力: {res:.1f} | 支撑: {sup:.1f}</p>
        </div>
        """, unsafe_allow_html=True)

        # 绘图层
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # 主图
        fig.add_trace(go.Candlestick(x=df['ts_dt'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="ETH"), row=1, col=1)
        fig.add_hline(y=res, line_dash="dash", line_color="magenta", row=1, col=1)
        fig.add_hline(y=sup, line_dash="dash", line_color="cyan", row=1, col=1)

        # 副图：24H量能进化曲线
        
        fig.add_trace(go.Scatter(x=df['ts_dt'], y=df['vol_ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能倍数"), row=2, col=1)
        fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.info(f"等待数据流更新: {e}")

if __name__ == "__main__":
    run_monitor()
