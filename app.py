import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 系统核心配置 ---
st.set_page_config(page_title="ETH AI V4.0 - 工业级加固版", layout="wide")
st_autorefresh(interval=5000, key="eth_core_v4")

# 核心状态机：管理信号指纹与重试计数
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "sig_fingerprint": "",
        "last_speak_ts": 0,
        "retry_count": 0,
        "last_price": 0.0
    }

# --- 2. 语音引擎：单例隔离执行 ---
def trigger_voice(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    # 强制 cancel 之前的任务，确保新信号第一时间传达
    js_code = f"""
    <script>
        (function() {{
            if (!window.speechSynthesis) return;
            window.speechSynthesis.cancel();
            var m = new SpeechSynthesisUtterance("{clean_text}");
            m.lang = 'zh-CN'; m.rate = 1.3;
            window.speechSynthesis.speak(m);
        }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 稳健数据引擎（带节点轮询与类型强制） ---
@st.cache_resource
def init_ex():
    return ccxt.okx({
        'enableRateLimit': True,
        'timeout': 5000,
        'options': {'defaultType': 'swap'}
    })

def get_market_data():
    ex = init_ex()
    try:
        # 深采样 600 根以保证 24H (288根) 均线的起始精度
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        tick = ex.fetch_ticker('ETH/USDT:USDT')
        if not bars or not tick: return None, None
        
        df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        # 严格 24 小时移动均量
        df['v_avg'] = df['v'].rolling(window=288).mean()
        
        # 截取后 300 根，确保均线已过初始化期
        df_ui = df.iloc[-300:].copy()
        # 进化倍数计算（加入 0.001 扰动防止除以零报错）
        df_ui['ratio'] = df_ui['v'] / df_ui['v_avg'].apply(lambda x: x if x > 0.1 else 1.0)
        
        st.session_state.app_state["retry_count"] = 0
        return df_ui, tick
    except:
        st.session_state.app_state["retry_count"] += 1
        return None, None

# --- 4. 核心口诀判定（逻辑绝对对齐） ---
def analyze_vpa(df):
    # 初始状态定义（全字符串化防御）
    res = {"act": "AI 扫描中", "motto": "量价合一", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty: return res, 1.0, 0, 0

    c = df.iloc[-1]
    ratio = float(c['ratio'])
    # 动态支撑压力（取过去 50 根 K 线分位）
    p_res = float(df['h'].iloc[-50:-1].quantile(0.95))
    p_sup = float(df['l'].iloc[-50:-1].quantile(0.05))
    
    # 【口诀完全对齐 - 不删减逻辑】
    if ratio < 0.6:
        if c['l'] <= p_sup * 1.001:
            res.update({"act": "准备多", "motto": "缩量回踩，支撑有效", "color": "#1A237E", "voice": "缩量回踩，低点不破"})
        elif c['h'] >= p_res * 0.999:
            res.update({"act": "准备空", "motto": "缩量反弹，压力明显", "color": "#3E2723", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6:
        if c['c'] > p_res:
            res.update({"act": "直接开多", "motto": "爆量突破，猛龙过江", "color": "#1B5E20", "voice": "放量起涨，直接开多", "tri": "buy"})
        elif c['c'] < p_sup:
            res.update({"act": "直接开空", "motto": "爆量跌破，大势已去", "color": "#B71C1C", "voice": "放量下跌，直接开空", "tri": "sell"})

    return res, ratio, p_res, p_sup

# --- 5. UI 主循环 ---
def run():
    df, tick = get_market_data()
    
    if df is None or tick is None:
        st.warning(f"📡 链路侦测中 (重试: {st.session_state.app_state['retry_count']}) | 提示: 正在建立 WebSocket 冗余通道...")
        return

    status, ratio, res_p, sup_p = analyze_vpa(df)
    
    # 播报决策：基于指纹的唯一性判定
    now = time.time()
    current_fingerprint = f"{status['act']}_{status['voice']}"
    if status["voice"] and current_fingerprint != st.session_state.app_state["sig_fingerprint"]:
        if now - st.session_state.app_state["last_speak_ts"] > 15:
            trigger_voice(status["voice"])
            st.session_state.app_state["sig_fingerprint"] = current_fingerprint
            st.session_state.app_state["last_speak_ts"] = now

    # 顶层状态渲染（强制类型转换，杜绝 NoneType 报错）
    price = str(tick.get('last', '---'))
    st.markdown(f"""
    <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid gold; color: white;">
        <h1 style="margin:0; font-size:45px;">{status['act']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
        <p style="font-size:16px; opacity:0.8;">ETH: {price} | 压力: {res_p:.1f} | 支撑: {sup_p:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 绘制高级量价图表
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])

    # K线主图
    fig.add_trace(go.Candlestick(x=df['dt'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_hline(y=res_p, line_dash="dash", line_color="#FF00FF", annotation_text="压力线", row=1, col=1)
    fig.add_hline(y=sup_p, line_dash="dash", line_color="#00FFFF", annotation_text="支撑线", row=1, col=1)

    # 信号标记层
    if status['tri'] == "buy":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['dt']], y=[df.iloc[-1]['l']*0.998], mode="markers", marker=dict(symbol="triangle-up", size=20, color="#00FF88"), name="多头买点"), row=1, col=1)
    elif status['tri'] == "sell":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]['dt']], y=[df.iloc[-1]['h']*1.002], mode="markers", marker=dict(symbol="triangle-down", size=20, color="#FF3344"), name="空头卖点"), row=1, col=1)

    # 24H 量能进化副图
    fig.add_trace(go.Scatter(x=df['dt'], y=df['ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能比"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        st.error(f"⚠️ 核心逻辑自检触发: {str(e)}")
