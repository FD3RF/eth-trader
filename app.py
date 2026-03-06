import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 核心环境加固 ---
st.set_page_config(page_title="ETH AI V4.1 - 极速加固版", layout="wide")
st_autorefresh(interval=5000, key="eth_core_v41")

# 全局状态字典：解决重复播报与数据断档
if "v41_state" not in st.session_state:
    st.session_state.v41_state = {
        "fingerprint": "",
        "last_voice_ts": 0,
        "err_count": 0,
        "cache_df": None,
        "cache_tick": None
    }

# --- 2. 语音引擎：增加排队抑制锁 ---
def trigger_voice(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    # 强制 cancel 并增加防抖，防止短时间内多次触发
    js_code = f"""
    <script>
        if (window.speechSynthesis) {{
            window.speechSynthesis.cancel();
            var m = new SpeechSynthesisUtterance("{clean_text}");
            m.lang = 'zh-CN'; m.rate = 1.4; m.pitch = 1.1;
            window.speechSynthesis.speak(m);
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 并行数据引擎（带数据缓存，防止黑屏） ---
@st.cache_resource
def get_ex():
    return ccxt.okx({'enableRateLimit': True, 'timeout': 3000})

def fetch_data_robust():
    ex = get_ex()
    try:
        # 尝试并行获取数据（此处通过顺序获取但带短超时实现）
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        tick = ex.fetch_ticker('ETH/USDT:USDT')
        
        if bars and tick:
            df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
            df['dt'] = pd.to_datetime(df['ts'], unit='ms')
            df['v_avg'] = df['v'].rolling(window=288).mean()
            df_ui = df.iloc[-300:].copy()
            # 强化量能比计算逻辑：epsilon 保护
            df_ui['ratio'] = df_ui['v'] / (df_ui['v_avg'] + 1e-9).apply(lambda x: x if x > 0.05 else 1.0)
            
            # 更新缓存
            st.session_state.v41_state["cache_df"] = df_ui
            st.session_state.v41_state["cache_tick"] = tick
            st.session_state.v41_state["err_count"] = 0
            return df_ui, tick
    except Exception:
        st.session_state.v41_state["err_count"] += 1
        # 返回缓存数据，实现“无缝断线重连”
        return st.session_state.v41_state["cache_df"], st.session_state.v41_state["cache_tick"]
    return None, None

# --- 4. 核心口诀判定（逻辑严丝合缝） ---
def analyze_vpa(df):
    res = {"act": "AI 扫描中", "motto": "量价合一", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty: return res, 1.0, 0, 0

    c = df.iloc[-1]
    ratio = float(c['ratio'])
    # 动态支撑压力：过去 50 周期高低点分位
    p_res = float(df['h'].iloc[-50:-1].quantile(0.95))
    p_sup = float(df['l'].iloc[-50:-1].quantile(0.05))
    
    # 【不删减口诀】：缩量不破 & 爆量突破
    if ratio < 0.6:
        if c['l'] <= p_sup * 1.0005:
            res.update({"act": "准备多", "motto": "缩量回踩，支撑有效", "color": "#1A237E", "voice": "缩量回踩，低点不破"})
        elif c['h'] >= p_res * 0.9995:
            res.update({"act": "准备空", "motto": "缩量反弹，压力明显", "color": "#3E2723", "voice": "缩量反弹，高点不破"})
    elif ratio > 1.6:
        if c['c'] > p_res:
            res.update({"act": "直接开多", "motto": "爆量突破，猛龙过江", "color": "#1B5E20", "voice": "放量起涨，直接开多", "tri": "buy"})
        elif c['c'] < p_sup:
            res.update({"act": "直接开空", "motto": "爆量跌破，大势已去", "color": "#B71C1C", "voice": "放量下跌，直接开空", "tri": "sell"})

    return res, ratio, p_res, p_sup

# --- 5. 渲染循环 ---
def run_v41():
    df, tick = fetch_data_robust()
    
    # 彻底解决 NoneType 报错：如果缓存和实时都没有，才显示侦测中
    if df is None or tick is None:
        st.warning(f"📡 链路侦测中 (尝试次数: {st.session_state.v41_state['err_count']})")
        return

    status, ratio, res_p, sup_p = analyze_vpa(df)
    
    # 播报锁：只有指纹改变且超过 15s 冷却才发声
    now = time.time()
    finger = f"{status['act']}_{status['voice']}"
    if status["voice"] and finger != st.session_state.v41_state["fingerprint"]:
        if now - st.session_state.v41_state["last_voice_ts"] > 15:
            trigger_voice(status["voice"])
            st.session_state.v41_state["fingerprint"] = finger
            st.session_state.v41_state["last_voice_ts"] = now

    # UI 渲染 (强制类型保护)
    price = str(tick.get('last', '---'))
    st.markdown(f"""
    <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
        <h1 style="margin:0; font-size:48px;">{status['act']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
        <p style="font-size:16px; opacity:0.8;">实时价: {price} | 压力线: {res_p:.1f} | 支撑线: {sup_p:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 图表绘制
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # K线主图
    fig.add_trace(go.Candlestick(x=df['dt'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="ETH-5M"), row=1, col=1)
    fig.add_hline(y=res_p, line_dash="dash", line_color="magenta", row=1, col=1)
    fig.add_hline(y=sup_p, line_dash="dash", line_color="cyan", row=1, col=1)
    
    # 24H 量能进化副图
    fig.add_trace(go.Scatter(x=df['dt'], y=df['ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能比"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="#FF4444", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="#44FF44", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    try:
        run_v41()
    except Exception as e:
        st.error(f"⚠️ 系统自动避险重连中... (代码: {str(e)})")
