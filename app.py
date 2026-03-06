import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit_autorefresh import st_autorefresh

# --- 1. 核心环境加固 ---
st.set_page_config(page_title="ETH AI V4.2 - 闭环版", layout="wide")
st_autorefresh(interval=5000, key="eth_v42_refresh")

# 全局状态字典：升级为“持久指纹锁”
if "v42_state" not in st.session_state:
    st.session_state.v42_state = {
        "fingerprint": "",
        "last_voice_ts": 0,
        "err_count": 0,
        "cache_df": None,
        "cache_tick": None,
        "last_confirmed_ts": 0 # 新增：用于锁定最后播报的K线时间戳
    }

# --- 2. 语音引擎：单例强制执行 ---
def trigger_voice(text):
    if not text: return
    clean_text = str(text).replace('"', '').replace("'", "").strip()
    js_code = f"""
    <script>
        (function() {{
            if (!window.speechSynthesis) return;
            window.speechSynthesis.cancel();
            var m = new SpeechSynthesisUtterance("{clean_text}");
            m.lang = 'zh-CN'; m.rate = 1.4;
            window.speechSynthesis.speak(m);
        }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)

# --- 3. 稳健数据引擎（带 epsilon 溢出保护） ---
@st.cache_resource
def get_ex():
    return ccxt.okx({'enableRateLimit': True, 'timeout': 3000})

def fetch_data_robust():
    ex = get_ex()
    try:
        bars = ex.fetch_ohlcv('ETH/USDT:USDT', '5m', limit=600)
        tick = ex.fetch_ticker('ETH/USDT:USDT')
        if bars and tick:
            df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
            df['dt'] = pd.to_datetime(df['ts'], unit='ms')
            # 严格计算：24H均量必须排除当前正在跳动的K线，取前288根
            df['v_avg'] = df['v'].shift(1).rolling(window=288).mean()
            df_ui = df.iloc[-300:].copy()
            # 比率计算优化：增加最小值过滤
            df_ui['ratio'] = df_ui['v'] / (df_ui['v_avg'].apply(lambda x: x if x > 0.01 else 1.0))
            
            st.session_state.v42_state["cache_df"] = df_ui
            st.session_state.v42_state["cache_tick"] = tick
            st.session_state.v42_state["err_count"] = 0
            return df_ui, tick
    except:
        st.session_state.v42_state["err_count"] += 1
        return st.session_state.v42_state["cache_df"], st.session_state.v42_state["cache_tick"]
    return None, None

# --- 4. 核心口诀判定（逻辑严丝合缝 + 收盘防误报） ---
def analyze_vpa(df):
    res = {"act": "AI 扫描中", "motto": "量价合一", "color": "#121212", "voice": "", "tri": None}
    if df is None or df.empty: return res, 1.0, 0, 0

    c = df.iloc[-1]
    ratio = float(c['ratio'])
    # 动态支撑压力：取前 50 根（不含当前根）保证基准稳定
    p_res = float(df['h'].iloc[-51:-1].quantile(0.95))
    p_sup = float(df['l'].iloc[-51:-1].quantile(0.05))
    
    # 判定时间锚点：防止在 5m 线开盘前 30 秒因为量小误报缩量
    is_new_bar = (time.time() * 1000 - c['ts']) < 30000 # 开盘前30秒

    # 【口诀逻辑】：缩量不破（避开新开盘） & 爆量突破
    if ratio < 0.6 and not is_new_bar:
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
def main():
    df, tick = fetch_data_robust()
    if df is None or tick is None:
        st.warning(f"📡 链路侦测中 (尝试: {st.session_state.v42_state['err_count']})")
        return

    status, ratio, res_p, sup_p = analyze_vpa(df)
    
    # 播报锁升级：指纹 + 时间戳 + 20s 冷却
    now = time.time()
    current_ts = int(df.iloc[-1]['ts'])
    finger = f"{status['act']}_{status['voice']}_{current_ts}"
    
    if status["voice"] and finger != st.session_state.v42_state["fingerprint"]:
        if now - st.session_state.v42_state["last_voice_ts"] > 20:
            trigger_voice(status["voice"])
            st.session_state.v42_state["fingerprint"] = finger
            st.session_state.v42_state["last_voice_ts"] = now

    # UI 界面
    price = str(tick.get('last', '---'))
    st.markdown(f"""
    <div style="background:{status['color']}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #FFD700; color: white;">
        <h1 style="margin:0; font-size:48px;">{status['act']} ({ratio:.2f}x)</h1>
        <h3 style="color:#FFD700; margin:10px 0;">“{status['motto']}”</h3>
        <p style="font-size:16px; opacity:0.8;">ETH: {price} | 压力: {res_p:.1f} | 支撑: {sup_p:.1f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 图表绘制
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['dt'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K线"), row=1, col=1)
    fig.add_hline(y=res_p, line_dash="dash", line_color="magenta", row=1, col=1)
    fig.add_hline(y=sup_p, line_dash="dash", line_color="cyan", row=1, col=1)
    fig.add_trace(go.Scatter(x=df['dt'], y=df['ratio'], fill='tozeroy', line=dict(color='gold', width=2), name="量能比"), row=2, col=1)
    fig.add_hline(y=1.6, line_dash="dot", line_color="#FF4444", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="#44FF44", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ 系统自动防崩溃重连: {str(e)}")
