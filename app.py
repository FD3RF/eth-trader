import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

# 初始化语音状态
if "last_alert_msg" not in st.session_state:
    st.session_state.last_alert_msg = ""
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; max-width: 100%; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #1f2937; padding: 10px; border-radius: 10px; min-height: 80px; }
    .status-card { transition: all 0.3s ease; background: #111827; border-radius: 12px; margin-bottom: 10px; border: 1px solid #1f2937; }
    .voice-btn { padding: 10px; background: #ef4444; color: white; border-radius: 8px; text-align: center; cursor: pointer; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 实时语音引擎
# ==========================================
def trigger_voice(text, placeholder):
    """通过注入 JS 实现真正的系统级语音播报"""
    if st.session_state.last_alert_msg != text:
        with placeholder:
            components.html(f"""
                <script>
                if ('speechSynthesis' in window) {{
                    var msg = new SpeechSynthesisUtterance('{text}');
                    msg.lang = 'zh-CN';
                    msg.rate = 1.2; 
                    window.speechSynthesis.speak(msg);
                }}
                </script>
            """, height=0)
        st.session_state.last_alert_msg = text

# ==========================================
# 3. 策略逻辑 (不删减)
# ==========================================
def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_size'] = abs(df['c'] - df['o'])
    df['total_size'] = (df['h'] - df['l']).replace(0, 0.001)
    df['body_ratio'] = df['body_size'] / df['total_size']
    
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    window = df.tail(30)
    v_max_down = window[window['c'] < window['o']]
    v_max_up = window[window['c'] > window['o']]
    anchors = {
        'upper': v_max_down.nlargest(1, 'v')['h'].values[0] if not v_max_down.empty else window['h'].max(),
        'lower': v_max_up.nlargest(1, 'v')['l'].values[0] if not v_max_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 4. 实时仪表盘渲染 (3秒心跳)
# ==========================================
def main():
    if 'http_client' not in st.session_state:
        st.session_state.http_client = httpx.Client(timeout=5.0)

    # 侧边栏控制
    st.sidebar.title("⚔️ Warrior Sniper")
    
    # 重要：手动激活按钮（解决浏览器不让自动发声的问题）
    if not st.session_state.voice_active:
        if st.sidebar.button("🔔 点击激活实时语音播报"):
            st.session_state.voice_active = True
            st.rerun()
    else:
        st.sidebar.success("✅ 语音监控已实时挂载")
        if st.sidebar.button("🔇 关闭语音"):
            st.session_state.voice_active = False
            st.rerun()

    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    with st.sidebar.expander("🏹 狙击校准", expanded=False):
        ma_len = st.number_input("均量周期", 5, 50, 10)
        expand_p = st.slider("放量判定 (%)", 100, 300, 150)
        body_r = st.slider("实体比率", 0.05, 0.90, 0.20)
        params = {"ma_len": ma_len, "expand_p": expand_p, "body_r": body_r}

    # UI 占位符
    msg_area = st.empty()
    metric_area = st.empty()
    chart_area = st.empty()
    voice_trigger_area = st.empty() # 核心语音触发点

    # 3秒局部刷新：达到 OKX 散户接口的实操极限
    @st.fragment(run_every="3s")
    def live_monitor():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            resp = st.session_state.http_client.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(resp.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            df, anchors = apply_warrior_logic(df, params)
            curr = df.iloc[-1]

            # 1. 战报与实时播报逻辑
            with msg_area.container():
                audio_text = ""
                if curr['buy_sig'] or curr['c'] > anchors['upper']:
                    status, color, audio_text = "🚀 多头进攻", "#10b981", "发现放量突破，建议做多"
                elif curr['sell_sig'] or curr['c'] < anchors['lower']:
                    status, color, audio_text = "❄️ 空头突袭", "#ef4444", "跌破关键支撑，建议做空"
                else:
                    status, color = "💎 窄幅震荡", "#3b82f6"
                
                st.markdown(f"<div class='status-card' style='border-left:8px solid {color}; padding:15px;'><h2 style='color:{color};margin:0;'>{status} | ETH: ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2></div>", unsafe_allow_html=True)
                
                # 实时语音触发
                if audio_text and st.session_state.voice_active:
                    trigger_voice(audio_text, voice_trigger_area)

            # 2. 实时指标
            with metric_area.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("现价", f"${curr['c']:.2f}")
                m2.metric("量能比", f"{curr['vol_ratio']:.2f}x")
                m3.metric("多头支撑", f"${anchors['lower']:.2f}")
                m4.metric("空头压力", f"${anchors['upper']:.2f}")

            # 3. K线图
            with chart_area.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 信号标记
                buys = df_p[df_p['buy_sig']]
                sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ef4444')), row=1, col=1)

                fig.update_layout(height=580, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0, b=0, l=10, r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception:
            pass

    live_monitor()

if __name__ == "__main__":
    main()
