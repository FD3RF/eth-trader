import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

# 状态初始化
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False
if "last_alert_msg" not in st.session_state:
    st.session_state.last_alert_msg = ""

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stMetric"] { 
        background: #0e1117; border: 1px solid #1f2937; 
        padding: 10px; border-radius: 10px; 
    }
    /* 侧边栏样式强化 */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 侧边栏：核心控制台 (强制优先渲染)
# ==========================================
with st.sidebar:
    st.title("⚔️ Warrior Sniper V6.2")
    st.markdown("---")
    
    # 语音授权按钮 - 这是解决“不出声”的关键
    if not st.session_state.voice_active:
        if st.button("🔔 授权并激活实时语音", use_container_width=True, type="primary"):
            st.session_state.voice_active = True
            st.rerun()
    else:
        st.success("✅ 语音播报已挂载")
        if st.button("🔇 关闭语音", use_container_width=True):
            st.session_state.voice_active = False
            st.rerun()
    
    st.markdown("---")
    symbol = st.text_input("交易对", "ETH-USDT-SWAP")
    
    with st.expander("🏹 狙击核心校准", expanded=True):
        ma_len = st.number_input("均量周期", 5, 50, 10)
        expand_p = st.slider("放量判定 (%)", 100, 300, 150)
        body_r = st.slider("实体比率", 0.05, 0.90, 0.20)
        params = {"ma_len": ma_len, "expand_p": expand_p, "body_r": body_r}

# ==========================================
# 3. 策略逻辑 (不删减)
# ==========================================
def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    win = df.tail(30)
    v_up = win[win['c'] > win['o']]; v_down = win[win['c'] < win['o']]
    anchors = {
        'upper': v_down.nlargest(1, 'v')['h'].values[0] if not v_down.empty else win['h'].max(),
        'lower': v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else win['l'].min()
    }
    return df, anchors

# ==========================================
# 4. 主看板渲染引擎 (3秒响应)
# ==========================================
def main_dashboard():
    # 占位符定义
    banner_slot = st.empty()
    metric_slot = st.empty()
    chart_slot = st.empty()
    voice_slot = st.empty()

    if 'client' not in st.session_state:
        st.session_state.client = httpx.Client(timeout=5.0)

    @st.fragment(run_every="3s")
    def refresh_loop():
        try:
            # 数据抓取
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.client.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            
            df, anchors = apply_warrior_logic(df, params)
            curr = df.iloc[-1]

            # 渲染顶部战报
            with banner_slot.container():
                audio_text = ""
                if curr['buy_sig'] or curr['c'] > anchors['upper']:
                    msg, color, audio_text = "🚀 多头进攻", "#10b981", "发现放量突破，主力进场"
                elif curr['sell_sig'] or curr['c'] < anchors['lower']:
                    msg, color, audio_text = "❄️ 空头突袭", "#ef4444", "跌破关键支撑，趋势转弱"
                else:
                    msg, color = "💎 窄幅震荡", "#3b82f6"
                
                st.markdown(f"""
                    <div style="background:#111827; border-left:8px solid {color}; padding:15px; border-radius:10px; margin-bottom:15px;">
                        <h2 style="color:{color}; margin:0;">{msg} | ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # 语音播报逻辑
                if audio_text and st.session_state.voice_active and st.session_state.last_alert_msg != audio_text:
                    with voice_slot:
                        components.html(f"""
                            <script>
                                var msg = new SpeechSynthesisUtterance('{audio_text}');
                                msg.lang = 'zh-CN';
                                window.speechSynthesis.speak(msg);
                            </script>
                        """, height=0)
                    st.session_state.last_alert_msg = audio_text

            # 渲染指标
            with metric_slot.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"${curr['c']:.2f}")
                c2.metric("量能系数", f"{curr['vol_ratio']:.2f}x")
                c3.metric("多头支撑", f"${anchors['lower']:.2f}")
                c4.metric("空头压力", f"${anchors['upper']:.2f}")

            # 渲染K线图
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                buys = df_p[df_p['buy_sig']]; sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ef4444')), row=1, col=1)

                fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=10,r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception:
            banner_slot.warning("📡 正在尝试重连 OKX 行情服务器...")

    refresh_loop()

if __name__ == "__main__":
    main_dashboard()
