import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 响应式·全屏布局配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

if "voice_on" not in st.session_state:
    st.session_state.voice_on = False
if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""

st.markdown("""
    <style>
    /* 强制背景色，防止内容未加载时的“黑屏感” */
    .stApp { background-color: #05070a; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; }
    
    /* 控制面板样式 */
    .control-panel {
        background: #111827; border: 1px solid #1f2937;
        padding: 20px; border-radius: 12px; margin-bottom: 20px;
    }
    
    /* 锁定看板高度 */
    [data-testid="stMetric"] { 
        background: #0e1117; border: 1px solid #1f2937; 
        padding: 10px; border-radius: 10px; min-height: 85px; 
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 策略指纹审计核心
# ==========================================
def warrior_audit(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_r'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_r'] > p['body_limit'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_r'] > p['body_limit'])
    
    win = df.tail(30)
    v_up = win[win['c'] > win['o']]; v_down = win[win['c'] < win['o']]
    anchors = {
        'up': v_down.nlargest(1, 'v')['h'].values[0] if not v_down.empty else win['h'].max(),
        'low': v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else win['l'].min()
    }
    return df, anchors

# ==========================================
# 3. 主程序：全页面控制
# ==========================================
def main():
    if 'http' not in st.session_state:
        st.session_state.http = httpx.Client(timeout=5.0)

    # --- 顶部控制面板 (取代侧边栏) ---
    with st.container():
        st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
        col_btn, col_sym, col_ma, col_exp, col_body = st.columns([1.5, 1, 1, 1.5, 1.5])
        
        with col_btn:
            if not st.session_state.voice_on:
                if st.button("🔔 点击激活实时语音播报", type="primary", use_container_width=True):
                    st.session_state.voice_on = True
                    st.rerun()
            else:
                if st.button("🔇 关闭语音", use_container_width=True):
                    st.session_state.voice_on = False
                    st.rerun()
        
        with col_sym:
            symbol = st.text_input("代码", "ETH-USDT-SWAP", label_visibility="collapsed")
        with col_ma:
            ma_val = st.number_input("MA", 5, 50, 10, label_visibility="collapsed")
        with col_exp:
            exp_val = st.slider("放量%", 100, 300, 150, label_visibility="collapsed")
        with col_body:
            body_val = st.slider("实体比", 0.05, 0.9, 0.2, label_visibility="collapsed")
            
        params = {"ma_len": ma_val, "expand_p": exp_val, "body_limit": body_val}
        st.markdown("</div>", unsafe_allow_html=True)

    # 渲染容器
    banner_slot = st.empty(); metric_slot = st.empty(); chart_slot = st.empty(); voice_slot = st.empty()

    @st.fragment(run_every="3s") # 3秒级心跳
    def engine_loop():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.http.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            
            df, anchors = warrior_audit(df, params)
            curr = df.iloc[-1]

            # 1. 顶部战报
            with banner_slot.container():
                audio_msg = ""
                if curr['buy_sig'] or curr['c'] > anchors['up']:
                    txt, cl, audio_msg = "🚀 多头进攻", "#10b981", "放量突破，主力进场"
                elif curr['sell_sig'] or curr['c'] < anchors['low']:
                    txt, cl, audio_msg = "❄️ 空头突袭", "#ef4444", "跌破支撑，空头活跃"
                else:
                    txt, cl = "💎 窄幅震荡", "#3b82f6"
                
                st.markdown(f"""
                    <div style='background:#111827; border-left:8px solid {cl}; padding:15px; border-radius:10px; margin-bottom:15px;'>
                        <h2 style='color:{cl}; margin:0;'>{txt} | ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # 语音执行
                if audio_msg and st.session_state.voice_on and st.session_state.last_alert != audio_msg:
                    with voice_slot:
                        components.html(f"<script>var m=new SpeechSynthesisUtterance('{audio_msg}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)
                    st.session_state.last_alert = audio_msg

            # 2. 实时看板
            with metric_slot.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("实时价", f"${curr['c']:.2f}")
                m2.metric("量能比", f"{curr['vol_r']:.2f}x")
                m3.metric("支撑锚点", f"${anchors['low']:.2f}")
                m4.metric("压力锚点", f"${anchors['up']:.2f}")

            # 3. K线图
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 信号三角形 (size=14)
                buys = df_p[df_p['buy_sig']]; sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ef4444')), row=1, col=1)

                fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=10,r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception:
            banner_slot.warning("📡 数据流同步中...")

    engine_loop()

if __name__ == "__main__":
    main()
