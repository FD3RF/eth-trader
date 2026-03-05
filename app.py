import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 配置与响应式 CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

# 初始化语音状态
if "last_alert_msg" not in st.session_state:
    st.session_state.last_alert_msg = ""

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; max-width: 100%; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #1f2937; padding: 10px; border-radius: 10px; min-height: 80px; }
    .status-card { transition: all 0.3s ease; background: #111827; border-radius: 12px; margin-bottom: 10px; border: 1px solid #1f2937; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心语音引擎 (修复：支持局部刷新)
# ==========================================
def trigger_voice(text):
    """确保在局部刷新中也能稳定触发语音"""
    if st.session_state.last_alert_msg != text:
        # 使用组件注入 JS 播放声音
        components.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance('{text}');
                msg.lang = 'zh-CN';
                msg.rate = 1.1; 
                msg.pitch = 1.0;
                window.speechSynthesis.speak(msg);
            </script>
        """, height=0)
        st.session_state.last_alert_msg = text

# ==========================================
# 3. 策略逻辑 (不删减)
# ==========================================
def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_size'] = abs(df['c'] - df['o'])
    df['total_size'] = (df['h'] - df['l']).replace(0, 0.001)
    df['body_ratio'] = df['body_size'] / df['total_size']
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
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
# 4. 仪表盘循环 (局部刷新范围)
# ==========================================
def main():
    if 'http_client' not in st.session_state:
        st.session_state.http_client = httpx.Client(timeout=10.0)

    st.sidebar.title("⚔️ Warrior Sniper")
    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    with st.sidebar.expander("🏹 核心校准", expanded=False):
        ma_len = st.number_input("均量周期", 5, 50, 10)
        expand_p = st.slider("放量判定 (%)", 100, 300, 150)
        body_r = st.slider("实体比率", 0.05, 0.90, 0.20)
        params = {"ma_len": ma_len, "expand_p": expand_p, "body_r": body_r}

    # 布局占位
    msg_placeholder = st.empty()
    metric_placeholder = st.empty()
    chart_placeholder = st.empty()
    voice_placeholder = st.empty() # 专门用于存放语音 JS 的隐形容器

    @st.fragment(run_every="10s")
    def sync_data():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            resp = st.session_state.http_client.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(resp.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            df, anchors = apply_warrior_logic(df, params)
            curr = df.iloc[-1]

            # 1. 战报与语音触发
            with msg_placeholder.container():
                if curr['buy_sig'] or curr['c'] > anchors['upper']:
                    status, color, audio_txt = "🚀 多头进攻", "#10b981", "放量突破压力位"
                elif curr['sell_sig'] or curr['c'] < anchors['lower']:
                    status, color, audio_txt = "❄️ 空头突袭", "#ef4444", "跌破支撑警戒线"
                else:
                    status, color, audio_txt = "💎 窄幅震荡", "#3b82f6", ""
                
                st.markdown(f"<div class='status-card' style='border-left:8px solid {color}; padding:15px;'><h2 style='color:{color};margin:0;'>{status} | ETH: ${curr['c']:.2f}</h2></div>", unsafe_allow_html=True)
                
                # 在 voice_placeholder 中触发语音
                if audio_txt:
                    with voice_placeholder:
                        trigger_voice(audio_txt)

            # 2. 指标展示
            with metric_placeholder.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("现价", f"${curr['c']:.2f}")
                m2.metric("量能比", f"{curr['vol_ratio']:.2f}x")
                m3.metric("多头支撑", f"${anchors['lower']:.2f}")
                m4.metric("空头压力", f"${anchors['upper']:.2f}")

            # 3. K线绘图 (视觉降噪优化)
            with chart_placeholder.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 信号点 (size=14)
                buys = df_p[df_p['buy_sig']]
                sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ef4444')), row=1, col=1)

                # 压力支撑线 (右侧标注)
                fig.add_hline(y=anchors['upper'], line_dash="dash", line_color="#ef4444", annotation_text="RESIST", annotation_position="top right", row=1, col=1)
                fig.add_hline(y=anchors['lower'], line_dash="dash", line_color="#10b981", annotation_text="SUPPORT", annotation_position="bottom right", row=1, col=1)

                # 量能柱 (opacity=0.6)
                v_colors = ['#10b981' if c >= o else '#ef4444' for c, o in zip(df_p['c'], df_p['o'])]
                fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], marker_color=v_colors, opacity=0.6), row=2, col=1)

                fig.update_layout(height=580, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0, b=0, l=10, r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception as e:
            st.error(f"连接中断: {e}")

    sync_data()

if __name__ == "__main__":
    main()
