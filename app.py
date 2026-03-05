import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 沉浸式宽屏布局
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

# 初始化状态：默认为数据运行，语音待授权
if "voice_granted" not in st.session_state:
    st.session_state.voice_granted = False
if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; }
    [data-testid="stMetric"] { 
        background: #0e1117; border: 1px solid #1f2937; 
        padding: 10px; border-radius: 10px; min-height: 85px; 
    }
    .banner-card { 
        padding: 15px; border-radius: 12px; margin-bottom: 15px; 
        border: 1px solid #1f2937; background: #111827;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心审计引擎
# ==========================================
def run_strategy_audit(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_r'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    
    # 策略指纹判定
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
# 3. 主程序控制台
# ==========================================
def main():
    if 'client' not in st.session_state:
        st.session_state.client = httpx.Client(timeout=5.0)

    # 侧边栏：状态管理
    st.sidebar.title("⚔️ Warrior Sniper")
    
    # 语音授权按钮
    if not st.session_state.voice_granted:
        if st.sidebar.button("🔔 点击授权实时语音播报"):
            st.session_state.voice_granted = True
            st.rerun()
    else:
        st.sidebar.success("✅ 语音监控已就绪")

    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    with st.sidebar.expander("🏹 狙击参数校准"):
        p = {
            "ma_len": st.number_input("均量周期", 5, 50, 10),
            "expand_p": st.slider("放量判定 (%)", 100, 300, 150),
            "body_limit": st.slider("实体比率", 0.05, 0.90, 0.20)
        }

    # UI 占位容器
    top_slot = st.empty(); metric_slot = st.empty(); chart_slot = st.empty(); voice_slot = st.empty()

    @st.fragment(run_every="3s") # 3秒心跳刷新
    def engine():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.client.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            
            df, anchors = run_strategy_audit(df, p)
            curr = df.iloc[-1]

            # 1. 顶部战报
            with top_slot.container():
                audio_text = ""
                if curr['buy_sig'] or curr['c'] > anchors['up']:
                    txt, cl, audio_text = "🚀 多头进攻", "#10b981", "放量突破，主力进场"
                elif curr['sell_sig'] or curr['c'] < anchors['low']:
                    txt, cl, audio_text = "❄️ 空头突袭", "#ef4444", "跌破支撑，趋势转弱"
                else:
                    txt, cl = "💎 窄幅震荡", "#3b82f6"
                
                st.markdown(f"<div class='banner-card' style='border-left:8px solid {cl};'><h2 style='color:{cl};margin:0;'>{txt} | ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2></div>", unsafe_allow_html=True)
                
                # 播报执行
                if audio_text and st.session_state.voice_granted and st.session_state.last_alert != audio_text:
                    with voice_slot:
                        components.html(f"<script>var m=new SpeechSynthesisUtterance('{audio_text}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)
                    st.session_state.last_alert = audio_text

            # 2. 数据看板
            with metric_slot.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("实时价", f"${curr['c']:.2f}")
                m2.metric("量能比", f"{curr['vol_ratio']:.2f}x")
                m3.metric("多头锚点", f"${anchors['low']:.2f}")
                m4.metric("空头锚点", f"${anchors['up']:.2f}")

            # 3. K线渲染
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 信号标记
                buys = df_p[df_p['buy_sig']]; sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ef4444')), row=1, col=1)

                fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=10,r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception:
            top_slot.warning("📡 正在同步 OKX 实时数据...")

    engine()

if __name__ == "__main__":
    main()
