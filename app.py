import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 极致布局与防黑屏 CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

if "voice_active" not in st.session_state:
    st.session_state.voice_active = False
if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""

st.markdown("""
    <style>
    /* 强制背景色，防止黑屏感 */
    .stApp { background-color: #05070a; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; }
    
    /* 指标栏占位背景 */
    [data-testid="stMetric"] { 
        background: #0e1117; border: 1px solid #1f2937; 
        padding: 10px; border-radius: 10px; min-height: 85px; 
    }
    
    /* 骨架屏动画：加载时不会完全黑屏 */
    .loading-shimmer {
        background: linear-gradient(90deg, #0e1117 25%, #161b22 50%, #0e1117 75%);
        background-size: 200% 100%; animation: shimmer 2s infinite;
    }
    @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 实时策略逻辑 (不删减)
# ==========================================
def apply_warrior_logic(df, p):
    try:
        df = df.dropna().reset_index(drop=True)
        df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
        df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
        df['body_size'] = abs(df['c'] - df['o'])
        df['total_size'] = (df['h'] - df['l']).replace(0, 0.001)
        df['body_ratio'] = df['body_size'] / df['total_size']
        
        df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
        df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
        df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
        
        win = df.tail(30)
        v_up = win[win['c'] > win['o']]
        v_down = win[win['c'] < win['o']]
        anchors = {
            'upper': v_down.nlargest(1, 'v')['h'].values[0] if not v_down.empty else win['h'].max(),
            'lower': v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else win['l'].min()
        }
        return df, anchors
    except:
        return df, {'upper': 0, 'lower': 0}

# ==========================================
# 3. 实时播报与绘图 (Fragment 架构)
# ==========================================
def main():
    if 'client' not in st.session_state:
        st.session_state.client = httpx.Client(timeout=5.0)

    # 侧边栏：初始化与权限
    st.sidebar.title("⚔️ Warrior Sniper")
    if not st.session_state.voice_active:
        if st.sidebar.button("🔔 点击激活语音并启动系统"):
            st.session_state.voice_active = True
            st.rerun()
        st.warning("系统待机中，请点击上方按钮激活。")
        return # 未点击前不进入渲染循环，防止空白渲染导致黑屏

    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    with st.sidebar.expander("🏹 核心校准"):
        p = {
            "ma_len": st.number_input("均量周期", 5, 50, 10),
            "expand_p": st.slider("放量判定 (%)", 100, 300, 150),
            "body_r": st.slider("实体比率", 0.05, 0.90, 0.20)
        }

    # 预设占位符，防止页面高度坍塌
    header_box = st.empty()
    metric_box = st.empty()
    chart_box = st.empty()
    voice_box = st.empty()

    @st.fragment(run_every="3s")
    def runner():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.client.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            data = r.json().get('data', [])
            if not data:
                header_box.error("📡 数据源重连中...")
                return

            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            df, anchors = apply_warrior_logic(df, p)
            curr = df.iloc[-1]

            # 1. 顶部状态与实时播报
            with header_box.container():
                audio_msg = ""
                if curr['buy_sig'] or curr['c'] > anchors['upper']:
                    st_txt, col, audio_msg = "🚀 多头进攻", "#10b981", "放量突破压力"
                elif curr['sell_sig'] or curr['c'] < anchors['lower']:
                    st_txt, col, audio_msg = "❄️ 空头突袭", "#ef4444", "跌破关键支撑"
                else:
                    st_txt, col = "💎 窄幅震荡", "#3b82f6"
                
                st.markdown(f"""
                    <div style="background: #111827; border-left: 8px solid {col}; padding: 15px; border-radius: 10px;">
                        <h2 style="color: {col}; margin: 0;">{st_txt} | ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                if audio_msg and st.session_state.voice_active and st.session_state.last_alert != audio_msg:
                    with voice_box:
                        components.html(f"<script>var m=new SpeechSynthesisUtterance('{audio_msg}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)
                    st.session_state.last_alert = audio_msg

            # 2. 响应式指标
            with metric_box.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"${curr['c']:.2f}")
                c2.metric("量能比", f"{curr['vol_ratio']:.2f}x")
                c3.metric("多头支撑", f"${anchors['lower']:.2f}")
                c4.metric("空头压力", f"${anchors['upper']:.2f}")

            # 3. 绘图引擎 (防跳动渲染)
            with chart_box.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 信号点 (size=14)
                buys = df_p[df_p['buy_sig']]
                sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ef4444')), row=1, col=1)

                fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=10,r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception as e:
            header_box.warning(f"🔄 自动重连中... ({e})")

    runner()

if __name__ == "__main__":
    main()
