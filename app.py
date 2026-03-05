import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 响应式布局优化 (适配宽屏)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

if "voice_active" not in st.session_state:
    st.session_state.voice_active = False
if "last_msg" not in st.session_state:
    st.session_state.last_msg = ""

st.markdown("""
    <style>
    /* 强制背景色，防止内容未加载时的“黑屏感” */
    .stApp { background-color: #05070a; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; }
    
    /* 锁定看板高度，防止局部刷新导致的视觉跳动 */
    [data-testid="stMetric"] { 
        background: #0e1117; border: 1px solid #1f2937; 
        padding: 10px; border-radius: 10px; min-height: 85px; 
    }
    
    /* 状态战报响应式效果 */
    .banner { 
        padding: 15px; border-radius: 12px; margin-bottom: 15px; 
        border: 1px solid #1f2937; background: #111827;
        transition: all 0.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 实时策略审计核心 (不删减逻辑)
# ==========================================
def apply_warrior_v6_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    # 30根K线动态锚点
    win = df.tail(30)
    v_up = win[win['c'] > win['o']]
    v_down = win[win['c'] < win['o']]
    anchors = {
        'upper': v_down.nlargest(1, 'v')['h'].values[0] if not v_down.empty else win['h'].max(),
        'lower': v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else win['l'].min()
    }
    return df, anchors

# ==========================================
# 3. 实时播报与无闪烁渲染引擎
# ==========================================
def main():
    if 'http' not in st.session_state:
        st.session_state.http = httpx.Client(timeout=5.0)

    # 侧边栏：激活按钮 (解决浏览器禁音)
    st.sidebar.title("⚔️ Warrior Sniper")
    if not st.session_state.voice_active:
        if st.sidebar.button("🔔 激活系统并开启语音"):
            st.session_state.voice_active = True
            st.rerun()
        st.info("系统待机中... 请点击上方按钮启动。")
        return #

    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    with st.sidebar.expander("🏹 狙击核心校准"):
        p = {
            "ma_len": st.number_input("均量周期", 5, 50, 10),
            "expand_p": st.slider("放量判定 (%)", 100, 300, 150),
            "body_r": st.slider("实体比率", 0.05, 0.90, 0.20)
        }

    # 建立固定占位符容器 (防黑屏/防闪烁核心)
    status_slot = st.empty()
    metric_slot = st.empty()
    chart_slot = st.empty()
    voice_slot = st.empty()

    @st.fragment(run_every="3s") # 3秒极速响应
    def live_runner():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.http.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            df, anchors = apply_warrior_v6_logic(df, p)
            curr = df.iloc[-1]

            # 1. 战报与语音
            with status_slot.container():
                audio_txt = ""
                if curr['buy_sig'] or curr['c'] > anchors['upper']:
                    st_txt, cl, audio_txt = "🚀 多头进攻", "#10b981", "放量突破压力位"
                elif curr['sell_sig'] or curr['c'] < anchors['lower']:
                    st_txt, cl, audio_msg = "❄️ 空头突袭", "#ef4444", "跌破关键支撑位"
                else:
                    st_txt, cl = "💎 窄幅震荡", "#3b82f6"
                
                st.markdown(f"<div class='banner' style='border-left:8px solid {cl};'><h2 style='color:{cl};margin:0;'>{st_txt} | ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2></div>", unsafe_allow_html=True)
                
                if audio_txt and st.session_state.last_msg != audio_txt:
                    with voice_slot:
                        components.html(f"<script>var m=new SpeechSynthesisUtterance('{audio_txt}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)
                    st.session_state.last_msg = audio_txt

            # 2. 响应式看板
            with metric_slot.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("现价", f"${curr['c']:.2f}")
                c2.metric("量能比", f"{curr['vol_ratio']:.2f}x")
                c3.metric("支撑点", f"${anchors['lower']:.2f}")
                c4.metric("压力点", f"${anchors['upper']:.2f}")

            # 3. K线绘图 (无闪烁模式)
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
            status_slot.warning("📡 数据流同步中...")

    live_runner()

if __name__ == "__main__":
    main()
