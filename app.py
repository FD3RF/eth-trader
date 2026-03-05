import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 响应式布局与视觉降噪配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

# 初始化语音权限状态
if "voice_ready" not in st.session_state:
    st.session_state.voice_ready = False
if "last_signal" not in st.session_state:
    st.session_state.last_signal = ""

st.markdown("""
    <style>
    /* 极致无边框沉浸效果 */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; max-width: 100%; }
    
    /* Metric 卡片锁定高度，消除重绘闪烁 */
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #1f2937; padding: 10px; border-radius: 10px; min-height: 85px; }
    
    /* 状态战报响应式样式 */
    .status-banner { 
        padding: 15px; border-radius: 12px; margin-bottom: 15px; 
        border: 1px solid #1f2937; transition: all 0.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 高频播报引擎 (Web Speech API)
# ==========================================
def run_voice_alert(text, container):
    """通过独立占位符注入 JS，避免主 UI 刷新中断声音"""
    if st.session_state.last_signal != text:
        with container:
            components.html(f"""
                <script>
                    var msg = new SpeechSynthesisUtterance('{text}');
                    msg.lang = 'zh-CN'; msg.rate = 1.1;
                    window.speechSynthesis.speak(msg);
                </script>
            """, height=0)
        st.session_state.last_signal = text

# ==========================================
# 3. 策略指纹审计逻辑
# ==========================================
def audit_warrior_v6(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_size'] = abs(df['c'] - df['o'])
    df['total_size'] = (df['h'] - df['l']).replace(0, 0.001)
    df['body_ratio'] = df['body_size'] / df['total_size']
    
    # 放量判定
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    # 30根K线动态锚点
    win = df.tail(30)
    v_max_down = win[win['c'] < win['o']]
    v_max_up = win[win['c'] > win['o']]
    anchors = {
        'upper': v_max_down.nlargest(1, 'v')['h'].values[0] if not v_max_down.empty else win['h'].max(),
        'lower': v_max_up.nlargest(1, 'v')['l'].values[0] if not v_max_up.empty else win['l'].min()
    }
    return df, anchors

# ==========================================
# 4. 主渲染循环 (3秒级响应)
# ==========================================
def main():
    if 'client' not in st.session_state:
        st.session_state.client = httpx.Client(timeout=5.0)

    # 侧边栏：激活按钮
    st.sidebar.title("⚔️ Warrior Sniper")
    if not st.session_state.voice_ready:
        if st.sidebar.button("🔔 点击激活实时语音权限"):
            st.session_state.voice_ready = True
            st.rerun()
    else:
        st.sidebar.success("✅ 实时语音监控中...")

    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    with st.sidebar.expander("🏹 狙击核心校准"):
        p = {
            "ma_len": st.number_input("均量周期", 5, 50, 10),
            "expand_p": st.slider("放量判定 (%)", 100, 300, 150),
            "body_r": st.slider("实体比率", 0.05, 0.90, 0.20)
        }

    # 建立固定占位符防止闪烁
    banner_slot = st.empty()
    metric_slot = st.empty()
    chart_slot = st.empty()
    voice_slot = st.empty()

    @st.fragment(run_every="3s")
    def refresh_engine():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.client.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            
            df, anchors = audit_warrior_v6(df, p)
            curr = df.iloc[-1]
            
            # 1. 战报渲染与实时播报
            with banner_slot.container():
                audio_msg = ""
                if curr['buy_sig'] or curr['c'] > anchors['upper']:
                    st, cl, audio_msg = "🚀 多头进攻", "#10b981", "发现放量突破，主力多头进场"
                elif curr['sell_sig'] or curr['c'] < anchors['lower']:
                    st, cl, audio_msg = "❄️ 空头突袭", "#ef4444", "跌破关键支撑，建议关注空单"
                else:
                    st, cl = "💎 窄幅震荡", "#3b82f6"
                
                st.markdown(f"<div class='status-banner' style='background:rgba(17,24,39,0.8); border-left:8px solid {cl};'><h2 style='color:{cl};margin:0;'>{st} | ETH: ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2></div>", unsafe_allow_html=True)
                
                if audio_msg and st.session_state.voice_ready:
                    run_voice_alert(audio_msg, voice_slot)

            # 2. 响应式指标栏
            with metric_slot.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("实时现价", f"${curr['c']:.2f}")
                c2.metric("量能系数", f"{curr['vol_ratio']:.2f}x")
                c3.metric("支撑位", f"${anchors['lower']:.2f}")
                c4.metric("压力位", f"${anchors['upper']:.2f}")

            # 3. K线绘图：视觉降噪版
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 信号三角形 (size=14)
                buys = df_p[df_p['buy_sig']]
                sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ef4444')), row=1, col=1)

                # 标注避让
                fig.add_hline(y=anchors['upper'], line_dash="dash", line_color="#ef4444", annotation_text="RESIST", annotation_position="top right", row=1, col=1)
                fig.add_hline(y=anchors['lower'], line_dash="dash", line_color="#10b981", annotation_text="SUPPORT", annotation_position="bottom right", row=1, col=1)

                fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0, b=0, l=10, r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception:
            pass

    refresh_engine()

if __name__ == "__main__":
    main()
