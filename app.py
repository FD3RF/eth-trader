import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 响应式布局与样式
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

if "voice_on" not in st.session_state:
    st.session_state.voice_on = False
if "last_signal_time" not in st.session_state:
    st.session_state.last_signal_time = None

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 1rem 2rem; }
    
    /* 顶部控制面板 */
    .top-panel {
        background: #111827; border: 1px solid #1f2937;
        padding: 15px; border-radius: 12px; margin-bottom: 15px;
    }
    [data-testid="stMetric"] { 
        background: #0e1117; border: 1px solid #1f2937; 
        padding: 10px; border-radius: 10px; 
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心审计引擎：精准对齐口诀
# ==========================================
def warrior_omni_audit(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_r'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    
    # 锚点计算 (最近30根K线)
    win = df.tail(30)
    v_up = win[win['c'] > win['o']]; v_down = win[win['c'] < win['o']]
    up_anchor = v_down.nlargest(1, 'v')['h'].values[0] if not v_down.empty else win['h'].max()
    low_anchor = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else win['l'].min()

    # --- 口诀判定逻辑 ---
    # 缩量判定 (口诀：缩量是提醒)
    is_shrink = df['vol_ratio'] < 0.7 
    # 放量判定 (口诀：放量是信号)
    is_expand = df['vol_ratio'] > (p['expand'] / 100.0)

    # 做多信号 (放量起涨，突破前高/阴线)
    df['buy_sig'] = is_expand & (df['c'] > df['o']) & (df['c'] > df['c'].shift(1)) & (df['body_r'] > p['body'])
    # 做空信号 (放量下跌，跌破前低/阳线)
    df['sell_sig'] = is_expand & (df['c'] < df['o']) & (df['c'] < df['c'].shift(1)) & (df['body_r'] > p['body'])
    
    # 状态归纳 (用于战报与语音对齐)
    last = df.iloc[-1]
    status = "wait"
    if last['buy_sig']: status = "buy"
    elif last['sell_sig']: status = "sell"
    elif is_shrink.iloc[-1]: status = "shrink"
    
    return df, {'up': up_anchor, 'low': low_anchor, 'status': status}

# ==========================================
# 3. 渲染引擎：战报/语音/信号 三位一体
# ==========================================
def main():
    with st.container():
        st.markdown("<div class='top-panel'>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1.5, 1.5])
        with c1:
            if st.button("🔔 授权语音并启动" if not st.session_state.voice_on else "🔇 关闭语音", type="primary", use_container_width=True):
                st.session_state.voice_on = not st.session_state.voice_on
                st.rerun()
        with c2: symbol = st.text_input("代码", "ETH-USDT-SWAP", label_visibility="collapsed")
        with c3: ma_len = st.number_input("均量", 5, 50, 10, label_visibility="collapsed")
        with c4: expand = st.slider("放量触发%", 100, 300, 150, label_visibility="collapsed")
        with c5: body = st.slider("实体过滤", 0.05, 0.9, 0.2, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    # 渲染容器
    banner_slot = st.empty(); metric_slot = st.empty(); chart_slot = st.empty(); voice_slot = st.empty()

    if 'http' not in st.session_state:
        st.session_state.http = httpx.Client(timeout=5.0)

    @st.fragment(run_every="3s")
    def engine():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.http.get(url, params={"instId": symbol, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            
            df, res = warrior_omni_audit(df, {"ma_len": ma_len, "expand": expand, "body": body})
            curr = df.iloc[-1]

            # --- 核心：三位一体对齐控制 ---
            # 1. 战报与语音文字定义 (严格对应口诀)
            status_map = {
                "buy": {"txt": "🚀 放量起涨：突破开多", "color": "#10b981", "voice": "放量起涨，突破前高，直接开多"},
                "sell": {"txt": "❄️ 放量下跌：破位开空", "color": "#ef4444", "voice": "放量下跌，跌破前低，直接开空"},
                "shrink": {"txt": "💎 缩量回踩：只看不动", "color": "#3b82f6", "voice": ""}, # 缩量不播报语音，避免吵闹
                "wait": {"txt": "📊 震荡蓄势：等待信号", "color": "#9ca3af", "voice": ""}
            }
            conf = status_map[res['status']]

            # 2. 渲染顶部实时战报
            with banner_slot.container():
                st.markdown(f"""
                    <div style='background:#111827; border-left:8px solid {conf['color']}; padding:15px; border-radius:10px; margin-bottom:15px;'>
                        <h2 style='color:{conf['color']}; margin:0;'>{conf['txt']} | ${curr['c']:.2f} | {datetime.now().strftime('%H:%M:%S')}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # 3. 触发语音播报 (仅在信号首次出现时)
            if conf['voice'] and st.session_state.voice_on:
                sig_key = f"{res['status']}_{curr['ts']}"
                if st.session_state.get("last_sig_key") != sig_key:
                    with voice_slot:
                        components.html(f"<script>var m=new SpeechSynthesisUtterance('{conf['voice']}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)
                    st.session_state.last_sig_key = sig_key

            # 4. 指标看板
            with metric_slot.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("当前价格", f"${curr['c']:.2f}")
                m2.metric("实时量能比", f"{curr['vol_ratio']:.2f}x")
                m3.metric("多头支撑", f"${res['low']:.2f}")
                m4.metric("空头压力", f"${res['up']:.2f}")

            # 5. K线图表渲染 (标记信号三角形)
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 信号标记 (与战报逻辑严格对齐)
                buys = df_p[df_p['buy_sig']]; sells = df_p[df_p['sell_sig']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=15, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=15, color='#ef4444')), row=1, col=1)

                fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=10,r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception:
            banner_slot.warning("📡 数据流同步中...")

    engine()

if __name__ == "__main__":
    main()
