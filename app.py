import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import streamlit.components.v1 as components

# ==========================================
# 1. 置顶战报 CSS 注入
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    /* 核心：将战报容器变为置顶悬浮 */
    .sticky-report {
        position: fixed; top: 50px; left: 350px; right: 20px;
        z-index: 1000; background: rgba(13, 17, 23, 0.95);
        border: 2px solid #30363d; border-radius: 12px;
        padding: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
    }
    .report-title { font-size: 32px; font-weight: 800; margin: 0; }
    /* 呼吸灯特效 */
    @keyframes glow-green { 0% { box-shadow: 0 0 5px #10b981; } 50% { box-shadow: 0 0 25px #10b981; } 100% { box-shadow: 0 0 5px #10b981; } }
    @keyframes glow-red { 0% { box-shadow: 0 0 5px #ef4444; } 50% { box-shadow: 0 0 25px #ef4444; } 100% { box-shadow: 0 0 5px #ef4444; } }
    .glow-buy { animation: glow-green 1.5s infinite; border-color: #10b981 !important; }
    .glow-sell { animation: glow-red 1.5s infinite; border-color: #ef4444 !important; }
    
    /* 修正下方内容间距，防止被遮挡 */
    .content-area { margin-top: 130px; }
    </style>
""", unsafe_allow_html=True)

if "sig_history" not in st.session_state: st.session_state.sig_history = []
if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig_ts" not in st.session_state: st.session_state.last_sig_ts = ""

# ==========================================
# 2. 高频计算引擎
# ==========================================
def warrior_core_logic(data, p):
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
    for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
    df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df = df.sort_values('time').reset_index(drop=True)
    
    # 均量判定
    ma_v = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / ma_v.replace(0, 1e-9)
    
    # 寻找锚点
    lookback = df.tail(30)
    v_dn = lookback[lookback['c'] < lookback['o']]
    v_up = lookback[lookback['c'] > lookback['o']]
    press = v_dn.nlargest(1, 'v')['h'].values[0] if not v_dn.empty else lookback['h'].max()
    supp = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else lookback['l'].min()

    # 信号捕捉
    df['buy_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] > df['o']) & (df['c'] > press)
    df['sell_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] < df['o']) & (df['c'] < supp)
    
    return df, {'press': press, 'supp': supp, 'shrink': df['vol_r'].iloc[-1] < 0.7}

# ==========================================
# 3. 实时中控
# ==========================================
def main():
    with st.sidebar:
        st.header("⚔️ Sniper V6.2 Pro")
        if st.button("🔔 启动监听" if not st.session_state.voice_on else "🛑 停止监听", type="primary", use_container_width=True):
            st.session_state.voice_on = not st.session_state.voice_on
            st.rerun()
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量触发%", 100, 300, 150)
        sym = st.text_input("合约", "ETH-USDT-SWAP")

    # 定义悬浮战报槽位
    report_slot = st.empty()
    
    # 下方内容区
    st.markdown("<div class='content-area'></div>", unsafe_allow_html=True)
    col_main, col_hist = st.columns([3.2, 0.8])

    with col_main:
        metric_slot = st.empty(); chart_slot = st.empty(); voice_slot = st.empty()
    with col_hist:
        st.markdown("### 📜 信号历史")
        hist_container = st.container()

    @st.fragment(run_every="2s")
    def tick():
        try:
            with httpx.Client(http2=True, timeout=2.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": sym, "bar": "5m", "limit": "80"})
                df, res = warrior_core_logic(r.json()['data'], {"ma_len": ma_len, "exp": exp})
            
            curr = df.iloc[-1]
            
            # --- 1. 悬浮战报逻辑 (关键升级) ---
            if curr['buy_tri']:
                st_txt, st_col, glow_cls, voice = "🚀 多头突袭：放量突破开多", "#10b981", "glow-buy", "放量起涨，突破压力，直接开多"
            elif curr['sell_tri']:
                st_txt, st_col, glow_cls, voice = "❄️ 空头突袭：放量跌破开空", "#ef4444", "glow-sell", "放量下跌，跌破支撑，直接开空"
            elif res['shrink']:
                st_txt, st_col, glow_cls, voice = "💎 缩量震荡：只看不动", "#3b82f6", "", ""
            else:
                st_txt, st_col, glow_cls, voice = "📊 震荡蓄势：等待信号", "#9ca3af", "", ""

            report_slot.markdown(f"""
                <div class='sticky-report {glow_cls}'>
                    <p style='color:#8b949e; margin:0; font-size:14px;'>核心逻辑：当前量能 {curr['vol_r']:.2f}x | 监控中...</p>
                    <h1 class='report-title' style='color:{st_col};'>{st_txt} | ${curr['c']:.2f}</h1>
                </div>
            """, unsafe_allow_html=True)

            # --- 2. 信号存储与语音 ---
            if (curr['buy_tri'] or curr['sell_tri']) and st.session_state.last_sig_ts != curr['ts']:
                st.session_state.sig_history.insert(0, {"t": curr['time'].strftime('%H:%M'), "type": st_txt[:4], "p": curr['c']})
                st.session_state.last_sig_ts = curr['ts']
                if st.session_state.voice_on and voice:
                    with voice_slot: components.html(f"<script>var m=new SpeechSynthesisUtterance('{voice}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)

            # --- 3. 渲染指标与K线 (确保多空三角不删减) ---
            with metric_slot.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("量能比", f"{curr['vol_r']:.2f}x")
                m2.metric("多头支撑锚点", f"${res['supp']:.2f}")
                m3.metric("空头压力锚点", f"${res['press']:.2f}")

            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                # 强化多空三角
                b = df_p[df_p['buy_tri']]; s = df_p[df_p['sell_tri']]
                fig.add_trace(go.Scatter(x=b['time'], y=b['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=16, color='#10b981', line=dict(width=1, color='white'))), row=1, col=1)
                fig.add_trace(go.Scatter(x=s['time'], y=s['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=16, color='#ef4444', line=dict(width=1, color='white'))), row=1, col=1)
                fig.update_layout(height=500, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with hist_container:
                for item in st.session_state.sig_history[:10]:
                    st.markdown(f"<div style='background:#161b22; padding:8px; border-radius:5px; margin-bottom:5px; border-left:4px solid #58a6ff;'>[{item['t']}] <b>{item['type']}</b> @ {item['p']}</div>", unsafe_allow_html=True)

        except Exception: pass

    tick()

if __name__ == "__main__": main()
