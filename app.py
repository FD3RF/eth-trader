import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 极致性能与 UI 配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700; color: #58a6ff; }
    .status-box { background: #0d1117; border-left: 10px solid #30363d; padding: 18px; border-radius: 8px; margin-bottom: 12px; }
    .history-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# 初始化核心状态
if "sig_history" not in st.session_state: st.session_state.sig_history = []
if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig_ts" not in st.session_state: st.session_state.last_sig_ts = ""

# ==========================================
# 2. 向量化计算引擎 (确保信号不删减)
# ==========================================
@st.cache_data(ttl=1)
def fast_warrior_engine(data_raw, p):
    df = pd.DataFrame(data_raw, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
    for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
    df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df = df.sort_values('time').reset_index(drop=True)

    # 均量与量能比计算 (Numpy 加速)
    c, o, h, l, v = df['c'].values, df['o'].values, df['h'].values, df['l'].values, df['v'].values
    ma_v = pd.Series(v).rolling(p['ma_len']).mean().values
    vol_r = np.divide(v, ma_v, out=np.zeros_like(v), where=ma_v!=0)
    
    # 锚点逻辑 (最近30根放量K线)
    win_v, win_c, win_o, win_h, win_l = v[-30:], c[-30:], o[-30:], h[-30:], l[-30:]
    
    dn_mask = (win_c < win_o) # 阴线
    press = win_h[dn_mask][np.argmax(win_v[dn_mask])] if dn_mask.any() else np.max(win_h)
    
    up_mask = (win_c > win_o) # 阳线
    supp = win_l[up_mask][np.argmax(win_v[up_mask])] if up_mask.any() else np.min(win_l)

    # 信号判定
    curr_v_r = vol_r[-1]
    is_expand = curr_v_r > (p['exp'] / 100.0)
    
    df['vol_r'] = vol_r
    df['buy_tri'] = (vol_r > (p['exp'] / 100.0)) & (c > o) & (c > press)
    df['sell_tri'] = (vol_r > (p['exp'] / 100.0)) & (c < o) & (c < supp)
    
    return df, {'press': press, 'supp': supp, 'vol_r': curr_v_r, 'shrink': curr_v_r < 0.7}

# ==========================================
# 3. 实时中控与历史记录面板
# ==========================================
def main():
    with st.sidebar:
        st.header("⚔️ 勇士狙击手 V6.2 Pro")
        if st.button("🔔 激活监听/授权语音" if not st.session_state.voice_on else "🛑 停止监听", type="primary", use_container_width=True):
            st.session_state.voice_on = not st.session_state.voice_on
            st.rerun()
        
        st.subheader("参数校准")
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量触发%", 100, 300, 150)
        sym = st.text_input("合约", "ETH-USDT-SWAP")
        if st.button("清除信号记录"): st.session_state.sig_history = []; st.rerun()

    col_left, col_right = st.columns([3.2, 0.8])

    with col_left:
        banner_slot = st.empty(); metric_slot = st.empty(); chart_slot = st.empty(); voice_slot = st.empty()

    with col_right:
        st.markdown("### 📜 信号历史记录")
        hist_container = st.container()

    @st.fragment(run_every="2s")
    def tick():
        try:
            with httpx.Client(http2=True, timeout=2.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": sym, "bar": "5m", "limit": "80"})
                data = r.json()['data']
            
            df, res = fast_warrior_engine(data, {"ma_len": ma_len, "exp": exp})
            curr = df.iloc[-1]

            # 信号入库与语音播报
            if (curr['buy_tri'] or curr['sell_tri']) and st.session_state.last_sig_ts != curr['ts']:
                s_type = "做多 🚀" if curr['buy_tri'] else "做空 ❄️"
                st.session_state.sig_history.insert(0, {"t": curr['time'].strftime('%H:%M'), "type": s_type, "p": curr['c'], "v": f"{curr['vol_r']:.2f}x"})
                st.session_state.last_sig_ts = curr['ts']
                if st.session_state.voice_on:
                    v_cmd = "放量起涨，突破压力，直接开多" if curr['buy_tri'] else "放量下跌，跌破支撑，直接开空"
                    with voice_slot: components.html(f"<script>var m=new SpeechSynthesisUtterance('{v_cmd}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)

            # 战报渲染
            st_msg, cl = ("🚀 多头突袭", "#10b981") if curr['buy_tri'] else ("❄️ 空头突袭", "#ef4444") if curr['sell_tri'] else ("💎 缩量震荡", "#3b82f6") if res['shrink'] else ("📊 等待信号", "#9ca3af")
            banner_slot.markdown(f"<div class='status-box' style='border-left-color:{cl};'><h2 style='color:{cl};margin:0;'>{st_msg} | ${curr['c']:.2f}</h2></div>", unsafe_allow_html=True)
            
            with metric_slot.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("现价", f"${curr['c']:.2f}")
                m2.metric("量能比", f"{curr['vol_r']:.2f}x")
                m3.metric("强支撑", f"${res['supp']:.2f}")
                m4.metric("强压力", f"${res['press']:.2f}")

            # K线渲染 (多空三角强化)
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
                df_p = df.tail(55)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 绘制显式多空三角
                b_tri = df_p[df_p['buy_tri']]; s_tri = df_p[df_p['sell_tri']]
                fig.add_trace(go.Scatter(x=b_tri['time'], y=b_tri['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=16, color='#10b981', line=dict(width=1, color='white'))), row=1, col=1)
                fig.add_trace(go.Scatter(x=s_tri['time'], y=s_tri['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=16, color='#ef4444', line=dict(width=1, color='white'))), row=1, col=1)
                
                # 绘制支撑压力锚点线
                fig.add_hline(y=res['press'], line_dash="dash", line_color="#ef4444", opacity=0.4, row=1, col=1)
                fig.add_hline(y=res['supp'], line_dash="dash", line_color="#10b981", opacity=0.4, row=1, col=1)
                
                fig.update_layout(height=550, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=5,b=0,l=0,r=30))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # 历史记录
            with hist_container:
                for item in st.session_state.sig_history[:12]:
                    h_cl = "#10b981" if "🚀" in item['type'] else "#ef4444"
                    st.markdown(f"<div class='history-card'><span style='color:#8b949e;'>[{item['t']}]</span> <b style='color:{h_cl};'>{item['type']}</b><br/>价: {item['p']} | 量: {item['v']}</div>", unsafe_allow_html=True)

        except Exception: pass

    tick()

if __name__ == "__main__": main()
