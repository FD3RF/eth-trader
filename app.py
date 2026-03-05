import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 极致性能配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700; color: #58a6ff; }
    .status-box { background: #0d1117; border-left: 10px solid #30363d; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .history-card { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# 初始化全局信号历史记录
if "sig_history" not in st.session_state: st.session_state.sig_history = []
if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig_ts" not in st.session_state: st.session_state.last_sig_ts = ""

# ==========================================
# 2. 核心逻辑引擎 (含信号捕捉)
# ==========================================
def warrior_engine_v2(df, p):
    df = df.copy()
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    
    # 定位支撑压力锚点
    lookback = df.tail(30)
    v_up = lookback[lookback['c'] > lookback['o']]
    v_dn = lookback[lookback['c'] < lookback['o']]
    press = v_dn.nlargest(1, 'v')['h'].values[0] if not v_dn.empty else lookback['h'].max()
    supp = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else lookback['l'].min()

    # 信号判定：放量 + 突破
    df['is_expand'] = df['vol_r'] > (p['exp'] / 100.0)
    df['buy_tri'] = df['is_expand'] & (df['c'] > df['o']) & (df['c'] > press)
    df['sell_tri'] = df['is_expand'] & (df['c'] < df['o']) & (df['c'] < supp)
    
    return df, {'press': press, 'supp': supp}

# ==========================================
# 3. 渲染主引擎
# ==========================================
def main():
    with st.sidebar:
        st.header("⚔️ 勇士狙击手 V6.2")
        if st.button("🔔 启动并授权语音" if not st.session_state.voice_on else "🛑 停止监听", type="primary", use_container_width=True):
            st.session_state.voice_on = not st.session_state.voice_on
            st.rerun()
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量触发%", 100, 300, 150)
        sym = st.text_input("合约代码", "ETH-USDT-SWAP")
        if st.button("清除历史记录"): 
            st.session_state.sig_history = []
            st.rerun()

    # 界面布局：左侧监控，右侧历史
    col_main, col_hist = st.columns([3.2, 0.8])

    with col_main:
        banner_slot = st.empty()
        metric_slot = st.empty()
        chart_slot = st.empty()
        voice_slot = st.empty()

    with col_hist:
        st.markdown("### 📜 信号历史记录")
        hist_container = st.container()

    @st.fragment(run_every="2s")
    def tick():
        try:
            with httpx.Client(timeout=3.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": sym, "bar": "5m", "limit": "100"})
                data = r.json()['data']
            
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            
            df, res = warrior_engine_v2(df, {"ma_len": ma_len, "exp": exp})
            curr = df.iloc[-1]

            # --- 信号捕捉与历史入库 ---
            if (curr['buy_tri'] or curr['sell_tri']) and st.session_state.last_sig_ts != curr['ts']:
                sig_type = "做多 🚀" if curr['buy_tri'] else "做空 ❄️"
                st.session_state.sig_history.insert(0, {
                    "time": curr['time'].strftime('%H:%M'),
                    "type": sig_type,
                    "price": curr['c'],
                    "vol": f"{curr['vol_r']:.2f}x"
                })
                st.session_state.last_sig_ts = curr['ts']
                # 语音逻辑
                if st.session_state.voice_on:
                    v_msg = "放量突破，直接开多" if curr['buy_tri'] else "放量跌破，直接开空"
                    with voice_slot:
                        components.html(f"<script>var m=new SpeechSynthesisUtterance('{v_msg}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)

            # --- 渲染逻辑 ---
            # 1. 战报
            color = "#10b981" if curr['buy_tri'] else "#ef4444" if curr['sell_tri'] else "#9ca3af"
            status = "多头突袭" if curr['buy_tri'] else "空头突袭" if curr['sell_tri'] else "震荡蓄势"
            banner_slot.markdown(f"<div class='status-box' style='border-left-color:{color};'><h2 style='color:{color};margin:0;'>{status} | ${curr['c']:.2f}</h2></div>", unsafe_allow_html=True)
            
            # 2. 指标
            with metric_slot.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("量能比", f"{curr['vol_r']:.2f}x")
                m2.metric("强支撑", f"${res['supp']:.2f}")
                m3.metric("强压力", f"${res['press']:.2f}")

            # 3. K线与三角信号
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
                df_p = df.tail(50)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 绘制信号三角 (确保不被删除)
                buys = df_p[df_p['buy_tri']]
                sells = df_p[df_p['sell_tri']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=15, color='#10b981', line=dict(width=1, color='white'))), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=15, color='#ef4444', line=dict(width=1, color='white'))), row=1, col=1)
                
                fig.update_layout(height=500, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # 4. 渲染历史记录面板
            with hist_container:
                for item in st.session_state.sig_history[:10]: # 仅显示最近10条
                    h_col = "#10b981" if "多" in item['type'] else "#ef4444"
                    st.markdown(f"""
                        <div class='history-card'>
                            <span style='color:#8b949e;'>[{item['time']}]</span> 
                            <b style='color:{h_col};'>{item['type']}</b><br/>
                            价格: <b>{item['price']}</b> | 量能: {item['vol']}
                        </div>
                        <div style='height:5px;'></div>
                    """, unsafe_allow_html=True)

        except Exception: pass

    tick()

if __name__ == "__main__": main()
