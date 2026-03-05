import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 布局与样式
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2", page_icon="⚔️")

if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig" not in st.session_state: st.session_state.last_sig = ""

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    #MainMenu, footer, header {visibility: hidden;}
    .top-panel { background: #111827; border: 1px solid #1f2937; padding: 15px; border-radius: 12px; margin-bottom: 10px; }
    .plan-card { background: #1a1c24; border-left: 5px solid #3b82f6; padding: 12px; border-radius: 8px; margin-top: 5px; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #1f2937; padding: 8px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 自动定位支撑压力 (基于量能锚点)
# ==========================================
def get_warrior_levels(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    
    # 核心：寻找最近30根5分钟K线的量能锚点
    lookback = df.tail(30)
    # 压力位：量能最大的阴线最高点
    v_down = lookback[lookback['c'] < lookback['o']]
    press = v_down.nlargest(1, 'v')['h'].values[0] if not v_down.empty else lookback['h'].max()
    # 支撑位：量能最大的阳线最低点
    v_up = lookback[lookback['c'] > lookback['o']]
    supp = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else lookback['l'].min()
    
    # 信号判定
    last = df.iloc[-1]
    is_expand = last['vol_r'] > (p['exp'] / 100.0)
    buy = is_expand and (last['c'] > last['o']) and (last['c'] > press)
    sell = is_expand and (last['c'] < last['o']) and (last['c'] < supp)
    
    return df, {'press': press, 'supp': supp, 'buy': buy, 'sell': sell}

# ==========================================
# 3. 渲染主引擎
# ==========================================
def main():
    with st.container():
        st.markdown("<div class='top-panel'>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1.5, 1.5])
        with c1:
            if st.button("🔔 启动授权" if not st.session_state.voice_on else "🔇 关闭语音", type="primary", use_container_width=True):
                st.session_state.voice_on = not st.session_state.voice_on
                st.rerun()
        with c2: sym = st.text_input("代码", "ETH-USDT-SWAP", label_visibility="collapsed")
        with c3: ma_len = st.number_input("均量", 5, 50, 10, label_visibility="collapsed")
        with c4: exp = st.slider("放量触发%", 100, 300, 150, label_visibility="collapsed")
        with c5: body = st.slider("实体比", 0.1, 0.9, 0.2, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    # 容器定义
    banner_slot = st.empty(); plan_slot = st.empty(); metric_slot = st.empty(); chart_slot = st.empty(); voice_slot = st.empty()

    if 'http' not in st.session_state: st.session_state.http = httpx.Client(timeout=5.0)

    @st.fragment(run_every="3s")
    def run():
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            r = st.session_state.http.get(url, params={"instId": sym, "bar": "5m", "limit": "100"})
            df = pd.DataFrame(r.json()['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
            df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df = df.sort_values('time').reset_index(drop=True)
            
            df, res = get_warrior_levels(df, {"ma_len": ma_len, "exp": exp})
            curr = df.iloc[-1]

            # 1. 战报与语音 (口诀同步)
            if res['buy']:
                msg, cl, voice = "🚀 放量起涨：突破开多", "#10b981", "放量起涨，突破前高压力，直接开多"
            elif res['sell']:
                msg, cl, voice = "❄️ 放量下跌：破位开空", "#ef4444", "放量下跌，跌破前低支撑，直接开空"
            elif curr['vol_r'] < 0.7:
                msg, cl, voice = "💎 缩量回踩：只看不动", "#3b82f6", ""
            else:
                msg, cl, voice = "📊 震荡蓄势：等待信号", "#9ca3af", ""

            banner_slot.markdown(f"<div style='background:#111827; border-left:8px solid {cl}; padding:12px; border-radius:8px;'><h2 style='color:{cl};margin:0;'>{msg} | ${curr['c']:.2f}</h2></div>", unsafe_allow_html=True)

            # 2. 进场交易计划面板
            with plan_slot.container():
                st.markdown("<div class='plan-card'>", unsafe_allow_html=True)
                p1, p2, p3 = st.columns(3)
                if res['buy'] or res['sell']:
                    side = "做多" if res['buy'] else "做空"
                    sl = res['supp'] if res['buy'] else res['press']
                    tp = curr['c'] + (curr['c'] - sl)*1.5 if res['buy'] else curr['c'] - (sl - curr['c'])*1.5
                    p1.write(f"🎯 **当前计划：{side}**")
                    p2.write(f"🛑 **止损位：{sl:.2f}**")
                    p3.write(f"💰 **止盈位：{tp:.2f}**")
                else:
                    p1.write("📝 **交易状态：观察中**")
                    p2.write(f"🔼 压力位：{res['press']:.2f}")
                    p3.write(f"🔽 支撑位：{res['supp']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            # 3. 语音播报
            if voice and st.session_state.voice_on and st.session_state.last_sig != f"{msg}_{curr['ts']}":
                with voice_slot:
                    components.html(f"<script>var m=new SpeechSynthesisUtterance('{voice}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)
                st.session_state.last_sig = f"{msg}_{curr['ts']}"

            # 4. K线图 (自动画支撑压力线)
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                # 绘制支撑压力横线
                fig.add_hline(y=res['press'], line_dash="dash", line_color="#ef4444", annotation_text="压力线", row=1, col=1)
                fig.add_hline(y=res['supp'], line_dash="dash", line_color="#10b981", annotation_text="支撑线", row=1, col=1)
                # 信号标记
                buys = df_p[df_p['vol_r'] > (exp/100.0)][df_p['c'] > df_p['o']][df_p['c'] > res['press']]
                sells = df_p[df_p['vol_r'] > (exp/100.0)][df_p['c'] < df_p['o']][df_p['c'] < res['supp']]
                fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=15, color='#10b981')), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=15, color='#ef4444')), row=1, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=10,r=50))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception: pass

    run()

if __name__ == "__main__": main()
