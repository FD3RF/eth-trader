import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import streamlit.components.v1 as components

# ==========================================
# 1. 深度布局与 CSS 样式
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    /* 顶部置顶战报样式 */
    .sticky-header {
        position: sticky; top: 0; z-index: 100;
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 15px; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    /* 底部历史记录卡片样式 */
    .history-card {
        background: #161b22; border-radius: 8px;
        padding: 12px; margin-bottom: 8px;
        border-left: 5px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

if "sig_history" not in st.session_state: st.session_state.sig_history = []
if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig_ts" not in st.session_state: st.session_state.last_sig_ts = ""

# ==========================================
# 2. 核心计算引擎 (不删减逻辑)
# ==========================================
def warrior_engine(data, p):
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
    for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
    df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df = df.sort_values('time').reset_index(drop=True)
    
    ma_v = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / ma_v.replace(0, 1e-9)
    
    # 支撑压力锚点锁定
    lookback = df.tail(30)
    v_dn = lookback[lookback['c'] < lookback['o']]
    v_up = lookback[lookback['c'] > lookback['o']]
    press = v_dn.nlargest(1, 'v')['h'].values[0] if not v_dn.empty else lookback['h'].max()
    supp = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else lookback['l'].min()

    # 三角信号捕捉
    df['buy_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] > df['o']) & (df['c'] > press)
    df['sell_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] < df['o']) & (df['c'] < supp)
    return df, {'press': press, 'supp': supp, 'vol_r': df['vol_r'].iloc[-1]}

# ==========================================
# 3. 页面渲染 (战报置顶，历史在底)
# ==========================================
def main():
    with st.sidebar:
        st.header("⚔️ Sniper V6.2 Pro")
        st.session_state.voice_on = st.toggle("开启语音咆哮提醒", st.session_state.voice_on)
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量触发%", 100, 300, 150)
        sym = st.text_input("合约代码", "ETH-USDT-SWAP")
        if st.button("清除信号历史"): st.session_state.sig_history = []; st.rerun()

    # 布局槽位定义
    report_slot = st.empty()
    chart_slot = st.empty()
    
    # --- 标题修正：放在 tick 循环之外，解决刷屏标题问题 ---
    st.markdown("### 📜 信号复盘历史 (最近 10 条)")
    history_container = st.container() 
    voice_slot = st.empty()

    @st.fragment(run_every="2s")
    def tick():
        try:
            with httpx.Client(http2=True, timeout=2.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": sym, "bar": "5m", "limit": "80"})
                df, res = warrior_engine(r.json()['data'], {"ma_len": ma_len, "exp": exp})
            
            curr = df.iloc[-1]
            
            # --- 1. 顶部：置顶实时战报 ---
            color = "#10b981" if curr['buy_tri'] else "#ef4444" if curr['sell_tri'] else "#3b82f6" if res['vol_r'] < 0.7 else "#9ca3af"
            status = "🚀 多头突袭" if curr['buy_tri'] else "❄️ 空头突袭" if curr['sell_tri'] else "💎 缩量震荡" if res['vol_r'] < 0.7 else "📊 震荡蓄势"
            
            report_slot.markdown(f"""
                <div class='sticky-header'>
                    <h1 style='color:{color}; margin:0;'>{status} | ${curr['c']:.2f}</h1>
                    <div style='display:flex; gap:30px; margin-top:10px; color:#8b949e;'>
                        <span>实时量能: <b style='color:white;'>{res['vol_r']:.2f}x</b></span>
                        <span>支撑锚点: <b style='color:#10b981;'>{res['supp']}</b></span>
                        <span>压力锚点: <b style='color:#ef4444;'>{res['press']}</b></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- 2. 中部：K 线主图 (强化多空三角) ---
            with chart_slot:
                fig = make_subplots(rows=1, cols=1)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']))
                
                # 绘制信号三角 (绝不删减)
                b = df_p[df_p['buy_tri']]; s = df_p[df_p['sell_tri']]
                fig.add_trace(go.Scatter(x=b['time'], y=b['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=15, color='#10b981', line=dict(width=1, color='white'))))
                fig.add_trace(go.Scatter(x=s['time'], y=s['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=15, color='#ef4444', line=dict(width=1, color='white'))))
                
                fig.update_layout(height=480, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # --- 3. 底部：历史记录逻辑 ---
            if (curr['buy_tri'] or curr['sell_tri']) and st.session_state.last_sig_ts != curr['ts']:
                sig_txt = "🚀 做多突袭" if curr['buy_tri'] else "❄️ 做空突袭"
                st.session_state.sig_history.insert(0, {"t": curr['time'].strftime('%H:%M:%S'), "type": sig_txt, "p": curr['c'], "v": f"{res['vol_r']:.2f}x"})
                st.session_state.last_sig_ts = curr['ts']
                if st.session_state.voice_on:
                    v_msg = "放量起涨，直接开多" if curr['buy_tri'] else "放量下跌，直接开空"
                    with voice_slot: components.html(f"<script>var m=new SpeechSynthesisUtterance('{v_msg}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)

            # 只刷新容器内的卡片内容
            with history_container:
                for item in st.session_state.sig_history[:10]:
                    h_color = "#10b981" if "多" in item['type'] else "#ef4444"
                    st.markdown(f"""
                        <div class='history-card' style='border-left-color: {h_color};'>
                            <span style='color:#8b949e;'>[{item['t']}]</span> 
                            <b style='color:{h_color};'>{item['type']}</b> | 
                            成交价: <b>{item['p']}</b> | 量能: {item['v']}
                        </div>
                    """, unsafe_allow_html=True)

        except Exception: pass

    tick()

if __name__ == "__main__": main()
