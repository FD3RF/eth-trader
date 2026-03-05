import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import streamlit.components.v1 as components

# ==========================================
# 1. 布局与样式
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    .sticky-header {
        position: sticky; top: 0; z-index: 100;
        background: rgba(13, 17, 23, 0.9); border: 1px solid #30363d;
        padding: 15px; border-radius: 10px; margin-bottom: 10px;
    }
    .history-card {
        background: #161b22; border-left: 5px solid #58a6ff;
        padding: 10px; margin-bottom: 5px; border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# 核心状态锁定
if "sig_history" not in st.session_state: st.session_state.sig_history = []
if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig_ts" not in st.session_state: st.session_state.last_sig_ts = ""

# ==========================================
# 2. 核心逻辑 (100% 保留多空三角)
# ==========================================
def warrior_logic(data, p):
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
    for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
    df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df = df.sort_values('time').reset_index(drop=True)
    
    ma_v = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / ma_v.replace(0, 1e-9)
    
    # 锚点锁定
    lookback = df.tail(30)
    v_dn = lookback[lookback['c'] < lookback['o']]
    v_up = lookback[lookback['c'] > lookback['o']]
    press = v_dn.nlargest(1, 'v')['h'].values[0] if not v_dn.empty else lookback['h'].max()
    supp = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else lookback['l'].min()

    # 三角信号
    df['buy_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] > df['o']) & (df['c'] > press)
    df['sell_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] < df['o']) & (df['c'] < supp)
    return df, {'press': press, 'supp': supp, 'vol_r': df['vol_r'].iloc[-1]}

# ==========================================
# 3. 页面渲染 (彻底修复重复标题问题)
# ==========================================
def main():
    with st.sidebar:
        st.header("⚔️ Sniper V6.2 Pro")
        st.session_state.voice_on = st.toggle("语音提醒", st.session_state.voice_on)
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量触发%", 100, 300, 150)
        sym = st.text_input("合约", "ETH-USDT-SWAP")

    # 布局占位
    report_area = st.empty()
    chart_area = st.empty()
    
    # --- 关键修改点：标题放在 tick() 外面，永远只显示一次 ---
    st.markdown("### 📜 信号复盘历史 (最近 10 条)") 
    history_list_area = st.container() # 信号卡片容器
    voice_slot = st.empty()

    @st.fragment(run_every="2s")
    def tick():
        try:
            with httpx.Client(http2=True, timeout=2.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": sym, "bar": "5m", "limit": "80"})
                df, res = warrior_logic(r.json()['data'], {"ma_len": ma_len, "exp": exp})
            
            curr = df.iloc[-1]
            
            # 1. 实时战报
            color = "#10b981" if curr['buy_tri'] else "#ef4444" if curr['sell_tri'] else "#3b82f6" if res['vol_r'] < 0.7 else "#9ca3af"
            status = "🚀 多头突袭" if curr['buy_tri'] else "❄️ 空头突袭" if curr['sell_tri'] else "💎 缩量震荡" if res['vol_r'] < 0.7 else "📊 震荡蓄势"
            report_area.markdown(f"<div class='sticky-header'><h1 style='color:{color};margin:0;'>{status} | ${curr['c']:.2f}</h1></div>", unsafe_allow_html=True)

            # 2. K线主图 (保留强化三角)
            with chart_area:
                fig = make_subplots(rows=1, cols=1)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']))
                b = df_p[df_p['buy_tri']]; s = df_p[df_p['sell_tri']]
                fig.add_trace(go.Scatter(x=b['time'], y=b['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=15, color='#10b981', line=dict(width=1, color='white'))))
                fig.add_trace(go.Scatter(x=s['time'], y=s['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=15, color='#ef4444', line=dict(width=1, color='white'))))
                fig.update_layout(height=450, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # 3. 历史记录逻辑
            if (curr['buy_tri'] or curr['sell_tri']) and st.session_state.last_sig_ts != curr['ts']:
                sig_type = "做多" if curr['buy_tri'] else "做空"
                st.session_state.sig_history.insert(0, {"t": curr['time'].strftime('%H:%M:%S'), "type": sig_type, "p": curr['c']})
                st.session_state.last_sig_ts = curr['ts']
                if st.session_state.voice_on:
                    v_cmd = "放量起涨" if curr['buy_tri'] else "放量下跌"
                    with voice_slot: components.html(f"<script>var m=new SpeechSynthesisUtterance('{v_cmd}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)

            # 4. 只刷新历史列表，不刷新标题
            with history_list_area:
                for item in st.session_state.sig_history[:10]:
                    h_col = "#10b981" if item['type'] == "做多" else "#ef4444"
                    st.markdown(f"<div class='history-card' style='border-left-color:{h_col};'>[{item['t']}] <b>{item['type']}</b> | 价格: {item['p']}</div>", unsafe_allow_html=True)

        except Exception: pass

    tick()

if __name__ == "__main__": main()
