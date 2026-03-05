import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import streamlit.components.v1 as components

# ==========================================
# 1. 深度 UI 布局定制 (垂直流布局)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 Pro", page_icon="⚔️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; }
    /* 置顶战报样式 */
    .sticky-report {
        position: sticky; top: 0; z-index: 1000;
        background: rgba(13, 17, 23, 0.9); border: 1px solid #30363d;
        padding: 15px; border-radius: 10px; margin-bottom: 20px;
        backdrop-filter: blur(5px);
    }
    /* 底部历史记录卡片 */
    .footer-history {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px; margin-top: 10px;
    }
    [data-testid="stMetricValue"] { font-size: 22px !important; color: #58a6ff; }
    </style>
""", unsafe_allow_html=True)

if "sig_history" not in st.session_state: st.session_state.sig_history = []
if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig_ts" not in st.session_state: st.session_state.last_sig_ts = ""

# ==========================================
# 2. 核心计算引擎 (保留所有锚点逻辑)
# ==========================================
def warrior_engine(data, p):
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
    for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
    df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df = df.sort_values('time').reset_index(drop=True)
    
    # 均量判定
    ma_v = df['v'].rolling(p['ma_len']).mean()
    df['vol_r'] = df['v'] / ma_v.replace(0, 1e-9)
    
    # 信号锚点
    lookback = df.tail(30)
    v_dn = lookback[lookback['c'] < lookback['o']]
    v_up = lookback[lookback['c'] > lookback['o']]
    press = v_dn.nlargest(1, 'v')['h'].values[0] if not v_dn.empty else lookback['h'].max()
    supp = v_up.nlargest(1, 'v')['l'].values[0] if not v_up.empty else lookback['l'].min()

    # 信号捕捉 (不删减逻辑)
    df['buy_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] > df['o']) & (df['c'] > press)
    df['sell_tri'] = (df['vol_r'] > (p['exp']/100)) & (df['c'] < df['o']) & (df['c'] < supp)
    
    return df, {'press': press, 'supp': supp, 'vol_r': df['vol_r'].iloc[-1]}

# ==========================================
# 3. 界面渲染 (战报置顶，历史在底)
# ==========================================
def main():
    with st.sidebar:
        st.header("⚔️ Sniper V6.2 Pro")
        if st.button("🔔 启动监听" if not st.session_state.voice_on else "🛑 停止监听", type="primary", use_container_width=True):
            st.session_state.voice_on = not st.session_state.voice_on
            st.rerun()
        ma_len = st.number_input("均量周期", 5, 60, 10)
        exp = st.slider("放量触发%", 100, 300, 150)
        sym = st.text_input("合约代码", "ETH-USDT-SWAP")
        if st.button("重置历史数据"): st.session_state.sig_history = []; st.rerun()

    # 布局槽位
    top_report = st.empty()
    mid_chart = st.container()
    bot_history = st.container()
    voice_slot = st.empty()

    @st.fragment(run_every="2s")
    def tick():
        try:
            with httpx.Client(http2=True, timeout=2.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": sym, "bar": "5m", "limit": "80"})
                df, res = warrior_engine(r.json()['data'], {"ma_len": ma_len, "exp": exp})
            
            curr = df.iloc[-1]
            
            # --- 1. 顶部：置顶实时战报 ---
            color = "#10b981" if curr['buy_tri'] else "#ef4444" if curr['sell_tri'] else "#9ca3af"
            status = "🚀 多头突袭" if curr['buy_tri'] else "❄️ 空头突袭" if curr['sell_tri'] else "📊 震荡蓄势"
            top_report.markdown(f"""
                <div class='sticky-report'>
                    <h1 style='color:{color}; margin:0;'>{status} | ${curr['c']:.2f}</h1>
                    <p style='color:#8b949e; margin:5px 0 0 0;'>
                        实时量能: <b>{res['vol_r']:.2f}x</b> | 支撑: {res['supp']} | 压力: {res['press']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # --- 2. 中间：K 线与指标 ---
            with mid_chart:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
                df_p = df.tail(60)
                fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c']), row=1, col=1)
                
                # 多空三角信号 (核心标记不删减)
                b = df_p[df_p['buy_tri']]; s = df_p[df_p['sell_tri']]
                fig.add_trace(go.Scatter(x=b['time'], y=b['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=16, color='#10b981', line=dict(width=1, color='white'))), row=1, col=1)
                fig.add_trace(go.Scatter(x=s['time'], y=s['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=16, color='#ef4444', line=dict(width=1, color='white'))), row=1, col=1)
                
                fig.update_layout(height=500, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # --- 3. 底部：信号历史记录 (新增) ---
            if (curr['buy_tri'] or curr['sell_tri']) and st.session_state.last_sig_ts != curr['ts']:
                sig_type = "做多" if curr['buy_tri'] else "做空"
                st.session_state.sig_history.insert(0, {"t": curr['time'].strftime('%H:%M:%S'), "type": sig_type, "p": curr['c'], "v": f"{res['vol_r']:.2f}x"})
                st.session_state.last_sig_ts = curr['ts']
                # 语音逻辑
                if st.session_state.voice_on:
                    v_msg = "放量起涨，突破压力，直接开多" if curr['buy_tri'] else "放量下跌，跌破支撑，直接开空"
                    with voice_slot: components.html(f"<script>var m=new SpeechSynthesisUtterance('{v_msg}'); m.lang='zh-CN'; window.speechSynthesis.speak(m);</script>", height=0)

            with bot_history:
                st.markdown("### 📜 信号复盘历史")
                if not st.session_state.sig_history:
                    st.info("等待第一个信号触发...")
                else:
                    # 采用横向分栏显示最近的信号历史记录
                    cols = st.columns(4)
                    for idx, item in enumerate(st.session_state.sig_history[:12]):
                        with cols[idx % 4]:
                            h_color = "#10b981" if item['type'] == "做多" else "#ef4444"
                            st.markdown(f"""
                                <div class='footer-history' style='border-top: 3px solid {h_color};'>
                                    <span style='color:#8b949e;'>[{item['t']}]</span> <b style='color:{h_color};'>{item['type']}</b><br/>
                                    价格: <b>{item['p']}</b><br/>量能: {item['v']}
                                </div>
                            """, unsafe_allow_html=True)

        except Exception: pass

    tick()

if __name__ == "__main__": main()
