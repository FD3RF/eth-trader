import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 极速渲染配置 [优化点：内存管理与缓存控制]
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior Sniper V6.2 Pro", page_icon="⚡")

# 初始化状态
if "voice_on" not in st.session_state: st.session_state.voice_on = False
if "last_sig_ts" not in st.session_state: st.session_state.last_sig_ts = 0

st.markdown("""
    <style>
    .stApp { background-color: #030508; }
    .top-panel { background: #0d1117; border: 1px solid #30363d; padding: 12px; border-radius: 10px; }
    .plan-card { background: #161b22; border-left: 5px solid #238636; padding: 10px; border-radius: 6px; }
    [data-testid="stMetricValue"] { font-size: 24px !important; color: #58a6ff; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 向量化计算引擎 [优化点：移除循环，使用 Numpy 加速]
# ==========================================
@st.cache_data(ttl=2) # 2秒超短缓存，防止重复重复计算相同时间戳数据
def fast_audit_engine(data_raw, p):
    df = pd.DataFrame(data_raw, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
    for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
    df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df = df.sort_values('time').reset_index(drop=True)

    # 均量计算（向量化）
    v_arr = df['v'].values
    ma_v = pd.Series(v_arr).rolling(p['ma_len']).mean().values
    vol_r = np.divide(v_arr, ma_v, out=np.zeros_like(v_arr), where=ma_v!=0)
    
    # 实体比率计算
    body_abs = np.abs(df['c'].values - df['o'].values)
    range_hl = (df['h'].values - df['l'].values) + 1e-9
    body_r = body_abs / range_hl

    # 寻找锚点 (最近30根K线)
    win = df.tail(30)
    up_idx = win[win['c'] > win['o']]['v'].idxmax() if not win[win['c'] > win['o']].empty else win.index[-1]
    dn_idx = win[win['c'] < win['o']]['v'].idxmax() if not win[win['c'] < win['o']].empty else win.index[-1]
    
    supp = df.at[up_idx, 'l']  # 支撑：放量阳线底
    press = df.at[dn_idx, 'h'] # 压力：放量阴线顶
    
    # 信号逻辑判定 [对齐口诀]
    curr_v_r = vol_r[-1]
    is_expand = curr_v_r > (p['exp'] / 100.0)
    is_shrink = curr_v_r < 0.65
    
    buy = is_expand and (df['c'].iloc[-1] > df['o'].iloc[-1]) and (df['c'].iloc[-1] > press)
    sell = is_expand and (df['c'].iloc[-1] < df['o'].iloc[-1]) and (df['c'].iloc[-1] < supp)
    
    return df, {'supp': supp, 'press': press, 'vol_r': curr_v_r, 'buy': buy, 'sell': sell, 'shrink': is_shrink}

# ==========================================
# 3. 实时执行控制台 [优化点：异步 IO 模拟]
# ==========================================
def main():
    with st.container():
        st.markdown("<div class='top-panel'>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1.5, 1.5])
        with c1:
            if st.button("⚡ 性能全开/授权" if not st.session_state.voice_on else "🛑 停止监听", type="primary", use_container_width=True):
                st.session_state.voice_on = not st.session_state.voice_on
                st.rerun()
        with c2: symbol = st.text_input("SYMBOL", "ETH-USDT-SWAP", label_visibility="collapsed")
        with c3: ma_len = st.number_input("MA", 5, 30, 10, label_visibility="collapsed")
        with c4: exp = st.slider("放量倍数%", 100, 300, 150, label_visibility="collapsed")
        with c5: body = st.slider("实体过滤", 0.1, 0.9, 0.2, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    banner_slot = st.empty(); plan_slot = st.empty(); chart_slot = st.empty(); voice_slot = st.empty()

    @st.fragment(run_every="2s") # 性能提升：从3秒缩短至2秒响应
    def engine_tick():
        try:
            # 极速数据获取
            with httpx.Client(http2=True, timeout=2.0) as client:
                r = client.get("https://www.okx.com/api/v5/market/candles", params={"instId": symbol, "bar": "5m", "limit": "80"})
                data = r.json()['data']
            
            df, res = fast_audit_engine(data, {"ma_len": ma_len, "exp": exp, "body": body})
            curr = df.iloc[-1]

            # --- 战报与语音 (100% 对应口诀) ---
            if res['buy']:
                msg, cl, voice = "🚀 放量起涨：突破开多", "#00ff88", "放量起涨，突破前高压力，直接开多"
            elif res['sell']:
                msg, cl, voice = "❄️ 放量下跌：破位开空", "#ff4444", "放量下跌，跌破前低支撑，直接开空"
            elif res['shrink']:
                msg, cl, voice = "💎 缩量回踩：只看不动", "#58a6ff", ""
            else:
                msg, cl, voice = "📊 震荡蓄势：等待信号", "#8b949e", ""

            # 渲染顶部战报
            banner_slot.markdown(f"""
                <div style='background:#0d1117; border-left:10px solid {cl}; padding:15px; border-radius:8px;'>
                    <h2 style='color:{cl};margin:0;'>{msg} | <span style='color:white;'>${curr['c']:.2f}</span></h2>
                </div>
            """, unsafe_allow_html=True)

            # 渲染进场交易计划 (口诀实战计划)
            with plan_slot.container():
                st.markdown("<div class='plan-card'>", unsafe_allow_html=True)
                p1, p2, p3, p4 = st.columns(4)
                if res['buy'] or res['sell']:
                    side = "做多" if res['buy'] else "做空"
                    sl = res['supp'] if res['buy'] else res['press']
                    tp = curr['c'] + (curr['c'] - sl)*2.0 if res['buy'] else curr['c'] - (sl - curr['c'])*2.0
                    p1.warning(f"🎯 动作：{side}")
                    p2.info(f"🛑 止损：{sl:.2f}")
                    p3.success(f"💰 止盈：{tp:.2f}")
                    p4.metric("量能爆发", f"{res['vol_r']:.2f}x")
                else:
                    p1.write("📝 状态：观察中")
                    p2.write(f"🔼 强压：{res['press']:.2f}")
                    p3.write(f"🔽 强支：{res['supp']:.2f}")
                    p4.metric("实时量能", f"{res['vol_r']:.2f}x")
                st.markdown("</div>", unsafe_allow_html=True)

            # 触发语音
            if voice and st.session_state.voice_on and st.session_state.last_sig_ts != curr['ts']:
                with voice_slot:
                    components.html(f"<script>var m=new SpeechSynthesisUtterance('{voice}'); m.lang='zh-CN'; m.rate=1.2; window.speechSynthesis.speak(m);</script>", height=0)
                st.session_state.last_sig_ts = curr['ts']

            # 图表渲染优化
            with chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
                df_v = df.tail(50)
                fig.add_trace(go.Candlestick(x=df_v['time'], open=df_v['o'], high=df_v['h'], low=df_v['l'], close=df_v['c']), row=1, col=1)
                # 支撑压力虚线
                fig.add_hline(y=res['press'], line_dash="dash", line_color="#ff4444", opacity=0.5, row=1, col=1)
                fig.add_hline(y=res['supp'], line_dash="dash", line_color="#00ff88", opacity=0.5, row=1, col=1)
                # 信号三角形
                if res['buy']: fig.add_trace(go.Scatter(x=[curr['time']], y=[curr['l']*0.999], mode='markers', marker=dict(symbol='triangle-up', size=18, color='#00ff88')), row=1, col=1)
                if res['sell']: fig.add_trace(go.Scatter(x=[curr['time']], y=[curr['h']*1.001], mode='markers', marker=dict(symbol='triangle-down', size=18, color='#ff4444')), row=1, col=1)
                
                fig.update_layout(height=550, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=5,b=0,l=10,r=40))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        except Exception as e: pass

    engine_tick()

if __name__ == "__main__":
    main()
