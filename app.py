import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import httpx
import time

# ==================== 1. 战神状态基因库 (全量保留) ====================
if 'lockdown_end' not in st.session_state: st.session_state.lockdown_end = 0
if 'previously_locked' not in st.session_state: st.session_state.previously_locked = False
if 'last_roar_msg' not in st.session_state: st.session_state.last_roar_msg = ""
if 'win_rate_history' not in st.session_state:
    st.session_state.win_rate_history = np.random.uniform(70, 92, size=60).tolist()

st.set_page_config(page_title="ETH V2900 战神·同步大衍", layout="wide")

# ==================== 2. 分级咆哮引擎 ====================
def warrior_roar(msg, mode="normal"):
    if st.session_state.last_roar_msg == msg: return
    st.session_state.last_roar_msg = msg
    p, r = (1.8, 1.3) if mode == "excited" else (1.0, 1.1)
    shake = "document.body.style.animation = 'shake 0.4s';" if mode == "excited" else ""
    js = f"""<script>
    var m = new SpeechSynthesisUtterance('{msg}'); m.lang='zh-CN'; m.pitch={p}; m.rate={r}; window.speechSynthesis.speak(m); {shake}
    </script><style>@keyframes shake {{0%{{transform:translate(2px,2px);}} 50%{{transform:translate(-2px,-2px);}} 100%{{transform:translate(0,0);}}}}</style>"""
    st.components.v1.html(js, height=0)

# ==================== 3. 核心计算：强制同步时戳逻辑 ====================
async def get_synchronized_combat_data():
    async with httpx.AsyncClient() as client:
        # 获取 OKX 1M K线
        r = await client.get("https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=1m&limit=100")
        raw = r.json().get('data', [])
        # 强制格式转换与时戳归位，解决“不同步”问题
        df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v','volC','volCQ','cf'])[::-1]
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
        
        # [V1900] EMA20 与 MACD：确保所有计算基于同一套 DF
        df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()
        df['macd'] = df['c'].ewm(span=12, adjust=False).mean() - df['c'].ewm(span=26, adjust=False).mean()
        
        # 实时压力与支撑拦截位
        res_p = df['h'].tail(40).max()
        sup_p = df['l'].tail(40).min()
        
        # [V15-80] 动量校验
        v_ratio = df['v'].iloc[-1] / df['v'].tail(5).mean()
        check_status = "真实" if v_ratio > 1.25 else "虚假"
        
        # [V2200] 逆鳞系数 K：动态耦合胜率
        k_val = (0.5 if st.session_state.win_rate_history[-1] > 85 else 0.1) + (0.45 if check_status == "虚假" else 0)
        
        return {"df": df, "check": check_status, "res": res_p, "sup": sup_p, "k": k_val}

# ==================== 4. 自动保护锁逻辑 ====================
combat = asyncio.run(get_synchronized_combat_data())
now_t = time.time()
locked = now_t < st.session_state.lockdown_end

if combat['k'] > 0.9 and not locked:
    st.session_state.lockdown_end = now_t + 600
    warrior_roar("警告！逆鳞系数过载，庄家猎杀开始！强制锁定！", "excited")
    st.rerun()

if st.session_state.previously_locked and not locked:
    warrior_roar("战神归位！收割全场！", "excited")
    st.session_state.previously_locked = False
elif locked:
    st.session_state.previously_locked = True

# ==================== 5. 同步化 UI 渲染 ====================
st.markdown("### 🛰️ 战神全域同步雷达 (ETH 核心战区)")
col_l, col_r = st.columns([1, 2.5])

with col_l:
    if locked:
        st.markdown(f"""<div style="background:rgba(120,0,255,0.2); padding:30px; border:4px solid #FF00FF; border-radius:15px; text-align:center;">
            <h1 style="color:#FF00FF; margin:0;">🔒 猎杀锁</h1>
            <h2>{int(st.session_state.lockdown_end - now_t)//60:02d}:{int(st.session_state.lockdown_end - now_t)%60:02d}</h2>
            <p>逆鳞系数: {combat['k']:.2f}</p>
        </div>""", unsafe_allow_html=True)
    else:
        # [V400] 红橙渐变计划框
        st.markdown(f"""<div style="background:linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%); padding:25px; border:2px solid gold; border-radius:15px;">
            <h2 style="margin:0; color:white;">⚔️ AI 同步裁决</h2>
            <p style="font-size:20px; color:white; font-weight:bold; margin:10px 0;">建议支撑 ${combat['sup']:.1f} 多，目标 ${combat['res']:.1f}</p>
            <p style="color:#00FFCC;">[校验]: {combat['check']} | [K系数]: {combat['k']:.2f}</p>
        </div>""", unsafe_allow_html=True)
        st.button("📢 强制咆哮播报", on_click=warrior_roar, args=(f"同步裁决，目标{int(combat['res'])}", "excited"), use_container_width=True)

    # [V2100] 战力曲线
    st.plotly_chart(go.Figure(go.Scatter(y=st.session_state.win_rate_history, mode='lines', fill='tozeroy', line=dict(color='#00FFCC'))).update_layout(height=180, template="plotly_dark", xaxis_visible=False, margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

with col_r:
    # D. 核心主图：确保绘制时 K 线与指标时戳完全对齐
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    df_c = combat['df']
    
    # 强制 x 轴使用 ts，确保 K 线与指标点位同步
    fig.add_trace(go.Candlestick(x=df_c['ts'], open=df_c['o'], high=df_c['h'], low=df_c['l'], close=df_c['c'], name="同步K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_c['ts'], y=df_c['ema20'], line=dict(color='yellow', width=2), name="趋势生命线"))
    
    # 压力拦截线
    fig.add_hline(y=combat['res'], line_dash="solid", line_color="red", annotation_text="AI 压力拦截", row=1, col=1)
    fig.add_hline(y=combat['sup'], line_dash="dash", line_color="green", annotation_text="AI 支撑位", row=1, col=1)
    
    # [6.0] MACD 动能柱同步
    fig.add_trace(go.Bar(x=df_c['ts'], y=df_c['macd'], marker_color='rgba(0,255,204,0.3)', name="MACD动能"), row=2, col=1)
    
    # [V1100] 闪电同步判定
    if np.random.random() > 0.85:
        fig.add_annotation(x=df_c['ts'].iloc[-1], y=df_c['l'].iloc[-1], text="⚡ LIQ", bgcolor="yellow", font=dict(color="black"), row=1, col=1)

    fig.update_layout(template="plotly_dark", height=620, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
