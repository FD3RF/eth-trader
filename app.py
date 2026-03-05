import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 工业级底层架构 (移除 http2 解决 ImportError)
# ==========================================
if 'http_client' not in st.session_state:
    st.session_state.http_client = httpx.Client(timeout=10.0)

if "last_voice" not in st.session_state:
    st.session_state.last_voice = ""

def voice_alert(text):
    if st.session_state.last_voice != text:
        components.html(f"""<script>
            var msg = new SpeechSynthesisUtterance('{text}');
            msg.rate = 1.1; window.speechSynthesis.speak(msg);
        </script>""", height=0)
        st.session_state.last_voice = text

# ==========================================
# 2. 巅峰实战视觉配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 | 信号修复版", page_icon="🏹")
st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #11141c; border: 1px solid #2d323e; padding: 15px; border-radius: 10px; }
    .status-card { background: #1a1c23; border-left: 8px solid #d4af37; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
    .plan-card { background: #11141c; border: 1px solid #ff4b4b; padding: 15px; border-radius: 10px; margin-top: 10px; border-left: 5px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 核心策略引擎 (带缩量审计与锚点逻辑)
# ==========================================
def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    
    # 缩量审计：前3根量能持续萎缩
    df['is_shrinking'] = (df['v'].shift(1) < df['v'].shift(2)) & (df['v'].shift(2) < df['v'].shift(3))
    
    # 信号判定：1.5倍放量 + 实体突破
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']
    
    # 局部锚点：30根K线回溯锁定
    window = df.tail(30) 
    local_down = window[window['c'] < window['o']].nlargest(1, 'v')
    local_up = window[window['c'] > window['o']].nlargest(1, 'v')
    
    anchors = {
        'upper': local_down['h'].values[0] if not local_down.empty else window['h'].max(),
        'lower': local_up['l'].values[0] if not local_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 4. 实时战报渲染 (解决黑屏与刷新)
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": st.session_state.symbol, "bar": "5m", "limit": "100"}
    
    try:
        resp = st.session_state.http_client.get(url, params=params)
        data = resp.json().get('data', [])
        if not data: return

        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
        df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df = df.sort_values('time').reset_index(drop=True)

        df, anchors = apply_warrior_logic(df, st.session_state.params)
        curr = df.iloc[-1]
        upper, lower = anchors['upper'], anchors['lower']
        
        # 1. 战报状态显示
        if curr['buy_sig'] and curr['c'] > upper:
            status, detail, color = "🚀 核心突破", "缩量审计通过，压力位已击穿，多头总攻！", "#26a69a"
            voice_alert("放量起涨，突破压力位，开多")
        elif curr['sell_sig'] or curr['c'] < lower:
            status, detail, color = "❄️ 趋势转弱", "跌破局部锚点支撑，空头占优。", "#ef5350"
            voice_alert("趋势转弱，注意回踩，准备做空")
        else:
            status, detail, color = "💎 震荡蓄势", f"当前量能 {curr['vol_ratio']:.2f}x，等待锚点突破。", "#1e90ff"

        st.markdown(f"<div class='status-card' style='border-left:8px solid {color};'><h1 style='color:{color};margin:0;'>{status}</h1><p style='color:#ccc;font-size:20px;'><b>逻辑分析：</b>{detail}</p></div>", unsafe_allow_html=True)

        # 2. 狙击手计划执行中 (强化保本提示)
        if status != "💎 震荡蓄势":
            is_buy = "突破" in status
            sl = lower if is_buy else upper
            risk = abs(curr['c'] - sl)
            be_level = curr['c'] + risk if is_buy else curr['c'] - risk 
            tp = curr['c'] + risk * st.session_state.params['rr_ratio'] if is_buy else curr['c'] - risk * st.session_state.params['rr_ratio']
            
            st.markdown(f"""<div class='plan-card'>
                <h3 style='color:#ff4b4b;margin-top:0;'>🎯 狙击手计划执行中</h3>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px;'>
                    <div><b>初始止损位:</b> ${sl:.2f}</div>
                    <div><b>保本触发位:</b> ${be_level:.2f}</div>
                    <div><b>目标止盈位:</b> ${tp:.2f}</div>
                </div>
                <p style='color:#ffcc00;margin-top:10px;font-weight:bold;'>⚠️ 强制铁律：价格触及 ${be_level:.2f} 后，请立即手动将止损上移至开仓位，锁定0风险！</p>
            </div>""", unsafe_allow_html=True)

        # 3. 核心指标
        c1, c2, c3 = st.columns(3)
        c1.metric("ETH 现价", f"${curr['c']:.2f}")
        c2.metric("当前量能比", f"{curr['vol_ratio']:.2f}x")
        c3.metric("刷新心跳", f"{datetime.now().strftime('%H:%M:%S')}")

        # 4. 图表渲染 (已修复缺失的红色三角)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        df_p = df.tail(80)
        fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c'], name="K线"), row=1, col=1)

        # 【核心修复】买卖信号标记
        buys = df_p[df_p['buy_sig']]
        sells = df_p[df_p['sell_sig']]
        
        # 绿色上三角 (做多)
        fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers',
                                 marker=dict(symbol='triangle-up', size=15, color='#00ffcc'), name='做多信号'), row=1, col=1)
        # 红色下三角 (做空) - 之前遗漏的部分
        fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers',
                                 marker=dict(symbol='triangle-down', size=15, color='#ff3333'), name='做空信号'), row=1, col=1)
        
        # 锚点线
        fig.add_hline(y=upper, line_dash="dot", line_color="#ef5350", annotation_text="压力:局部阴高", row=1, col=1)
        fig.add_hline(y=lower, line_dash="dot", line_color="#26a69a", annotation_text="支撑:局部阳低", row=1, col=1)

        v_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_p['c'], df_p['o'])]
        fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)

        fig.update_layout(height=700, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.error(f"连接出错: {e}")

# ==========================================
# 5. 控制中心
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V6.2 Sniper")
    with st.sidebar.expander("🏹 实战口诀锦囊", expanded=True):
        st.markdown("缩量是提醒，放量是信号。触及保本点即锁仓。")

    with st.sidebar.expander("策略校准", expanded=True):
        st.session_state.params = {
            "ma_len": st.number_input("均量周期", 5, 100, 10),
            "expand_p": st.slider("放量判定 (%)", 110, 500, 150),
            "body_r": st.slider("突破实体比", 0.05, 0.90, 0.20),
            "rr_ratio": st.slider("盈亏比 (1:X)", 1.0, 3.0, 1.5, step=0.1)
        }
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    dashboard_loop()

if __name__ == "__main__":
    main()
