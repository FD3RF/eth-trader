import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 内核驱动：高并发异步兼容架构
# ==========================================
if 'http_client' not in st.session_state:
    # 彻底移除 http2 参数，确保在所有 Streamlit 云端节点无错启动
    st.session_state.http_client = httpx.Client(timeout=10.0, follow_redirects=True)

if "last_voice" not in st.session_state:
    st.session_state.last_voice = ""

def voice_alert(text):
    if st.session_state.last_voice != text:
        components.html(f"""<script>
            var msg = new SpeechSynthesisUtterance('{text}');
            msg.rate = 1.2; window.speechSynthesis.speak(msg);
        </script>""", height=0)
        st.session_state.last_voice = text

# 顶级视觉注入
st.set_page_config(layout="wide", page_title="Warrior V6.2 Sniper", page_icon="⚔️")
st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #1f2937; padding: 15px; border-radius: 12px; }
    .status-card { background: #111827; border-left: 8px solid #fbbf24; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #1f2937; }
    .plan-card { background: #0f172a; border: 2px solid #ef4444; padding: 20px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2); }
    .stMetric label { color: #9ca3af !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 顶级策略引擎：全量审计逻辑
# ==========================================
def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    # 指标计算：均量、实体比、放量比
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_size'] = abs(df['c'] - df['o'])
    df['total_size'] = (df['h'] - df['l']).replace(0, 0.001)
    df['body_ratio'] = df['body_size'] / df['total_size']
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    
    # 缩量审计：前两根 K 线量能必须健康，防止过度放量后的假突破
    df['is_shrinking'] = (df['v'].shift(1) < df['v'].shift(2) * 1.2) 
    
    # 放量判定：爆发力度
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    
    # 核心：多空信号全对称审计
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']
    
    # 局部锚点回溯：30根K线动态支撑压力锁定
    window = df.tail(30) 
    v_max_down = window[window['c'] < window['o']]
    v_max_up = window[window['c'] > window['o']]
    
    anchors = {
        'upper': v_max_down.nlargest(1, 'v')['h'].values[0] if not v_max_down.empty else window['h'].max(),
        'lower': v_max_up.nlargest(1, 'v')['l'].values[0] if not v_max_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 3. 实时战场监控：10s 局部精准刷新
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": st.session_state.symbol, "bar": "5m", "limit": "100"}
    
    try:
        resp = st.session_state.http_client.get(url, params=params)
        if resp.status_code != 200: return
            
        data = resp.json().get('data', [])
        if not data: return

        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
        df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df = df.sort_values('time').reset_index(drop=True)

        df, anchors = apply_warrior_logic(df, st.session_state.params)
        curr = df.iloc[-1]
        upper, lower = anchors['upper'], anchors['lower']
        
        # 1. 战报分析与语音指引
        if curr['buy_sig'] or (curr['c'] > upper and curr['vol_ratio'] > 1.2):
            status, detail, color = "🚀 多头总攻", "压力锚点已被强力击穿，量价指纹吻合，多头动能爆发！", "#10b981"
            voice_alert("放量突破压力位，进场做多")
        elif curr['sell_sig'] or (curr['c'] < lower and curr['vol_ratio'] > 1.2):
            status, detail, color = "❄️ 空头突袭", "支撑锚点失守，空头放量杀跌，趋势转弱，注意撤退。", "#ef4444"
            voice_alert("跌破支撑，空头信号亮起")
        else:
            status, detail, color = "💎 窄幅震荡", f"量能比 {curr['vol_ratio']:.2f}x，处于缩量蓄势区间，静待方向确认。", "#3b82f6"

        st.markdown(f"<div class='status-card' style='border-left:8px solid {color};'><h1 style='color:{color};margin:0;'>{status}</h1><p style='color:#9ca3af;font-size:18px;'><b>深度逻辑：</b>{detail}</p></div>", unsafe_allow_html=True)

        # 2. 顶级计划执行中心：保本位强制提醒
        if status != "💎 窄幅震荡":
            is_long = "多头" in status
            sl = lower if is_long else upper
            risk = abs(curr['c'] - sl)
            # 顶级铁律：当盈利达到 1 倍风险时，触发保本位提损
            be_trigger = curr['c'] + risk if is_long else curr['c'] - risk
            tp = curr['c'] + (risk * st.session_state.params['rr_ratio']) if is_long else curr['c'] - (risk * st.session_state.params['rr_ratio'])
            
            st.markdown(f"""<div class='plan-card'>
                <h3 style='color:#ef4444;margin-top:0;'>🎯 狙击手计划执行中 (方向: {'LONG' if is_long else 'SHORT'})</h3>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;'>
                    <div><b>开仓/现价:</b><br><span style='font-size:22px;font-weight:bold;'>${curr['c']:.2f}</span></div>
                    <div><b>止损价:</b><br><span style='font-size:22px;'>${sl:.2f}</span></div>
                    <div><b>保本提损点:</b><br><span style='font-size:22px;color:#fbbf24;'>${be_trigger:.2f}</span></div>
                </div>
                <p style='color:#fbbf24;margin-top:15px;font-weight:bold;'>⚠️ 强制纪律：价格一旦触及 ${be_trigger:.2f}，立即将止损移动至开仓位，确保 0 风险持仓！</p>
            </div>""", unsafe_allow_html=True)

        # 3. 核心指标看板
        c1, c2, c3 = st.columns(3)
        c1.metric("ETH 现价", f"${curr['c']:.2f}")
        c2.metric("当前量能比", f"{curr['vol_ratio']:.2f}x")
        c3.metric("实时心跳", f"{datetime.now().strftime('%H:%M:%S')}")

        # 4. 高性能绘图引擎：解决拥挤与信号缺失
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        df_p = df.tail(60) # 固定 60 根 K 线，保证手机端与 PC 端画面清晰度
        
        fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c'], name="K线"), row=1, col=1)

        # 修复做多/做空三角标记
        buys = df_p[df_p['buy_sig']]
        sells = df_p[df_p['sell_sig']]
        fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.997, mode='markers', marker=dict(symbol='triangle-up', size=20, color='#10b981'), name='B'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.003, mode='markers', marker=dict(symbol='triangle-down', size=20, color='#ef4444'), name='S'), row=1, col=1)
        
        # 支撑压力锚点线
        fig.add_hline(y=upper, line_dash="dash", line_color="#ef4444", annotation_text="压力锚点", row=1, col=1)
        fig.add_hline(y=lower, line_dash="dash", line_color="#10b981", annotation_text="支撑锚点", row=1, col=1)

        # 成交量
        v_colors = ['#10b981' if c >= o else '#ef4444' for c, o in zip(df_p['c'], df_p['o'])]
        fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.warning(f"🛡️ 自动防御重连中... {e}")

# ==========================================
# 4. 指挥中枢
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior Sniper V6.2")
    with st.sidebar.expander("🏹 狙击核心校准", expanded=True):
        ma_len = st.number_input("均量周期", 5, 50, 10)
        expand_p = st.slider("放量判定 (%)", 100, 300, 150)
        body_r = st.slider("实体比率", 0.1, 0.9, 0.2)
        rr_ratio = st.slider("目标盈亏比 (1:X)", 1.0, 5.0, 1.5, step=0.1)
        st.session_state.params = {"ma_len": ma_len, "expand_p": expand_p, "body_r": body_r, "rr_ratio": rr_ratio}

    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    st.sidebar.divider()
    st.sidebar.info("🏹 指纹引擎已就绪\n实时审计缩量蓄势与放量脉冲。")
    
    dashboard_loop()

if __name__ == "__main__":
    main()
