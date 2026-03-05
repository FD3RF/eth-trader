import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 内核驱动：高兼容性网络架构 (修复黑屏问题)
# ==========================================
if 'http_client' not in st.session_state:
    # 彻底禁用 http2 以确保在 Streamlit Cloud 环境稳定运行
    st.session_state.http_client = httpx.Client(timeout=15.0, follow_redirects=True)

if "last_voice" not in st.session_state:
    st.session_state.last_voice = ""

def voice_alert(text):
    if st.session_state.last_voice != text:
        components.html(f"""<script>
            var msg = new SpeechSynthesisUtterance('{text}');
            msg.rate = 1.2; window.speechSynthesis.speak(msg);
        </script>""", height=0)
        st.session_state.last_voice = text

# 视觉配置：强化深色实战风格
st.set_page_config(layout="wide", page_title="Warrior V6.2 Sniper", page_icon="⚔️")
st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #1f2937; padding: 15px; border-radius: 12px; }
    .status-card { background: #111827; border-left: 8px solid #fbbf24; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #1f2937; }
    .plan-card { background: #0f172a; border: 2px solid #ef4444; padding: 20px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 顶级策略引擎：30根K线动态审计
# ==========================================
def apply_warrior_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_size'] = abs(df['c'] - df['o'])
    df['total_size'] = (df['h'] - df['l']).replace(0, 0.001)
    df['body_ratio'] = df['body_size'] / df['total_size']
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    
    # 核心：放量脉冲判定
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    
    # 修复：全量对称信号识别
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    # 局部极值锚点回溯
    window = df.tail(30)
    v_max_down = window[window['c'] < window['o']]
    v_max_up = window[window['c'] > window['o']]
    
    anchors = {
        'upper': v_max_down.nlargest(1, 'v')['h'].values[0] if not v_max_down.empty else window['h'].max(),
        'lower': v_max_up.nlargest(1, 'v')['l'].values[0] if not v_max_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 3. 实时战场监控：10s 动态渲染
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
        
        # 1. 顶部战报逻辑展示
        if curr['buy_sig'] or curr['c'] > upper:
            status, detail, color = "🚀 多头总攻", "压力位已放量突破，主力意图向上。", "#10b981"
            voice_alert("放量突破，进场做多")
        elif curr['sell_sig'] or curr['c'] < lower:
            status, detail, color = "❄️ 空头突袭", "支撑位失守，空头动能正在释放。", "#ef4444"
            voice_alert("跌破支撑，空头信号亮起")
        else:
            status, detail, color = "💎 窄幅震荡", f"量能比 {curr['vol_ratio']:.2f}x，等待锚点区破位。", "#3b82f6"

        st.markdown(f"<div class='status-card' style='border-left:8px solid {color};'><h1 style='color:{color};margin:0;'>{status}</h1><p style='color:#ccc;font-size:18px;'><b>核心逻辑：</b>{detail}</p></div>", unsafe_allow_html=True)

        # 2. 狙击计划清单 (包含保本逻辑提示)
        if status != "💎 窄幅震荡":
            is_long = "多头" in status
            sl = lower if is_long else upper
            risk = abs(curr['c'] - sl)
            # 顶级纪律：盈利达到 1 倍风险时触发保本位
            be_trigger = curr['c'] + risk if is_long else curr['c'] - risk
            tp = curr['c'] + (risk * st.session_state.params['rr_ratio']) if is_long else curr['c'] - (risk * st.session_state.params['rr_ratio'])
            
            st.markdown(f"""<div class='plan-card'>
                <h3 style='color:#ef4444;margin-top:0;'>🎯 狙击作战计划 (方向: {'LONG' if is_long else 'SHORT'})</h3>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;'>
                    <div><b>开仓/现价:</b><br><span style='font-size:22px;'>${curr['c']:.2f}</span></div>
                    <div><b>初始止损:</b><br><span style='font-size:22px;'>${sl:.2f}</span></div>
                    <div><b>保本触发点:</b><br><span style='font-size:22px;color:#fbbf24;'>${be_trigger:.2f}</span></div>
                </div>
                <p style='color:#fbbf24;margin-top:15px;font-weight:bold;'>⚠️ 强制纪律：价格触及 ${be_trigger:.2f} 后，请立即将止损提至开仓位，锁定 0 风险！</p>
            </div>""", unsafe_allow_html=True)

        # 3. 核心指标看板
        c1, c2, c3 = st.columns(3)
        c1.metric("ETH 现价", f"${curr['c']:.2f}")
        c2.metric("当前量能比", f"{curr['vol_ratio']:.2f}x")
        c3.metric("实时心跳", f"{datetime.now().strftime('%H:%M:%S')}")

        # 4. 高级绘图引擎 (优化：60根K线解决拥挤)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        df_p = df.tail(60) 
        
        fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c'], name="K线"), row=1, col=1)

        # [全量修复] 多空双向信号标记
        buys = df_p[df_p['buy_sig']]
        sells = df_p[df_p['sell_sig']]
        fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers', marker=dict(symbol='triangle-up', size=18, color='#10b981'), name='B'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers', marker=dict(symbol='triangle-down', size=18, color='#ef4444'), name='S'), row=1, col=1)
        
        # 锚点线
        fig.add_hline(y=upper, line_dash="dash", line_color="#ef4444", annotation_text="压力锚点", row=1, col=1)
        fig.add_hline(y=lower, line_dash="dash", line_color="#10b981", annotation_text="支撑锚点", row=1, col=1)

        # 成交量
        v_colors = ['#10b981' if c >= o else '#ef4444' for c, o in zip(df_p['c'], df_p['o'])]
        fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.warning(f"🛡️ 自动防御重载中... {e}")

# ==========================================
# 4. 控制中枢
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior Sniper V6.2")
    with st.sidebar.expander("🏹 狙击核心校准", expanded=True):
        ma_len = st.number_input("均量周期", 5, 50, 10)
        expand_p = st.slider("放量判定 (%)", 100, 300, 150)
        body_r = st.slider("实体比率", 0.05, 0.90, 0.20)
        rr_ratio = st.slider("盈亏比 (1:X)", 1.0, 5.0, 1.5, step=0.1)
        st.session_state.params = {"ma_len": ma_len, "expand_p": expand_p, "body_r": body_r, "rr_ratio": rr_ratio}

    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    st.sidebar.divider()
    st.sidebar.success("🏹 指纹审计引擎已就绪")
    
    dashboard_loop()

if __name__ == "__main__":
    main()
