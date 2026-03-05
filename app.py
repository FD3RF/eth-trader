import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 初始化配置 (防止黑屏的关键)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 Sniper", page_icon="🏹")

# 使用 Session State 保持 HTTP 客户端，避免重复创建连接
if 'http_client' not in st.session_state:
    st.session_state.http_client = httpx.Client(timeout=10.0, http2=True)

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
# 2. 整合后的核心逻辑函数
# ==========================================
def apply_warrior_sniper_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['vol_ratio'] = df['v'] / df['ma_v'].replace(0, 1e-9)
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)

    # --- 优化1：缩量观察条件 (前3根K线成交量递减) ---
    df['is_shrinking'] = (df['v'].shift(1) < df['v'].shift(2)) & (df['v'].shift(2) < df['v'].shift(3))
    
    # 信号触发：放量突破 + 前置缩量确认
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']

    # 局部锚点锁定
    window = df.tail(30)
    local_down = window[window['c'] < window['o']].nlargest(1, 'v')
    local_up = window[window['c'] > window['o']].nlargest(1, 'v')
    
    anchors = {
        'upper': local_down['h'].values[0] if not local_down.empty else window['h'].max(),
        'lower': local_up['l'].values[0] if not local_up.empty else window['l'].min()
    }
    return df, anchors

# ==========================================
# 3. 实时局部刷新引擎
# ==========================================
@st.fragment(run_every="10s")
def sync_dashboard():
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": st.session_state.symbol, "bar": "5m", "limit": "100"}
    
    try:
        resp = st.session_state.http_client.get(url, params=params)
        data = resp.json().get('data', [])
        if not data: return
        
        # 数据处理
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
        df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df = df.sort_values('time').reset_index(drop=True)
        
        # 调用新逻辑函数
        df, anchors = apply_warrior_sniper_logic(df, st.session_state.params)
        curr = df.iloc[-1]
        
        # 4. 动态状态与语音告警
        if curr['buy_sig'] and curr['c'] > anchors['upper']:
            status, color = "🚀 核心突破", "#26a69a"
            voice_alert("缩量完成，放量起涨，准备做多")
        elif curr['sell_sig'] or curr['c'] < anchors['lower']:
            status, color = "❄️ 趋势转弱", "#ef5350"
            voice_alert("缩量反弹结束，趋势转弱，准备做空")
        else:
            status, color = "💎 震荡蓄势", "#1e90ff"

        st.markdown(f"<h1 style='color:{color}; text-align:center;'>{status}</h1>", unsafe_allow_html=True)
        
        # 5. 渲染进场计划 (包含保本位)
        if "💎" not in status:
            is_buy = "突破" in status
            entry_p = curr['c']
            sl = anchors['lower'] if is_buy else anchors['upper']
            risk = abs(entry_p - sl)
            be_level = entry_p + risk if is_buy else entry_p - risk
            tp = entry_p + risk * st.session_state.params['rr_ratio'] if is_buy else entry_p - risk * st.session_state.params['rr_ratio']
            
            st.markdown(f"""<div style='background:#11141c; border:1px solid #444; padding:15px; border-radius:10px;'>
                <h3 style='color:{color}; margin:0;'>🎯 狙击手计划执行中</h3>
                <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:10px;'>
                    <div><b>开仓参考:</b> ${entry_p:.2f}</div>
                    <div><b>初始止损:</b> ${sl:.2f}</div>
                    <div><b>最终止盈:</b> ${tp:.2f}</div>
                </div>
                <p style='color:#ffa500; margin-top:10px; font-weight:bold;'>⚠️ 保本提示：价格触及 ${be_level:.2f} 后，请立即将止损移至 ${entry_p:.2f} 锁定风险！</p>
            </div>""", unsafe_allow_html=True)

        # 6. 图表渲染
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        df_p = df.tail(60)
        fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c'], name="K线"), row=1, col=1)
        fig.add_hline(y=anchors['upper'], line_dash="dot", line_color="#ef5350", annotation_text="压力锚点", row=1, col=1)
        fig.add_hline(y=anchors['lower'], line_dash="dot", line_color="#26a69a", annotation_text="支撑锚点", row=1, col=1)
        fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], marker_color=np.where(df_p['c']>df_p['o'], '#26a69a', '#ef5350'), opacity=0.3), row=2, col=1)
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    except Exception as e:
        st.error(f"连接中断: {e}")

# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V6.2 Sniper")
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    with st.sidebar.expander("狙击参数", expanded=True):
        ma = st.number_input("均量周期", 5, 50, 10)
        exp = st.slider("放量比例 (%)", 110, 300, 150)
        body = st.slider("实体占比", 0.1, 0.9, 0.2)
        rr = st.slider("盈亏比设置", 1.0, 3.0, 1.5)
    
    st.session_state.params = {"ma_len": ma, "expand_p": exp, "body_r": body, "rr_ratio": rr}
    
    sync_dashboard()

if __name__ == "__main__":
    main()
