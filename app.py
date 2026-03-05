import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 初始化配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.2 Sniper", page_icon="🏹")

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
# 2. 核心狙击逻辑
# ==========================================
def apply_warrior_sniper_logic(df, p):
    df = df.dropna().reset_index(drop=True)
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)

    # 缩量审计
    df['is_shrinking'] = (df['v'].shift(1) < df['v'].shift(2)) & (df['v'].shift(2) < df['v'].shift(3))
    
    # 信号触发
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r']) & df['is_shrinking']

    # 局部锚点
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
        
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
        df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df = df.sort_values('time').reset_index(drop=True)
        
        df, anchors = apply_warrior_sniper_logic(df, st.session_state.params)
        curr = df.iloc[-1]
        
        # 状态判断与语音
        if curr['buy_sig'] and curr['c'] > anchors['upper']:
            status, color = "🚀 核心突破", "#26a69a"
            voice_alert("缩量完成，放量突破，准备做多")
        elif curr['sell_sig'] or curr['c'] < anchors['lower']:
            status, color = "❄️ 趋势转弱", "#ef5350"
            voice_alert("趋势转弱，注意离场或反手")
        else:
            status, color = "💎 震荡蓄势", "#1e90ff"

        st.markdown(f"<h1 style='color:{color}; text-align:center;'>{status}</h1>", unsafe_allow_html=True)
        
        # 4. 保本提示逻辑
        if "💎" not in status:
            is_buy = "突破" in status
            sl = anchors['lower'] if is_buy else anchors['upper']
            risk = abs(curr['c'] - sl)
            be_level = curr['c'] + risk if is_buy else curr['c'] - risk
            st.warning(f"⚠️ **移动止损提醒**：当前价格 ${curr['c']:.2f}。若价格触及 **${be_level:.2f}**，请立即保本！")

        # ==========================================
        # 5. 图表渲染 (已完美补回三角信号)
        # ==========================================
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        df_p = df.tail(60) # 画面显示最近60根
        
        # A. K线图
        fig.add_trace(go.Candlestick(x=df_p['time'], open=df_p['o'], high=df_p['h'], low=df_p['l'], close=df_p['c'], name="K线"), row=1, col=1)
        
        # B. 锚点线
        fig.add_hline(y=anchors['upper'], line_dash="dot", line_color="#ef5350", annotation_text="压力", row=1, col=1)
        fig.add_hline(y=anchors['lower'], line_dash="dot", line_color="#26a69a", annotation_text="支撑", row=1, col=1)
        
        # C. 【核心修复】补回买卖信号三角形标记
        # 筛选出需要标记信号的数据点
        buys = df_p[df_p['buy_sig']]
        sells = df_p[df_p['sell_sig']]
        
        # 添加做多信号 (绿色上三角，位于最低价下方)
        fig.add_trace(go.Scatter(
            x=buys['time'], 
            y=buys['l'] * 0.998, # 稍微偏下一点，避免重叠
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#00ffcc', line=dict(width=1, color='white')),
            name='做多信号'
        ), row=1, col=1)
        
        # 添加做空信号 (红色下三角，位于最高价上方)
        fig.add_trace(go.Scatter(
            x=sells['time'], 
            y=sells['h'] * 1.002, # 稍微偏上一点
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='#ff3333', line=dict(width=1, color='white')),
            name='做空信号'
        ), row=1, col=1)
        
        # D. 成交量图
        fig.add_trace(go.Bar(x=df_p['time'], y=df_p['v'], opacity=0.3, name="成交量"), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    except Exception as e:
        st.error(f"连接中断: {e}")

# ==========================================
# 6. 主程序
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V6.2 Sniper")
    st.session_state.symbol = st.sidebar.text_input("合约", "ETH-USDT-SWAP")
    st.session_state.params = {
        "ma_len": st.sidebar.number_input("均量周期", 5, 50, 10),
        "expand_p": st.sidebar.slider("放量比例 (%)", 110, 300, 150),
        "body_r": st.sidebar.slider("实体占比", 0.1, 0.9, 0.2),
        "rr_ratio": st.sidebar.slider("盈亏比", 1.0, 3.0, 1.5)
    }
    sync_dashboard()

if __name__ == "__main__":
    main()
