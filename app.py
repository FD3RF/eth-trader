import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import time

# ==========================================
# 1. 视觉架构：暗黑战场风格
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V5.4 | 口诀实战", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #11141c; border: 1px solid #2d323e; padding: 15px; border-radius: 10px; }
    .status-card { background: #1a1c23; border-left: 5px solid #d4af37; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    .mnemonic-active { color: #d4af37; font-weight: bold; border: 1px solid #d4af37; padding: 2px 5px; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心引擎：量价扫描器
# ==========================================
class WarriorCore:
    def __init__(self):
        self.client = httpx.Client(timeout=5.0, http2=True)

    def fetch_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            params = {"instId": symbol, "bar": "5m", "limit": "100", "_t": time.time_ns()}
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except: return None

# ==========================================
# 3. 核心逻辑：口诀对号入座
# ==========================================
def analyze_mnemonics(df, p):
    curr, prev = df.iloc[-1], df.iloc[-2]
    ma_v = df['v'].rolling(p['ma_len']).mean().iloc[-1]
    
    # 基础状态定义
    is_shrink = curr['v'] < ma_v * (p['shrink_p']/100)  # 缩量
    is_expand = curr['v'] > ma_v * (p['expand_p']/100)  # 放量
    p_low = df['l'].iloc[-15:-1].min()  # 近期前低
    p_high = df['h'].iloc[-15:-1].max() # 近期前高
    
    msg = "💎 动能积蓄中..."
    action = "👀 只看不动"
    sig_type = None

    # --- 做多口诀判定 ---
    if is_shrink and curr['l'] >= p_low * 0.999:
        msg = "🟡 缩量回踩，低点不破"
        action = "🔔 准备动手 (等放量)"
    elif is_expand and curr['c'] > prev['h'] and curr['c'] > p_high:
        msg = "🔥 放量起涨，突破前高"
        action = "🚀 直接开多"
        sig_type = "LONG"
    elif is_expand and curr['l'] <= p_low and curr['c'] > p_low:
        msg = "⚡ 放量急跌，底部不破"
        action = "🎯 激进多单机会"
        sig_type = "LONG"

    # --- 做空口诀判定 ---
    elif is_shrink and curr['h'] <= p_high * 1.001:
        msg = "🔵 缩量反弹，高点不破"
        action = "🔔 准备动手 (等放量)"
    elif is_expand and curr['c'] < prev['l'] and curr['c'] < p_low:
        msg = "❄️ 放量下跌，跌破前低"
        action = "💀 直接开空"
        sig_type = "SHORT"
    elif is_expand and curr['h'] >= p_high and curr['c'] < p_high:
        msg = "⚠️ 放量急涨，顶部不破"
        action = "🎯 激进空单机会"
        sig_type = "SHORT"

    return {"msg": msg, "action": action, "sig": sig_type, "p_high": p_high, "p_low": p_low}

# ==========================================
# 4. UI 渲染：战报与图表
# ==========================================
@st.fragment(run_every="5s")
def render_v5_4():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    df = core.fetch_data(symbol)
    if df is None: return

    res = analyze_mnemonics(df, st.session_state.params)
    curr = df.iloc[-1]

    # 顶部指标
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ETH 现价", f"${curr['c']:.2f}")
    m2.metric("当前量能比", f"{curr['v']/(df['v'].tail(10).mean()):.2f}x")
    m3.metric("实时指令", res['action'])
    m4.metric("心跳刷新", f"{int(time.time()%60)}s")

    # 左右布局
    t1, t2 = st.columns([1, 3])

    with t1:
        st.markdown(f"### 📑 当前口诀对号")
        st.info(res['msg'])
        
        with st.container(border=True):
            st.markdown("**实战检查清单：**")
            st.checkbox("找点：前高/前低已标记", value=True)
            st.checkbox("看量：缩量准备/放量执行", value=True)
            st.checkbox("控仓：单笔风险 < 2%", value=True)
        
        st.divider()
        st.write(f"关键压力: **{res['p_high']:.2f}**")
        st.write(f"关键支撑: **{res['p_low']:.2f}**")

    with t2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"), row=1, col=1)
        
        # 绘制支撑压力线
        fig.add_hline(y=res['p_high'], line_color="#ef5350", line_dash="dash", opacity=0.5, row=1, col=1)
        fig.add_hline(y=res['p_low'], line_color="#26a69a", line_dash="dash", opacity=0.5, row=1, col=1)
        
        # 成交量
        v_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
        fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)

        fig.update_layout(height=700, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def main():
    st.sidebar.title("⚔️ Warrior V5.4 Control")
    with st.sidebar.expander("策略参数调节", expanded=True):
        ma_len = st.number_input("均量周期", 5, 60, 10)
        shrink_p = st.slider("缩量判定 (%)", 20, 100, 60)
        expand_p = st.slider("放量判定 (%)", 110, 300, 150)
        body_r = st.slider("突破实体比", 0.05, 0.50, 0.20)
    
    st.session_state.params = {"ma_len": ma_len, "shrink_p": shrink_p, "expand_p": expand_p, "body_r": body_r}
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    **核心总结：**
    - 做多：缩量不破底，放量突破买
    - 做空：缩量不过顶，放量跌破空
    - 核心：缩量是提醒，放量是信号
    """)
    
    render_v5_4()

if __name__ == "__main__":
    main()
