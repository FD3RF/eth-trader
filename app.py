import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import time
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. 初始化配置与样式
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V5.7 | 大衍终极版", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #11141c; border: 1px solid #2d323e; padding: 15px; border-radius: 10px; }
    .stAlert { background-color: #1a1c23; border: 1px solid #d4af37; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 语音咆哮组件
# ==========================================
def voice_alert(text):
    """口诀形成肌肉记忆：放量起涨，直接开多！"""
    components.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 1.1; 
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# ==========================================
# 3. 健壮型数据引擎
# ==========================================
class WarriorEngine:
    def __init__(self):
        self.client = httpx.Client(timeout=10.0, http2=True)

    def get_market_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": "5m", "limit": "100", "_t": int(time.time()*1000)}
        try:
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if not data: return None
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
            return None
        except Exception:
            return None

# ==========================================
# 4. 大衍心法：量价与锚点逻辑
# ==========================================
def apply_strategy(df, p):
    # 均量线
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    # 实体比
    df['body_size'] = abs(df['c'] - df['o'])
    df['range'] = (df['h'] - df['l']).replace(0, 0.001)
    df['body_ratio'] = df['body_size'] / df['range']
    
    # 信号判定
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    # 关键阴阳线锚点：突破阴线跟多
    big_down = df[(df['c'] < df['o']) & (df['v'] > df['ma_v'] * 1.3)].iloc[-1:]
    big_up = df[(df['c'] > df['o']) & (df['v'] > df['ma_v'] * 1.3)].iloc[-1:]
    
    anchors = {}
    if not big_down.empty: anchors['down_high'] = big_down['h'].values[0]
    if not big_up.empty: anchors['up_low'] = big_up['l'].values[0]
    
    return df, anchors

# ==========================================
# 5. 主渲染循环
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    engine = WarriorEngine()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    params = st.session_state.params
    
    df = engine.get_market_data(symbol)
    
    # 黑屏防护核心
    if df is None or df.empty:
        st.warning("📡 正在同步 OKX 实时数据流，请稍后...")
        st.stop()

    df, anchors = apply_strategy(df, params)
    curr = df.iloc[-1]
    
    # 顶部仪表盘
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ETH 现价", f"${curr['c']:.2f}")
    c2.metric("当前量能比", f"{curr['v']/curr['ma_v']:.2f}x")
    
    status = "💎 震荡蓄势"
    if curr['buy_sig']: 
        status = "🔥 多头进攻"
        voice_alert("放量起涨，突破前高，直接开多")
    elif curr['sell_sig']: 
        status = "❄️ 空头下砸"
        voice_alert("放量下砸，跌破支撑，果断止损")
    elif curr['v'] < curr['ma_v'] * 0.6: 
        status = "⏳ 动能衰减"
    
    c3.metric("实时战报", status)
    c4.metric("心跳刷新", f"{datetime.now().strftime('%H:%M:%S')}")

    # 绘制 K 线图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="K线"
    ), row=1, col=1)

    # 关键锚点虚线
    if 'down_high' in anchors:
        fig.add_hline(y=anchors['down_high'], line_dash="dot", line_color="#ef5350", 
                     annotation_text="压力: 阴线高点", row=1, col=1)
    if 'up_low' in anchors:
        fig.add_hline(y=anchors['up_low'], line_dash="dot", line_color="#26a69a", 
                     annotation_text="支撑: 阳线低点", row=1, col=1)

    # 信号标记
    buys = df[df['buy_sig']]
    fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.999, mode='markers', 
                             marker=dict(symbol='triangle-up', size=15, color='#26a69a'), name='多'), row=1, col=1)
    
    # 成交量
    v_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='#ffa500', width=1.5)), row=2, col=1)

    fig.update_layout(height=800, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 6. 控制中心入口
# ==========================================
def main():
    st.sidebar.markdown("### ⚔️ Warrior V5.7 | 控制中心")
    
    with st.sidebar.expander("策略参数调节", expanded=True):
        ma_p = st.number_input("均量周期", 5, 100, 10)
        expand_p = st.slider("放量判定 (%)", 110, 500, 150)
        body_r = st.slider("突破实体比", 0.05, 0.90, 0.20)
    
    st.session_state.params = {"ma_len": ma_p, "expand_p": expand_p, "body_r": body_r}
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    st.sidebar.divider()
    st.sidebar.info("口诀：缩量是提醒，放量是信号，位置是关键。")
    
    dashboard_loop()

if __name__ == "__main__":
    main()
