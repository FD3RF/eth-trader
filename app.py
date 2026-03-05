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
# 1. 顶级视觉初始化
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V6.0 | 双向靶心版", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #11141c; border: 1px solid #2d323e; padding: 15px; border-radius: 10px; }
    .status-card { background: #1a1c23; border: 4px left solid #d4af37; padding: 20px; border-radius: 12px; margin-bottom: 20px; border-left: 5px solid #d4af37; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 语音咆哮引擎（双向播报控制）
# ==========================================
def voice_alert(text):
    components.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 1.1; window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# ==========================================
# 3. 核心引擎与逻辑
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
        except Exception: return None

def apply_warrior_logic(df, p):
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l']).replace(0, 0.001)
    
    # 信号判定：量价共振
    df['is_expand'] = df['v'] > df['ma_v'] * (p['expand_p'] / 100.0)
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    # 锚点锁定：阴高阳低
    big_down = df[(df['c'] < df['o']) & (df['v'] > df['ma_v'] * 1.3)].iloc[-1:]
    big_up = df[(df['c'] > df['o']) & (df['v'] > df['ma_v'] * 1.3)].iloc[-1:]
    
    anchors = {
        'down_high': big_down['h'].values[0] if not big_down.empty else 999999,
        'up_low': big_up['l'].values[0] if not big_up.empty else 0
    }
    return df, anchors

# ==========================================
# 4. 实时战报渲染（包含双向语音逻辑）
# ==========================================
def render_detailed_report(curr, anchors):
    vol_ratio = curr['v'] / curr['ma_v']
    price = curr['c']
    upper = anchors['down_high']
    lower = anchors['up_low']

    # 做多播报逻辑
    if curr['buy_sig'] and price > upper:
        status, detail, color = "🚀 核心突破", "满足放量且突破阴线高点，多头总攻！", "#26a69a"
        voice_alert("放量起涨，突破前高，直接开多")
    # 做空/止损播报逻辑
    elif curr['sell_sig'] or price < lower:
        status, detail, color = "❄️ 趋势转弱", "跌破支撑或出现放量阴线，执行空头保护。", "#ef5350"
        voice_alert("趋势转弱，注意离场或反手做空")
    elif vol_ratio < 0.6:
        status, detail, color = "⏳ 动能衰减", "缩量回踩中，低点不破不入场。", "#888888"
    else:
        status, detail, color = "💎 震荡蓄势", f"量能 {vol_ratio:.2f}x，观察关键位支撑情况。", "#1e90ff"

    st.markdown(f"""
        <div class="status-card" style="border-left: 8px solid {color};">
            <h1 style='color:{color}; margin:0;'>{status}</h1>
            <p style='color:#ccc; font-size:20px; margin-top:10px;'><b>实战逻辑：</b>{detail}</p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. 主图渲染（准星+靶心）
# ==========================================
@st.fragment(run_every="10s")
def dashboard_loop():
    engine = WarriorEngine()
    df = engine.get_market_data(st.session_state.symbol)
    if df is None or df.empty:
        st.warning("📡 正在校准大衍系统...")
        st.stop()

    df, anchors = apply_warrior_logic(df, st.session_state.params)
    curr = df.iloc[-1]
    
    render_detailed_report(curr, anchors)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("ETH 现价", f"${curr['c']:.2f}")
    c2.metric("当前量能比", f"{curr['v']/curr['ma_v']:.2f}x")
    c3.metric("最后更新", f"{datetime.now().strftime('%H:%M:%S')}")

    # 绘图逻辑
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
                                 increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="K线"), row=1, col=1)

    # A. 准星信号：三角回档
    buys = df[df['buy_sig']]
    fig.add_trace(go.Scatter(x=buys['time'], y=buys['l']*0.998, mode='markers',
                             marker=dict(symbol='triangle-up', size=15, color='#00ffcc', line=dict(width=1, color='white')),
                             name='做多信号'), row=1, col=1)
    sells = df[df['sell_sig']]
    fig.add_trace(go.Scatter(x=sells['time'], y=sells['h']*1.002, mode='markers',
                             marker=dict(symbol='triangle-down', size=15, color='#ff3366', line=dict(width=1, color='white')),
                             name='做空信号'), row=1, col=1)

    # B. 离场靶心线
    if curr['buy_sig']:
        sl = anchors['up_low'] 
        risk = curr['c'] - sl
        if risk > 0:
            tp = curr['c'] + (risk * st.session_state.params['rr_ratio'])
            fig.add_hline(y=tp, line_dash="dash", line_color="#00ffcc", annotation_text="🎯 止盈靶心", row=1, col=1)
            fig.add_hline(y=sl, line_dash="dash", line_color="#ef5350", annotation_text="🛡️ 止损保护", row=1, col=1)

    # C. 压力支撑锚点
    fig.add_hline(y=anchors['down_high'], line_dash="dot", line_color="#ef5350", annotation_text="压力:阴高", row=1, col=1)
    fig.add_hline(y=anchors['up_low'], line_dash="dot", line_color="#26a69a", annotation_text="支撑:阳低", row=1, col=1)

    # D. 成交量
    v_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='#ffa500', width=1.5)), row=2, col=1)

    fig.update_layout(height=750, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 6. 控制中心
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V6.0")
    ma_p = st.sidebar.number_input("均量周期", 5, 100, 10)
    expand_p = st.sidebar.slider("放量判定 (%)", 110, 500, 150)
    body_r = st.sidebar.slider("突破实体比", 0.05, 0.90, 0.20)
    rr_ratio = st.sidebar.slider("盈亏比 (1:X)", 1.0, 3.0, 1.5, step=0.1)
    
    st.session_state.params = {"ma_len": ma_p, "expand_p": expand_p, "body_r": body_r, "rr_ratio": rr_ratio}
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    st.sidebar.divider()
    st.sidebar.markdown("### 🏹 战神状态")
    st.sidebar.write("✅ 双向信号播报已开启")
    st.sidebar.write("✅ 离场靶心自动测算")
    
    dashboard_loop()

if __name__ == "__main__":
    main()
