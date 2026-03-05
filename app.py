import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import time

# ==========================================
# 1. 顶级视觉配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V5.1 | 九阴真经", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    [data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 10px; border-radius: 8px; }
    .stAlert { padding: 0.5rem; border-radius: 4px; }
    .modebar { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 工业级数据引擎
# ==========================================
class WarriorCore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = httpx.Client(timeout=4.0, http2=True)
        return cls._instance

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
# 3. “九阴真经”算法内核
# ==========================================
def apply_nine_yin_logic(df, p):
    # A. 基础指标：5周期均量
    df['v_ma5'] = df['v'].rolling(5).mean()
    
    # B. 动态锚定前高前低 (Lookback=10)
    lookback = 10
    df['local_high'] = df['h'].rolling(window=lookback, center=True).max()
    df['local_low'] = df['l'].rolling(window=lookback, center=True).min()
    
    # 提取当前状态
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    p_high = df['h'].iloc[-lookback:-1].max()
    p_low = df['l'].iloc[-lookback:-1].min()
    
    # C. 量能状态判定
    is_shrinking = curr['v'] < curr['v_ma5'] * 0.6
    is_expanding = curr['v'] > curr['v_ma5'] * p['e_ratio']
    
    # D. 九阴真经逻辑校验
    checklist = {
        "做多准备 (缩量不破底)": is_shrinking and curr['l'] >= p_low * 0.9998,
        "做多起航 (放量破前高)": is_expanding and curr['c'] > p_high,
        "做空准备 (缩量不过顶)": is_shrinking and curr['h'] <= p_high * 1.0002,
        "做空起航 (放量破前低)": is_expanding and curr['c'] < p_low
    }
    
    # E. 诱多诱空判定 (长影线 + 巨量)
    is_trap_long = is_expanding and (curr['h'] > p_high) and (curr['c'] < p_high)
    is_trap_short = is_expanding and (curr['l'] < p_low) and (curr['c'] > p_low)
    
    return checklist, p_high, p_low, is_trap_long, is_trap_short

# ==========================================
# 4. 实时动态扫描 UI
# ==========================================
@st.fragment(run_every="5s")
def render_warrior_v5_1():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    p = st.session_state.get('params')
    
    df = core.fetch_data(symbol)
    if df is None: return

    # 逻辑运算
    check, p_high, p_low, t_long, t_short = apply_nine_yin_logic(df, p)
    last = df.iloc[-1]

    # 布局：左侧看板，右侧图表
    col_check, col_chart = st.columns([1, 4])

    with col_check:
        st.subheader("⚔️ 战术校验")
        for key, val in check.items():
            color = "green" if val else "gray"
            st.markdown(f"**{'✅' if val else '⚪'} {key}**")
        
        st.divider()
        if t_long: st.error("⚠️ 警惕：放量诱多陷阱")
        if t_short: st.error("⚠️ 警惕：放量诱空陷阱")
        
        st.metric("前高阻力", f"{p_high:.2f}")
        st.metric("前低支撑", f"{p_low:.2f}")

    with col_chart:
        # 仪表盘
        m1, m2, m3 = st.columns(3)
        m1.metric("ETH Price", f"${last['c']:.2f}")
        m2.metric("Vol Ratio", f"{last['v']/last['v_ma5']:.2f}x")
        m3.metric("Heartbeat", f"{int(time.time()%5)}s")

        # 绘图
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # K线与前高低线
        fig.add_trace(go.Candlestick(x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="K"), row=1, col=1)
        fig.add_hline(y=p_high, line_dash="dot", line_color="#ff4b4b", opacity=0.5, row=1, col=1)
        fig.add_hline(y=p_low, line_dash="dot", line_color="#26a69a", opacity=0.5, row=1, col=1)
        
        # 现价金色准星
        fig.add_hline(y=last['c'], line_color="#d4af37", line_dash="dash", row=1, col=1)

        # 成交量
        v_colors = ['#26a69a' if _c >= _o else '#ef5350' for _c, _o in zip(df['c'], df['o'])]
        fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['v_ma5'], line=dict(color='#ffa500', width=1)), row=2, col=1)

        fig.update_layout(height=700, template="plotly_dark", showlegend=False, 
                          xaxis_rangeslider_visible=False, margin=dict(t=10,b=10,l=10,r=50),
                          uirevision=symbol)
        
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

# ==========================================
# 5. 主程序
# ==========================================
def main():
    st.sidebar.title("Warrior V5.1")
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    st.session_state.params = {
        "e_ratio": st.sidebar.slider("放量起步倍数", 1.5, 3.0, 1.8),
        "v_len": 5
    }
    
    st.sidebar.info("💡 缩量 = < 60% 均量\n💡 放量 = > 1.8x 均量")
    
    render_warrior_v5_1()

if __name__ == "__main__":
    main()
