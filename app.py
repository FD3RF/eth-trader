import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import time

# ==========================================
# 1. 像素级视觉复刻 (CSS 深度定制)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V5.3 | 控制中心", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    /* 复刻顶部指标块样式 */
    [data-testid="stMetric"] { 
        background: #161a25; 
        border: 1px solid #2d323e; 
        padding: 15px; 
        border-radius: 10px; 
    }
    [data-testid="stMetricValue"] { color: #d4af37 !important; font-family: 'Courier New', monospace; }
    .modebar { display: none !important; }
    /* 侧边栏底部文字 */
    .sidebar-footer { position: fixed; bottom: 20px; left: 20px; font-size: 12px; color: #666; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心数据引擎 (同步高并发)
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
            params = {"instId": symbol, "bar": "5m", "limit": "150", "_t": time.time_ns()}
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except: return None

# ==========================================
# 3. 稳定版量价逻辑 (精准信号锚点)
# ==========================================
def apply_stable_logic(df, p):
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    
    # 计算信号：九阴真经·量价流
    # 缩量 = 当前量 < 均量 * 缩量判定比
    # 放量 = 当前量 > 均量 * 放量判定比
    shrink_thresh = p['shrink_p'] / 100.0
    expand_thresh = p['expand_p'] / 100.0
    
    df['is_expand'] = df['v'] > (df['ma_v'] * expand_thresh)
    df['is_shrink'] = df['v'] < (df['ma_v'] * shrink_thresh)
    
    # 判定信号点 (用于图表锚点)
    # 做多锚点：放量阳线且实体占优
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & ((df['c']-df['o'])/(df['h']-df['l']) > p['body_r'])
    # 做空锚点：放量阴线且实体占优
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & ((df['o']-df['c'])/(df['h']-df['l']) > p['body_r'])
    
    return df

# ==========================================
# 4. 主渲染模块
# ==========================================
@st.fragment(run_every="5s")
def render_control_center():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    params = st.session_state.get('params')
    
    df = core.fetch_data(symbol)
    if df is None or df.empty: return

    df = apply_stable_logic(df, params)
    last, prev = df.iloc[-1], df.iloc[-2]
    
    # --- 1. 顶部四大战术指标 ---
    m1, m2, m3, m4 = st.columns(4)
    
    price_delta = ((last['c'] - prev['c']) / prev['c']) * 100
    m1.metric("ETH 现价", f"${last['c']:.2f}", f"{price_delta:.2f}%")
    
    vol_ratio = last['v'] / last['ma_v']
    m2.metric("当前量能比", f"{vol_ratio:.2f}x")
    
    # 实时战报逻辑
    status = "💎 震荡蓄势"
    if last['buy_sig']: status = "🔥 多头进攻"
    elif last['sell_sig']: status = "❄️ 空头下砸"
    elif last['is_shrink']: status = "⏳ 动能衰减"
    m3.metric("实时战报", status)
    
    m4.metric("心跳刷新", f"{int(time.time()%60)}s")

    # --- 2. 深度定制 K 线图 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    
    # 主 K 线
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="价格"
    ), row=1, col=1)

    # 绘制信号锚点 (精准三角形)
    buy_df = df[df['buy_sig']]
    fig.add_trace(go.Scatter(
        x=buy_df['time'], y=buy_df['l'] * 0.998,
        mode='markers', marker=dict(symbol='triangle-up', size=12, color='#26a69a'),
        name='做多信号'
    ), row=1, col=1)

    sell_df = df[df['sell_sig']]
    fig.add_trace(go.Scatter(
        x=sell_df['time'], y=sell_df['h'] * 1.002,
        mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ef5350'),
        name='做空信号'
    ), row=1, col=1)

    # 现价基准线
    fig.add_hline(y=last['c'], line_color="#d4af37", line_width=1, opacity=0.8, row=1, col=1)

    # 成交量与均量线
    v_colors = ['#26a69a' if _c >= _o else '#ef5350' for _c, _o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_colors, opacity=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='#ffa500', width=1.5)), row=2, col=1)

    fig.update_layout(
        height=780, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=60),
        uirevision=symbol
    )
    
    st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

# ==========================================
# 5. 指挥中心入口
# ==========================================
def main():
    # 侧边栏：Warrior 控制中心
    st.sidebar.title("⚔️ Warrior 控制中心")
    
    with st.sidebar.expander("策略参数调节", expanded=True):
        ma_len = st.number_input("均量周期", 5, 60, 10)
        shrink_p = st.slider("缩量判定 (%)", 20, 100, 60)
        expand_p = st.slider("放量判定 (%)", 110, 300, 150)
        body_r = st.slider("突破实体比", 0.05, 0.50, 0.20)
    
    st.session_state.params = {
        "ma_len": ma_len, "shrink_p": shrink_p, 
        "expand_p": expand_p, "body_r": body_r
    }
    
    st.sidebar.divider()
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    # 侧边栏页脚
    st.sidebar.markdown('<div class="sidebar-footer">不灭大衍系统 · 纯净量价流</div>', unsafe_allow_html=True)
    
    render_control_center()

if __name__ == "__main__":
    main()
