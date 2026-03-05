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
st.set_page_config(layout="wide", page_title="Warrior V5.3 | 稳定版", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1.5rem 2rem; }
    /* 复刻顶部指标块样式 - 深蓝色调卡片 */
    [data-testid="stMetric"] { 
        background: #161a25; 
        border: 1px solid #2d323e; 
        padding: 20px; 
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] { color: #d4af37 !important; font-size: 1.8rem !important; font-family: 'Courier New', monospace; }
    [data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 0.9rem !important; }
    .modebar { display: none !important; }
    /* 侧边栏样式优化 */
    .st-expanderHeader { color: #d4af37 !important; font-weight: bold; }
    .sidebar-footer { position: fixed; bottom: 30px; left: 25px; font-size: 13px; color: #555; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心数据引擎 (OKX API 高性能连接)
# ==========================================
class WarriorCore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = httpx.Client(timeout=5.0, http2=True)
        return cls._instance

    def fetch_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            params = {"instId": symbol, "bar": "5m", "limit": "120", "_t": time.time_ns()}
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                for col in ['o','h','l','c','v']: df[col] = pd.to_numeric(df[col])
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except Exception: return None

# ==========================================
# 3. 大衍核心逻辑 (量价流判定)
# ==========================================
def apply_warrior_v5_3_logic(df, p):
    # 计算均量线
    df['ma_v'] = df['v'].rolling(p['ma_len']).mean()
    
    # 定义量能状态
    shrink_factor = p['shrink_p'] / 100.0
    expand_factor = p['expand_p'] / 100.0
    
    df['is_expand'] = df['v'] > (df['ma_v'] * expand_factor)
    df['is_shrink'] = df['v'] < (df['ma_v'] * shrink_factor)
    
    # 信号锚点判定 (精准对齐图片逻辑)
    # 计算K线实体比例：abs(C-O) / (H-L)
    df['body_size'] = abs(df['c'] - df['o'])
    df['range_size'] = df['h'] - df['l']
    df['body_ratio'] = df['body_size'] / df['range_size'].replace(0, 0.0001)
    
    # 做多信号：放量 + 阳线 + 实体比达标
    df['buy_sig'] = df['is_expand'] & (df['c'] > df['o']) & (df['body_ratio'] > p['body_r'])
    # 做空信号：放量 + 阴线 + 实体比达标
    df['sell_sig'] = df['is_expand'] & (df['c'] < df['o']) & (df['body_ratio'] > p['body_r'])
    
    return df

# ==========================================
# 4. 实时控制中心渲染
# ==========================================
@st.fragment(run_every="5s")
def render_v5_3_ui():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    params = st.session_state.get('params')
    
    df = core.fetch_data(symbol)
    if df is None or df.empty: return

    df = apply_warrior_v5_3_logic(df, params)
    last, prev = df.iloc[-1], df.iloc[-2]
    
    # --- A. 顶部四大核心战况盒 ---
    c1, c2, c3, c4 = st.columns(4)
    
    # 1. 价格与涨跌
    price_diff = ((last['c'] - prev['c']) / prev['c']) * 100
    color = "normal" if abs(price_diff) < 0.05 else ("inverse" if price_diff > 0 else "off")
    c1.metric("ETH 现价", f"${last['c']:.2f}", f"{price_diff:+.2f}%", delta_color=color)
    
    # 2. 量能比
    vol_ratio = last['v'] / last['ma_v']
    c2.metric("当前量能比", f"{vol_ratio:.2f}x")
    
    # 3. 实时战报 (对齐图片状态)
    status_text = "💎 震荡蓄势"
    if last['buy_sig']: status_text = "🔥 多头进攻"
    elif last['sell_sig']: status_text = "❄️ 空头下砸"
    elif last['is_shrink']: status_text = "⏳ 动能衰减"
    c3.metric("实时战报", status_text)
    
    # 4. 心跳刷新
    c4.metric("心跳刷新", f"{int(time.time()%60)}s")

    # --- B. 深度复刻 K 线图 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    
    # 主K线图层
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', 
        increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350', name="Price"
    ), row=1, col=1)

    # 精准信号锚点 (三角形)
    buys = df[df['buy_sig']]
    fig.add_trace(go.Scatter(
        x=buys['time'], y=buys['l'] * 0.9985, mode='markers',
        marker=dict(symbol='triangle-up', size=13, color='#26a69a'), name='Buy'
    ), row=1, col=1)

    sells = df[df['sell_sig']]
    fig.add_trace(go.Scatter(
        x=sells['time'], y=sells['h'] * 1.0015, mode='markers',
        marker=dict(symbol='triangle-down', size=13, color='#ef5350'), name='Sell'
    ), row=1, col=1)

    # 黄金现价横线
    fig.add_hline(y=last['c'], line_color="#d4af37", line_width=1, opacity=0.8, row=1, col=1)

    # 成交量柱状图与均量线
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=colors, opacity=0.35, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='#ffa500', width=1.5), name="Vol MA"), row=2, col=1)

    fig.update_layout(
        height=820, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=60),
        plot_bgcolor="#10121a", paper_bgcolor="#10121a",
        uirevision=symbol
    )
    
    st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

# ==========================================
# 5. 指挥中心主入口
# ==========================================
def main():
    st.sidebar.markdown("### ⚔️ Warrior 控制中心")
    
    # 对齐图片参数
    with st.sidebar.expander("策略参数调节", expanded=True):
        ma_period = st.number_input("均量周期", 5, 100, 10, step=1)
        shrink_p = st.slider("缩量判定 (%)", 10, 100, 60)
        expand_p = st.slider("放量判定 (%)", 110, 500, 150)
        body_ratio = st.slider("突破实体比", 0.05, 0.90, 0.20)
    
    st.session_state.params = {
        "ma_len": ma_period, "shrink_p": shrink_p, 
        "expand_p": expand_p, "body_r": body_ratio
    }
    
    st.sidebar.divider()
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    # 侧边栏页脚对齐图片
    st.sidebar.markdown('<div class="sidebar-footer">不灭大衍系统 · 纯净量价流</div>', unsafe_allow_html=True)
    
    render_v5_3_ui()

if __name__ == "__main__":
    main()
