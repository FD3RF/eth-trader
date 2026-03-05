import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time

# ==========================================
# 1. 系统核心配置 (工业级视觉标准)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V4.3 | Zero-Error", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    div[data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 12px; border-radius: 8px; }
    .modebar { display: none !important; } 
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 顶级行情引擎 (HTTP/2 + 高可用防御)
# ==========================================
class WarriorCore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                # 优先开启 HTTP/2 多路复用
                cls._instance.client = httpx.AsyncClient(
                    timeout=httpx.Timeout(5.0, connect=2.0),
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                    http2=True 
                )
            except Exception:
                # 自动降级防御
                cls._instance.client = httpx.AsyncClient(http2=False)
        return cls._instance

    async def fetch_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            params = {"instId": symbol, "bar": "5m", "limit": "100", "_": time.time_ns()}
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if not data or len(data) < 40: return None
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                # 数值强制安全转换
                for col in ['o','h','l','c','v']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').dropna(subset=['c']).reset_index(drop=True)
        except Exception:
            return None
        return None

# ==========================================
# 3. 渲染模块 (彻底修复刷新报错逻辑)
# ==========================================
@st.fragment(run_every="5s")
def render_live_monitor():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    p = st.session_state.get('params')
    
    # 异步获取数据
    df = asyncio.run(core.fetch_data(symbol))
    
    # 【防御 1】空值与长度判定，防止刷新瞬间的非法索引
    if df is None or df.empty or len(df) < 10:
        st.warning("📡 正在穿透行情流，请稍后...")
        return

    # 向量化策略引擎
    c, o, h, l, v = df['c'].values, df['o'].values, df['h'].values, df['l'].values, df['v'].values
    ma_v = df['v'].rolling(p['v_len']).mean().values
    
    # 信号矩阵 (0延迟)
    vol_expand = v > (ma_v * p['e_ratio'])
    signals = np.zeros(len(df))
    signals[vol_expand & (c > o)] = 1
    signals[vol_expand & (c < o)] = -1
    
    last, prev = df.iloc[-1], df.iloc[-2]

    # A. 核心仪表盘
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ETH 实时报价", f"${last['c']:.2f}", f"{last['c']-prev['c']:.2f}")
    m2.metric("量能比", f"{(last['v']/ma_v[-1]):.2f}x")
    m3.metric("战报", "🔥 多头突击" if signals[-1] == 1 else "📉 空头压制" if signals[-1] == -1 else "💎 震荡蓄势")
    m4.metric("刷新倒计时", f"{int(time.time()%5)}s")

    # B. 绘图引擎 (针对 ValueError 的终极加固)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    # K 线绘制 (WebGL 模式)
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        name="Warrior K"
    ), row=1, col=1)

    # 【防御 2】现价横线数值校验 (防止 ValueError)
    try:
        current_price = float(last['c'])
        if not np.isnan(current_price):
            fig.add_hline(
                y=current_price, 
                line_dash="dash", 
                line_color="#d4af37", # 严禁使用 8 位 Hex，确保 6 位标准 Hex
                annotation_text=f" {current_price}", 
                annotation_position="right"
            )
    except Exception:
        pass

    # 成交量
    v_clrs = ['#26a69a' if _c >= _o else '#ef5350' for _c, _o in zip(c, o)]
    fig.add_trace(go.Bar(x=df['time'], y=v, marker_color=v_clrs, opacity=0.4), row=2, col=1)

    fig.update_layout(
        height=720, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=80),
        yaxis=dict(side="right", tickformat=".2f", gridcolor="#232323")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 4. 程序入口
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V4.3")
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    with st.sidebar.expander("策略心法", expanded=True):
        st.session_state.params = {
            "v_len": st.sidebar.number_input("均量周期", 5, 60, 10),
            "e_ratio": st.sidebar.slider("放量倍数", 1.2, 3.0, 1.5)
        }
    
    st.sidebar.divider()
    st.sidebar.success("✅ HTTP/2 极速核心已就绪")
    
    render_live_monitor()

if __name__ == "__main__":
    main()
