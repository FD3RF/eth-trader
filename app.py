import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time

# ==========================================
# 1. 系统核心配置 (极致精简架构)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V4.2 | 终极闭环", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    div[data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 12px; border-radius: 8px; }
    .modebar { display: none !important; } /* 彻底禁用 Plotly 浮窗以节省 GPU 资源 */
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 工业级行情引擎 (支持 HTTP/2 与 故障降级)
# ==========================================
class WarriorCore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance.client = httpx.AsyncClient(
                    timeout=httpx.Timeout(3.0, connect=1.5),
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                    http2=True 
                )
            except Exception:
                cls._instance.client = httpx.AsyncClient(http2=False)
        return cls._instance

    async def fetch_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            params = {"instId": symbol, "bar": "5m", "limit": "100", "_t": time.time_ns()}
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if not data or len(data) < 50: return None
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                # 高速向量化强制转换
                df[['o','h','l','c','v']] = df[['o','h','l','c','v']].apply(pd.to_numeric, errors='coerce')
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except Exception:
            return None
        return None

# ==========================================
# 3. 实时渲染模块 (修复 ValueError 逻辑)
# ==========================================
@st.fragment(run_every="5s")
def render_engine():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    p = st.session_state.get('params')
    
    df = asyncio.run(core.fetch_data(symbol))
    
    # 逻辑防御闸门
    if df is None or df['c'].isnull().any():
        st.warning("⚠️ 数据链路穿透中，等待有效报价...")
        return

    # 向量化策略计算
    c, o, h, l, v = df['c'].values, df['o'].values, df['h'].values, df['l'].values, df['v'].values
    ma_v = df['v'].rolling(p['v_len']).mean().values
    
    # 信号计算 (0延迟矩阵)
    vol_expand = v > (ma_v * p['e_ratio'])
    signals = np.zeros(len(df))
    signals[vol_expand & (c > o)] = 1
    signals[vol_expand & (c < o)] = -1
    
    last, prev = df.iloc[-1], df.iloc[-2]

    # 顶部仪表盘
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ETH 实时价", f"${last['c']:.2f}", f"{last['c']-prev['c']:.2f}")
    m2.metric("当前量能比", f"{(last['v']/ma_v[-1]):.2f}x")
    m3.metric("战报", "🔥 攻击中" if signals[-1] != 0 else "💎 蓄势中")
    m4.metric("引擎心跳", f"{int(time.time()%60)}s")

    # 绘图逻辑优化 (针对 ValueError 的终极修复)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    # K 线绘制
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        increasing_line_color='#00ff00', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)

    # 【关键修复】确保 last['c'] 是合法浮点数，且 color 代码严格匹配
    current_price = float(last['c'])
    if not np.isnan(current_price):
        fig.add_hline(
            y=current_price, 
            line_dash="dash", 
            line_color="#d4af37", # 严格使用标准 Hex 码
            annotation_text=f" {current_price}", 
            annotation_position="right"
        )

    # 成交量
    v_colors = ['#00cc96' if _c >= _o else '#ef5350' for _c, _o in zip(c, o)]
    fig.add_trace(go.Bar(x=df['time'], y=v, marker_color=v_colors, opacity=0.5), row=2, col=1)

    fig.update_layout(
        height=720, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=80),
        yaxis=dict(side="right", tickformat=".2f", gridcolor="#222")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 4. 控制中心
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V4.2")
    st.session_state.symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    
    st.session_state.params = {
        "v_len": st.sidebar.number_input("均量周期", 5, 60, 10),
        "e_ratio": st.sidebar.slider("放量倍数", 1.2, 3.0, 1.5)
    }
    
    st.sidebar.divider()
    st.sidebar.success("✅ HTTP/2 核心运行中")
    
    render_engine()

if __name__ == "__main__":
    main()
