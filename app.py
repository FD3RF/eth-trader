import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time

# ==========================================
# 1. 核心架构：性能级配置
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V3.1 | Pro", page_icon="⚔️")

# 极致精简 CSS，减少前端 DOM 解析耗时
st.markdown("""
    <style>
    .block-container { padding: 1rem 2rem; }
    div[data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 12px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #f0c05a; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 顶级引擎：高性能异步 Client (线程安全单例)
# ==========================================
class HighPerformanceEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 建立高性能连接池，复用连接减少握手耗时
            cls._instance.client = httpx.AsyncClient(
                timeout=httpx.Timeout(3.0, connect=1.5),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                http2=True # 开启 HTTP2
            )
        return cls._instance

    async def get_market_data(self, inst_id):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            # 增加时间戳参数绕过 CDN 缓存，确保数据的“新鲜度”
            params = {"instId": inst_id, "bar": "5m", "limit": "100", "_t": time.time()}
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                raw = resp.json().get('data', [])
                if len(raw) < 50: return None
                
                # 直接在内存中构造，避免多次转换
                df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                # 高速向量化类型转换
                num_cols = ['o','h','l','c','v']
                df[num_cols] = df[num_cols].astype(float)
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except Exception:
            return None
        return None

# ==========================================
# 3. 算法心法：向量化策略计算 (无循环逻辑)
# ==========================================
def warrior_logic_vectorized(df, p):
    """
    顶级程序员逻辑：利用 Numpy 底层优化。
    拒绝所有 for 循环，所有计算均为矩阵运算，耗时微秒级。
    """
    df = df.copy()
    c, o, h, l, v = df['c'].values, df['o'].values, df['h'].values, df['l'].values, df['v'].values
    
    # 均量与波幅
    ma_v = df['v'].rolling(p['v_len']).mean().values
    body = np.abs(c - o)
    price_range = (h - l) + 1e-9
    body_ratio = body / price_range
    
    # 信号矩阵初始化
    signals = np.zeros(len(df))
    
    # 逻辑 1：放量攻击 (布尔掩码运算)
    vol_expand = v > (ma_v * p['e_ratio'])
    is_up = c > o
    is_down = c < o
    strong_body = body_ratio > p['b_min']
    
    signals[vol_expand & is_up & strong_body] = 1   # 🔥 多头突击
    signals[vol_expand & is_down & strong_body] = -1 # 📉 空头压制
    
    # 逻辑 2：缩量拐点
    vol_shrink = v < (ma_v * p['s_ratio'])
    is_local_low = l == df['l'].rolling(10).min().values
    signals[vol_shrink & is_local_low] = 0.5        # 👀 缩量探底
    
    df['signal'] = signals
    df['ma_v'] = ma_v
    return df

# ==========================================
# 4. 实时渲染：Fragment 隔离更新 (解决性能闪烁)
# ==========================================
@st.fragment(run_every="5s")
def render_warrior_core():
    engine = HighPerformanceEngine()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    params = st.session_state.get('params')
    
    # 获取数据并校验
    df_raw = asyncio.run(engine.get_market_data(symbol))
    
    # 【顶级防御】双重边界校验，彻底告别 IndexError
    if df_raw is None or df_raw.empty or len(df_raw) < 50:
        st.warning("⚔️ 数据链路穿透中... 正在建立高性能同步...")
        return

    # 算法执行
    df = warrior_logic_vectorized(df_raw, params)
    last, prev = df.iloc[-1], df.iloc[-2]

    # 看板矩阵
    m1, m2, m3, m4 = st.columns(4)
    price_diff = last['c'] - prev['c']
    m1.metric("ETH PRICE", f"${last['c']:.2f}", f"{price_diff:.2f}")
    m2.metric("VOL RATIO", f"{(last['v']/last['ma_v']):.2f}x")
    
    sig_map = {1: "🔥 多头进攻", -1: "📉 空头压制", 0.5: "👀 缩量探底", 0: "💎 蓄势"}
    m3.metric("WARRIOR SIGNAL", sig_map.get(last['signal'], "💎 震荡蓄势"))
    m4.metric("HEARTBEAT", f"{int(time.time()%60)}s")

    # 交易计划交互
    if last['signal'] != 0:
        with st.chat_message("assistant"):
            st.write(f"**检测到 {sig_map[last['signal']]} 信号！**")
            st.write(f"建议进场位: {last['c']} | 逻辑: 量能比 {(last['v']/last['ma_v']):.2f}x")

    # ==========================================
    # 5. 绘图引擎优化：右侧交易轴布局
    # ==========================================
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
    
    # 主图 (Candlestick)
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        name="Price", increasing_line_color='#00ff00', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)

    # 现价穿透横线 (Annotation 优化)
    fig.add_hline(y=last['c'], line_dash="dash", line_color="#f0c05a", 
                  annotation_text=f"LIVE: {last['c']}", annotation_position="right")

    # 副图 (Volume)
    colors = ['#00cc96' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=colors, name="Vol"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='orange', width=1)), row=2, col=1)

    fig.update_layout(
        height=720, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=80),
        yaxis=dict(side="right", gridcolor="#232323", tickformat=".2f")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 6. 控制中心
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior Control")
    st.session_state.symbol = st.sidebar.text_input("CONTRACT", "ETH-USDT-SWAP")
    
    with st.sidebar.expander("ALGO PARAMS", expanded=True):
        st.session_state.params = {
            "v_len": st.number_input("MA Period", 5, 60, 10),
            "s_ratio": st.slider("Shrink %", 30, 90, 60) / 100,
            "e_ratio": st.slider("Expand X", 1.2, 3.0, 1.5),
            "b_min": st.slider("Body %", 0.0, 0.5, 0.2)
        }
    
    st.sidebar.divider()
    st.sidebar.markdown("✅ **Engine: High-Performance Mode**")
    
    render_warrior_core()

if __name__ == "__main__":
    main()
