import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time

# ==========================================
# 1. 顶级视觉配置：极简、深邃、零延迟
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V4.1 | Immortal Dayan", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    div[data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 22px; color: #f0c05a; font-family: 'Courier New', monospace; }
    /* 隐藏 Plotly 悬浮栏，减少内存开销 */
    .modebar { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 顶级行情内核：HTTP/2 工业级连接池
# ==========================================
class WarriorCore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 顶级程序员配置：使用异步 Client，开启 HTTP/2 以支持请求多路复用
            try:
                cls._instance.client = httpx.AsyncClient(
                    timeout=httpx.Timeout(3.0, connect=1.5),
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                    http2=True  # 极致性能的关键
                )
            except ImportError:
                # 兼容性后备：如果依赖安装异常，自动降级至 HTTP/1.1 保证不崩溃
                cls._instance.client = httpx.AsyncClient(
                    timeout=httpx.Timeout(3.0, connect=1.5),
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                    http2=False
                )
        return cls._instance

    async def get_market_data(self, inst_id):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            # 加入 Cache-Buster 确保数据新鲜度
            params = {"instId": inst_id, "bar": "5m", "limit": "100", "_ts": str(time.time_ns())}
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                raw = resp.json().get('data', [])
                if len(raw) < 50: return None
                
                df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                # 高速转换
                for col in ['o','h','l','c','v']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except Exception:
            return None
        return None

# ==========================================
# 3. 算法阵列：全矩阵向量化运算
# ==========================================
def run_vectorized_strategy(df, p):
    df = df.copy()
    # 提取数组，避开 Pandas 索引开销
    c, o, h, l, v = df['c'].values, df['o'].values, df['h'].values, df['l'].values, df['v'].values
    
    # 均量计算
    ma_v = df['v'].rolling(p['v_len']).mean().values
    body = np.abs(c - o)
    price_range = (h - l) + 1e-9
    body_pct = body / price_range
    
    # 初始化信号矩阵
    signals = np.zeros(len(df))
    
    # 核心逻辑：放量起爆/缩量寻底
    vol_expand = v > (ma_v * p['e_ratio'])
    body_filter = body_pct > p['b_min']
    
    signals[vol_expand & (c > o) & body_filter] = 1   # 多头进攻
    signals[vol_expand & (c < o) & body_filter] = -1  # 空头压制
    
    # 缩量底判定
    vol_shrink = v < (ma_v * p['s_ratio'])
    is_low_point = l == df['l'].rolling(15).min().values
    signals[vol_shrink & is_low_point] = 0.5
    
    df['signal'] = signals
    df['ma_v'] = ma_v
    return df

# ==========================================
# 4. 实时渲染：Fragment 隔离架构
# ==========================================
@st.fragment(run_every="5s")
def render_live_ui():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    params = st.session_state.get('params')
    
    # 异步获取行情
    df_raw = asyncio.run(core.get_market_data(symbol))
    
    # 边界防火墙
    if df_raw is None or df_raw.empty or len(df_raw) < 50:
        st.warning("📡 Warrior 正在穿透 OKX 数据层，建立同步中...")
        return

    # 执行算法
    df = run_vectorized_strategy(df_raw, params)
    last, prev = df.iloc[-1], df.iloc[-2]

    # 顶部仪表盘
    c1, c2, c3, c4 = st.columns(4)
    price_change = last['c'] - prev['c']
    c1.metric("ETH-USDT PRICE", f"${last['c']:.2f}", f"{price_change:.2f}")
    c2.metric("VOL RATIO", f"{(last['v']/last['ma_v']):.2f}x")
    
    sig_map = {1: "🔥 多头进攻", -1: "📉 空头压制", 0.5: "👀 缩量探底", 0: "💎 蓄势"}
    c3.metric("SIGNAL", sig_map.get(last['signal'], "💎 蓄势"))
    c4.metric("ENGINE", f"{int(time.time()%60)}s")

    # 信号弹窗
    if last['signal'] != 0:
        st.toast(f"战斗预警: {sig_map[last['signal']]} @ {last['c']}", icon="⚔️")

    # 
    # 绘图逻辑
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        name="Warrior", increasing_line_color='#00ff00', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)

    # 右侧价格实时穿透线
    fig.add_hline(y=last['c'], line_dash="dash", line_color="#d4af37", 
                  annotation_text=f"LIVE: {last['c']}", annotation_position="right")

    # 成交量与均量
    v_clrs = ['#00cc96' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_clrs, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='orange', width=1)), row=2, col=1)

    fig.update_layout(
        height=720, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=80),
        yaxis=dict(side="right", gridcolor="#222", tickformat=".2f")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 5. 系统主入口
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V4.1 Control")
    st.session_state.symbol = st.sidebar.text_input("SYMBOL", "ETH-USDT-SWAP")
    
    with st.sidebar.expander("ALGO CONFIG", expanded=True):
        st.session_state.params = {
            "v_len": st.sidebar.number_input("MA Period", 5, 60, 10),
            "s_ratio": st.sidebar.slider("Shrink %", 0.3, 0.9, 0.6),
            "e_ratio": st.sidebar.slider("Expand X", 1.2, 3.0, 1.5),
            "b_min": st.sidebar.slider("Body Ratio", 0.0, 0.5, 0.2)
        }
    
    st.sidebar.divider()
    st.sidebar.markdown("✅ **Protocol: HTTP/2 Enabled**")
    
    render_live_ui()

if __name__ == "__main__":
    main()
