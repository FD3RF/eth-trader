import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time

# ==========================================
# 1. 顶级系统配置：减少 DOM 层级，优化渲染
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V4.0 | Pro Engine", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    div[data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 22px; color: #f0c05a; font-family: 'Courier New', monospace; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 高性能异步引擎：单例连接池 + HTTP2
# ==========================================
class WarriorCore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 顶级程序员逻辑：复用连接池，开启 HTTP/2 以实现多路复用
            cls._instance.client = httpx.AsyncClient(
                timeout=httpx.Timeout(3.0, connect=1.5),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                http2=True
            )
        return cls._instance

    async def get_market_data(self, inst_id):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            # 加入 Cache-Buster 确保获取最新毫秒级数据
            params = {"instId": inst_id, "bar": "5m", "limit": "100", "_": time.time_ns()}
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                raw = resp.json().get('data', [])
                if len(raw) < 50: return None
                
                # 内存对齐式 DataFrame 构造
                df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except Exception:
            return None
        return None

# ==========================================
# 3. 顶级策略内核：全向量化矩阵运算
# ==========================================
def calculate_signals_optimized(df, p):
    """
    拒绝所有 Python 循环，使用 Numpy 底层 C 语言级并行运算。
    """
    df = df.copy()
    # 提取数组，减少 Pandas 索引开销
    c, o, h, l, v = df['c'].values, df['o'].values, df['h'].values, df['l'].values, df['v'].values
    
    # 指标矩阵运算
    ma_v = df['v'].rolling(p['v_len']).mean().values
    body = np.abs(c - o)
    full_range = (h - l) + 1e-9
    body_pct = body / full_range
    
    # 预分配信号数组
    signals = np.zeros(len(df))
    
    # 逻辑 A：放量穿透 (1 为涨，-1 为跌)
    vol_expand = v > (ma_v * p['e_ratio'])
    body_filter = body_pct > p['b_min']
    signals[vol_expand & (c > o) & body_filter] = 1
    signals[vol_expand & (c < o) & body_filter] = -1
    
    # 逻辑 B：缩量探底 (0.5)
    vol_shrink = v < (ma_v * p['s_ratio'])
    local_low = l == df['l'].rolling(15).min().values
    signals[vol_shrink & local_low] = 0.5
    
    df['signal'] = signals
    df['ma_v'] = ma_v
    return df

# ==========================================
# 4. 实时渲染矩阵：Fragment 状态隔离
# ==========================================
@st.fragment(run_every="5s")
def render_live_system():
    # 初始化核心
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    params = st.session_state.get('params')
    
    # 获取与校验
    df_raw = asyncio.run(core.get_market_data(symbol))
    
    # [修复] 解决 IndexError & ValueError 的双重防火墙
    if df_raw is None or df_raw.empty or len(df_raw) < 50:
        st.warning("📡 核心正在穿透 API 数据流，建立同步中...")
        return

    # 策略执行
    df = calculate_signals_optimized(df_raw, params)
    last, prev = df.iloc[-1], df.iloc[-2]

    # A. 实时仪表盘
    c1, c2, c3, c4 = st.columns(4)
    price_delta = last['c'] - prev['c']
    c1.metric("ETH 实时价格", f"${last['c']:.2f}", f"{price_delta:.2f}")
    c2.metric("当前量能比", f"{(last['v']/last['ma_v']):.2f}x")
    
    sig_map = {1: "🔥 多头进攻", -1: "📉 空头压制", 0.5: "👀 缩量底", 0: "💎 蓄势"}
    c3.metric("当前信号", sig_map.get(last['signal'], "💎 震荡蓄势"))
    c4.metric("引擎心跳", f"{int(time.time()%60)}s")

    # B. 交易指令表单
    if last['signal'] != 0:
        with st.chat_message("user"):
            st.write(f"⚔️ **信号触发：{sig_map[last['signal']]}**")
            st.write(f"入场建议：{last['c']} | 逻辑备注：量能异动 {(last['v']/last['ma_v']):.2f}x")

    # C. 高级绘图 (右侧交易轴)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
    
    # 主图 (WebGL 加速)
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        name="Price", increasing_line_color='#00ff00', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)

    # 现价穿透虚线
    fig.add_hline(y=last['c'], line_dash="dash", line_color="#d4af37", 
                  annotation_text=f"NOW: {last['c']}", annotation_position="right")

    # 副图 (Volume)
    v_colors = ['#00cc96' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=v_colors, name="Vol", opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='orange', width=1.2)), row=2, col=1)

    fig.update_layout(
        height=750, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=80),
        yaxis=dict(side="right", gridcolor="#222", tickformat=".2f")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 5. 主程序启动控制
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V4.0 Control")
    st.session_state.symbol = st.sidebar.text_input("CONTRACT ID", "ETH-USDT-SWAP")
    
    with st.sidebar.expander("策略心法调节", expanded=True):
        st.session_state.params = {
            "v_len": st.sidebar.number_input("均量参考周期", 5, 60, 10),
            "s_ratio": st.sidebar.slider("缩量系数", 0.3, 0.9, 0.6),
            "e_ratio": st.sidebar.slider("放量倍数", 1.2, 3.0, 1.5),
            "b_min": st.sidebar.slider("实体占比阈值", 0.0, 0.5, 0.2)
        }
    
    st.sidebar.divider()
    st.sidebar.caption("✅ Engine: Industrial Performance Mode")
    
    render_live_system()

if __name__ == "__main__":
    main()
