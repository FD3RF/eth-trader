import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import time

# ==========================================
# 1. 顶级视觉配置 (极简、专注、零性能浪费)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V5.0 | 巅峰架构", page_icon="⚔️")

st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem; }
    div[data-testid="stMetric"] { background: #0e1117; border: 1px solid #262730; padding: 12px; border-radius: 8px; }
    /* 彻底隐藏顶部无用元素和 Plotly 浮动菜单，榨干每一滴渲染性能 */
    .modebar { display: none !important; } 
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 工业级同步 HTTP/2 引擎 (彻底告别 asyncio 内存泄漏)
# ==========================================
class WarriorCore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                # 顶级重构：在 Streamlit 这种多线程模型下，同步 HTTP/2 连接池性能反而更稳定
                cls._instance.client = httpx.Client(
                    timeout=httpx.Timeout(4.0, connect=1.5),
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
                    http2=True 
                )
            except Exception:
                # 终极防御：如果环境真的连 h2 都没有，平滑降级，绝不崩溃
                cls._instance.client = httpx.Client(http2=False)
        return cls._instance

    def fetch_data(self, symbol):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            # 纳秒级时间戳，穿透所有 CDN 缓存
            params = {"instId": symbol, "bar": "5m", "limit": "100", "_t": time.time_ns()}
            resp = self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if not data or len(data) < 40: return None
                
                # 极限清洗与向量化转换
                df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                for col in ['o','h','l','c','v']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                
                # 强制剔除任何脏数据，保障 Plotly 引擎绝对安全
                return df.sort_values('time').dropna(subset=['o', 'h', 'l', 'c', 'v']).reset_index(drop=True)
        except Exception:
            return None
        return None

# ==========================================
# 3. 实时渲染模块 (已锁死 UI 视角状态)
# ==========================================
@st.fragment(run_every="5s")
def render_live_monitor():
    core = WarriorCore()
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    p = st.session_state.get('params')
    
    # 获取行情 (已从耗时的异步重构为极速同步调用)
    df = core.fetch_data(symbol)
    
    # 绝对防御闸门
    if df is None or df.empty:
        st.warning(f"📡 正在与 OKX [ {symbol} ] 建立量子纠缠，请维持网络畅通...")
        return

    # 顶级量价策略内核 (Numpy C级底层加速)
    c, o, h, l, v = df['c'].values, df['o'].values, df['h'].values, df['l'].values, df['v'].values
    ma_v = df['v'].rolling(p['v_len']).mean().values
    
    # 矩阵化信号判定 (消除所有循环运算)
    vol_expand = v > (ma_v * p['e_ratio'])
    signals = np.zeros(len(df))
    signals[vol_expand & (c > o)] = 1   # 多头爆破
    signals[vol_expand & (c < o)] = -1  # 空头下砸
    
    last, prev = df.iloc[-1], df.iloc[-2]

    # 战术仪表盘
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ETH 实时现价", f"${last['c']:.2f}", f"{last['c']-prev['c']:.2f}")
    m2.metric("实时量能突变率", f"{(last['v']/ma_v[-1]):.2f}x")
    m3.metric("大衍系统战报", "🔥 总攻时刻" if signals[-1] != 0 else "💎 震荡隐忍")
    m4.metric("引擎刷新周期", f"{int(time.time()%5)}s")

    # 绘图逻辑
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    # K 线绘制 (WebGL 极致渲染)
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="价格流"
    ), row=1, col=1)

    # 金色穿透线 (严格遵守 6位 Hex 规范)
    cur_p = float(last['c'])
    if not np.isnan(cur_p):
        fig.add_hline(
            y=cur_p, 
            line_dash="dash", 
            line_color="#d4af37", 
            annotation_text=f" 狙击位: {cur_p}", 
            annotation_position="right"
        )

    # 动态量能柱与移动均线
    v_colors = ['#26a69a' if _c >= _o else '#ef5350' for _c, _o in zip(c, o)]
    fig.add_trace(go.Bar(x=df['time'], y=v, marker_color=v_colors, opacity=0.45), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=ma_v, line=dict(color='#ffa500', width=1.5)), row=2, col=1)

    # 【巅峰优化】加入了 uirevision，确保 5 秒刷新时，你放大的图表视角绝对不会重置！
    fig.update_layout(
        height=750, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=80),
        yaxis=dict(side="right", tickformat=".2f", gridcolor="#1e1e1e"),
        uirevision=symbol # 关键：根据交易对锁死 UI 缩放状态
    )

    # 渲染图表 (严格使用 width='stretch')
    st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

# ==========================================
# 4. 指挥中心主入口
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior V5.0 Core")
    st.session_state.symbol = st.sidebar.text_input("目标合约", "ETH-USDT-SWAP")
    
    with st.sidebar.expander("大衍算力配置", expanded=True):
        st.session_state.params = {
            "v_len": st.sidebar.number_input("均量侦测周期", 5, 60, 10),
            "e_ratio": st.sidebar.slider("起爆阈值 (倍)", 1.2, 3.0, 1.5)
        }
    
    st.sidebar.divider()
    st.sidebar.success("✅ V5.0 量子引擎持续压制中...")
    
    render_live_monitor()

if __name__ == "__main__":
    main()
