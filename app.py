import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time
from datetime import datetime

# ==========================================
# 1. 核心系统配置 (极致响应速度优化)
# ==========================================
st.set_page_config(layout="wide", page_title="Warrior V3.0 | 极致性能版", page_icon="⚔️")

# 强制深色 UI 注入，减少 DOM 渲染压力
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    div[data-testid="stMetric"] { background: #11141a; border: 1px solid #2d3139; padding: 10px; border-radius: 8px; }
    .stAlert { background-color: #1a1c24; border: 1px solid #d4af37; color: #d4af37; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 异步高性能引擎 (解决 ModuleNotFound 与超时)
# ==========================================
class WarriorEngine:
    """顶级程序员逻辑：使用 Singleton 模式思想与连接池管理"""
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True
        )

    async def get_candles(self, instId, bar="5m"):
        url = "https://www.okx.com/api/v5/market/candles"
        try:
            # 增加随机数后缀防止缓存污染
            resp = await self.client.get(url, params={"instId": instId, "bar": bar, "limit": "100", "t": time.time()})
            if resp.status_code == 200:
                raw_data = resp.json().get('data', [])
                if not raw_data or len(raw_data) < 50: return None
                
                df = pd.DataFrame(raw_data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                # 向量化转换，性能远超 apply
                for col in ['o','h','l','c','v']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['time'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
                return df.sort_values('time').reset_index(drop=True)
        except Exception as e:
            return None
        return None

# 初始化引擎
if 'engine' not in st.session_state:
    st.session_state.engine = WarriorEngine()

# ==========================================
# 3. 顶级交易员逻辑模块 (信号穿透)
# ==========================================
def calculate_warrior_signals(df, p):
    df = df.copy()
    # 核心指标向量化计算
    df['ma_v'] = df['v'].rolling(p['v_len']).mean()
    df['body'] = (df['c'] - df['o']).abs()
    df['range'] = (df['h'] - df['l']).replace(0, 1e-9)
    df['body_ratio'] = df['body'] / df['range']
    
    # 信号逻辑矩阵
    df['signal'] = 0 
    # 放量起涨 (1) / 放量杀跌 (-1)
    df.loc[(df['v'] > df['ma_v'] * p['e_ratio']) & (df['c'] > df['o']) & (df['body_ratio'] > p['b_min']), 'signal'] = 1
    df.loc[(df['v'] > df['ma_v'] * p['e_ratio']) & (df['c'] < df['o']) & (df['body_ratio'] > p['b_min']), 'signal'] = -1
    # 缩量底 (0.5) / 缩量顶 (-0.5)
    df.loc[(df['v'] < df['ma_v'] * p['s_ratio']) & (df['l'] == df['l'].rolling(10).min()), 'signal'] = 0.5
    
    return df

# ==========================================
# 4. 实时渲染矩阵 (解决 IndexError 与 ValueError)
# ==========================================
@st.fragment(run_every="5s")
def live_dashboard():
    # 动态参数获取
    symbol = st.session_state.get('symbol', 'ETH-USDT-SWAP')
    params = st.session_state.get('params', {"v_len":10, "s_ratio":0.6, "e_ratio":1.5, "b_min":0.2})
    
    df = asyncio.run(st.session_state.engine.get_candles(symbol))
    
    # 【修复 1】解决 IndexError: 严格检测数据饱和度
    if df is None or len(df) < 40:
        st.warning("📡 正在穿透 OKX 数据层，请保持阵地...")
        return

    df = calculate_warrior_signals(df, params)
    last, prev = df.iloc[-1], df.iloc[-2]

    # 看板布局
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ETH 实时价格", f"${last['c']:.2f}", f"{last['c']-prev['c']:.2f}")
    m2.metric("当前量能比", f"{(last['v']/last['ma_v']):.2f}x")
    
    sig_text = {1: "🔥 多头突击", -1: "📉 空头镇压", 0.5: "👀 缩量探底", 0: "💎 蓄势中"}
    m3.metric("战报状态", sig_text.get(last['signal'], "等待中"))
    m4.metric("心跳频率", f"{int(time.time()%60)}s")

    # 【修复 2】交易计划表单 (状态保持)
    with st.expander("📝 进场计划与风险评估", expanded=(last['signal'] != 0)):
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"建议进场: {last['c']} | 止损参考: {last['l'] if last['c']>last['o'] else last['h']}")
        with c2:
            st.button("🎯 确认进场并记录日志", use_container_width=True)

    # 【修复 3】Plotly 性能加固与坐标轴优化
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    
    # K 线图 (WebGL 模式)
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['o'], high=df['h'], low=df['l'], close=df['c'],
        name="Warrior K", increasing_line_color='#00ff00', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)

    # 【新增】右侧穿透价格线
    fig.add_hline(y=last['c'], line_dash="dash", line_color="#d4af37", 
                  annotation_text=f"LIVE: {last['c']}", annotation_position="right")

    # 成交量
    colors = ['#00cc96' if c >= o else '#ef5350' for c, o in zip(df['c'], df['o'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['v'], marker_color=colors, name="Vol"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ma_v'], line=dict(color='orange', width=1.5)), row=2, col=1)

    fig.update_layout(
        height=700, template="plotly_dark", showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=60),
        yaxis=dict(side="right", tickformat=".2f") # 符合顶级交易员习惯的右侧轴
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    st.sidebar.title("⚔️ Warrior 控制中心")
    st.session_state.symbol = st.sidebar.text_input("交易对", "ETH-USDT-SWAP")
    
    with st.sidebar.expander("策略心法参数", expanded=True):
        st.session_state.params = {
            "v_len": st.number_input("均量周期", 5, 60, 10),
            "s_ratio": st.slider("缩量阈值 (%)", 30, 90, 60) / 100,
            "e_ratio": st.slider("放量倍数", 1.2, 3.0, 1.5),
            "b_min": st.slider("实体占比", 0.0, 0.5, 0.2)
        }
    
    st.sidebar.divider()
    st.sidebar.success("系统状态：战斗准备就绪")
    
    live_dashboard()

if __name__ == "__main__":
    main()
