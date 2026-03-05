import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx  # 异步请求库
import asyncio
import time
from datetime import datetime

# ---------- 1. 极致性能配置 ----------
st.set_page_config(layout="wide", page_title="ETH Warrior 高性能版", page_icon="⚔️")

# CSS 优化：锁定布局，防止刷新闪烁，提升黑金视觉对比度
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #f0c05a; font-family: 'Courier New', monospace; }
    canvas { image-rendering: -webkit-optimize-contrast; } 
    </style>
""", unsafe_allow_html=True)

# ---------- 2. 异步数据引擎 (OKX V5) ----------
class OKXDataEngine:
    def __init__(self):
        self.api_url = "https://www.okx.com/api/v5/market/candles"
        self.limits = 150 # 仅维护必要的窗口数据，节省内存

    async def get_candles(self, instId, bar='5m'):
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            try:
                params = {"instId": instId, "bar": bar, "limit": self.limits}
                response = await client.get(self.api_url, params=params)
                if response.status_code == 200:
                    data = response.json().get('data', [])
                    return self._process_data(data)
                return None
            except Exception:
                return None

    def _process_data(self, data):
        if not data: return None
        # 批量向量化转换
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.columns = ['time','open','high','low','close','volume']
        return df.sort_values('time').reset_index(drop=True)

engine = OKXDataEngine()

# ---------- 3. 策略内核：全向量化计算 (性能提升 > 100x) ----------
def apply_warrior_vectorized(df, s_ratio, e_ratio, v_len, b_min):
    """摒弃 for 循环，使用矩阵运算直接标记所有信号点"""
    df['v_ma'] = df['volume'].rolling(v_len).mean()
    df['h_ref'] = df['high'].rolling(20).max().shift(1)
    df['l_ref'] = df['low'].rolling(20).min().shift(1)
    
    # 定义量能状态
    is_shrink = df['volume'] < (df['v_ma'] * s_ratio)
    is_expand = df['volume'] > (df['v_ma'] * e_ratio)
    
    # 价格形态计算
    body = (df['close'] - df['open']).abs()
    range_val = (df['high'] - df['low']) + 1e-9
    body_pct = body / range_val
    
    df['signal'] = 0.0
    
    # 状态 A/C: 缩量观察区 (0.5=多头预警, -0.5=空头预警)
    df.loc[is_shrink & (df['low'] <= df['l_ref'] * 1.002), 'signal'] = 0.5
    df.loc[is_shrink & (df['high'] >= df['h_ref'] * 0.998), 'signal'] = -0.5
    
    # 状态 B: 放量起涨 (做多执行)
    long_cond = is_expand & (df['close'] > df['open']) & \
                (df['close'] > df['open'].shift(1)) & (body_pct > b_min)
    df.loc[long_cond, 'signal'] = 1.0
    
    # 状态 D: 放量杀跌 (做空执行)
    short_cond = is_expand & (df['close'] < df['open']) & \
                 (df['close'] < df['open'].shift(1)) & (body_pct > b_min)
    df.loc[short_cond, 'signal'] = -1.0
    
    return df

# ---------- 4. 局部刷新组件 (@st.fragment) ----------
@st.fragment(run_every="5s")
def render_live_monitor(symbol, params):
    df_raw = asyncio.run(engine.get_candles(symbol))
    
    if df_raw is not None:
        df = apply_warrior_vectorized(df_raw, **params)
        last = df.iloc[-1]
        
        # --- 1. 核心看板 ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ETH 现价", f"${last['close']}", f"{last['close']-df.iloc[-2]['close']:.2f}")
        c2.metric("当前量能比", f"{(last['volume']/last['v_ma']):.2f}x")
        
        sig_map = {1.0: "🔥 放量突破(多)", -1.0: "📉 放量跌破(空)", 
                   0.5: "👀 缩量探底", -0.5: "👀 缩量摸顶", 0.0: "💎 震荡蓄势"}
        c3.metric("实时战报", sig_map.get(last['signal']))
        c4.metric("物理心跳", f"{int(time.time() % 60)}s")

        # --- 2. 深度可视化 (WebGL 加速) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # K线 (关闭 RangeSlider 减少前端压力)
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name="K"), row=1, col=1)
        
        # 信号执行点 (Scattergl 渲染更快)
        buy_pts = df[df['signal'] == 1]
        sell_pts = df[df['signal'] == -1]
        fig.add_trace(go.Scattergl(x=buy_pts['time'], y=buy_pts['low']*0.998, mode='markers', 
                                   marker=dict(symbol='triangle-up', size=16, color='#00ff00'), name="多"), row=1, col=1)
        fig.add_trace(go.Scattergl(x=sell_pts['time'], y=sell_pts['high']*1.002, mode='markers', 
                                   marker=dict(symbol='triangle-down', size=16, color='#ff4b4b'), name="空"), row=1, col=1)

        # 支撑/压力动态参考线
        fig.add_hline(y=last['h_ref'], line_dash="dash", line_color="#ff4b4b55", row=1, col=1)
        fig.add_hline(y=last['l_ref'], line_dash="dash", line_color="#00ff0055", row=1, col=1)

        # 量能视图
        v_colors = ['#ff4b4b' if r['close'] < r['open'] else '#00cc96' for _, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=v_colors, opacity=0.4), row=2, col=1)
        fig.add_trace(go.Scattergl(x=df['time'], y=df['v_ma'], line=dict(color='orange', width=1.5)), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, 
                          xaxis_rangeslider_visible=False, margin=dict(t=5, b=5, l=5, r=5))
        st.plotly_chart(fig, width='stretch', use_container_width=True, config={'displayModeBar': False})
    else:
        st.error("数据源异常，请检查网络或代理...")

# ---------- 5. 主流程 ----------
def main():
    st.sidebar.title("⚔️ Warrior V2.0")
    
    with st.sidebar.expander("心法参数调节", expanded=True):
        params = {
            "v_len": st.number_input("均量周期 (MA)", 5, 30, 10),
            "s_ratio": st.slider("缩量判定 (均量x%)", 30, 80, 60) / 100,
            "e_ratio": st.slider("放量判定 (均量x%)", 120, 300, 150) / 100,
            "b_min": st.slider("突破实体饱满度", 0.0, 0.5, 0.20)
        }
    
    symbol = st.sidebar.text_input("监控合约", "ETH-USDT-SWAP")
    st.sidebar.divider()
    st.sidebar.caption("不灭大衍系统运行中...")
    
    render_live_monitor(symbol, params)

if __name__ == "__main__":
    main()
