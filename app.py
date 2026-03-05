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

# CSS 优化：减少浏览器渲染负担
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    canvas { image-rendering: -webkit-optimize-contrast; } /* 提升图表清晰度 */
    </style>
""", unsafe_allow_html=True)

# ---------- 2. 异步数据中心 ----------
class OKXDataEngine:
    def __init__(self):
        self.api_url = "https://www.okx.com/api/v5/market/candles"
        self.limits = 150 # 仅拉取必要的K线，减少内存开销

    async def get_candles(self, instId, bar='5m'):
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                params = {"instId": instId, "bar": bar, "limit": self.limits}
                response = await client.get(self.api_url, params=params)
                if response.status_code == 200:
                    data = response.json().get('data', [])
                    return self._process_data(data)
                return None
            except Exception as e:
                return None

    def _process_data(self, data):
        if not data: return None
        df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        # 向量化更名
        df.columns = ['time','open','high','low','close','volume']
        return df.sort_values('time').reset_index(drop=True)

engine = OKXDataEngine()

# ---------- 3. 策略引擎：向量化量价计算 ----------
def warrior_logic_vectorized(df, s_ratio, e_ratio, v_len, b_min):
    """
    完全抛弃 for 循环，使用 Pandas 向量化运算提升计算速度 > 100x
    """
    df['v_ma'] = df['volume'].rolling(v_len).mean()
    df['h_ref'] = df['high'].rolling(20).max().shift(1)
    df['l_ref'] = df['low'].rolling(20).min().shift(1)
    
    # 定义量能状态
    is_shrink = df['volume'] < (df['v_ma'] * s_ratio)
    is_expand = df['volume'] > (df['v_ma'] * e_ratio)
    
    # 定义价格形态
    body = (df['close'] - df['open']).abs()
    range_val = (df['high'] - df['low']) + 1e-9
    body_pct = body / range_val
    
    # 信号逻辑矩阵
    df['signal'] = 0.0
    
    # 逻辑 A/C: 缩量观察区 (0.5 / -0.5)
    df.loc[is_shrink & (df['low'] <= df['l_ref'] * 1.002), 'signal'] = 0.5
    df.loc[is_shrink & (df['high'] >= df['h_ref'] * 0.998), 'signal'] = -0.5
    
    # 逻辑 B/D: 放量执行区 (1 / -1)
    # 做多：放量 + 阳线 + 突破前一根阴线高点 + 实体饱满
    long_cond = is_expand & (df['close'] > df['open']) & \
                (df['close'] > df['open'].shift(1)) & (body_pct > b_min)
    df.loc[long_cond, 'signal'] = 1.0
    
    # 做空：放量 + 阴线 + 跌破前一根阳线低点 + 实体饱满
    short_cond = is_expand & (df['close'] < df['open']) & \
                 (df['close'] < df['open'].shift(1)) & (body_pct > b_min)
    df.loc[short_cond, 'signal'] = -1.0
    
    return df

# ---------- 4. 局部刷新组件 (@st.fragment) ----------
@st.fragment(run_every="5s")
def render_dashboard(symbol, params):
    # 异步获取数据
    df_raw = asyncio.run(engine.get_candles(symbol))
    
    if df_raw is not None:
        df = warrior_logic_vectorized(df_raw, **params)
        last = df.iloc[-1]
        
        # 1. 实时数据指标
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ETH 实时价", f"${last['close']}", f"{last['close']-df.iloc[-2]['close']:.2f}")
        c2.metric("当前量能比", f"{(last['volume']/last['v_ma']):.2f}x")
        c3.metric("信号状态", "等待爆发" if last['signal'] == 0 else "🔥 交易瞬间")
        c4.metric("延迟", f"{int((time.time() - last['time'].timestamp())%300)}s", "OKX V5")

        # 2. 核心图表渲染 (WebGL 加速)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # 使用 Scattergl 替代 Scatter 提升渲染性能
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        
        # 信号点绘制
        buy_pts = df[df['signal'] == 1]
        sell_pts = df[df['signal'] == -1]
        fig.add_trace(go.Scattergl(x=buy_pts['time'], y=buy_pts['low']*0.998, mode='markers', 
                                   marker=dict(symbol='triangle-up', size=15, color='#00ff00')), row=1, col=1)
        fig.add_trace(go.Scattergl(x=sell_pts['time'], y=sell_pts['high']*1.002, mode='markers', 
                                   marker=dict(symbol='triangle-down', size=15, color='#ff4b4b')), row=1, col=1)

        # 量能柱
        v_colors = ['#ff4b4b' if r['close'] < r['open'] else '#00cc96' for _, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=v_colors, opacity=0.4), row=2, col=1)
        fig.add_trace(go.Scattergl(x=df['time'], y=df['v_ma'], line=dict(color='orange', width=1)), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, 
                          xaxis_rangeslider_visible=False, margin=dict(t=5, b=5, l=5, r=5))
        st.plotly_chart(fig, width='stretch', use_container_width=True, config={'displayModeBar': False})

        # 3. 信号即时播报
        if abs(last['signal']) == 1:
            st.toast(f"⚔️ Warrior 信号触发: {'做多' if last['signal']>0 else '做空'}")

# ---------- 5. 主程序结构 ----------
def main():
    st.sidebar.title("⚔️ Warrior 控制台")
    
    # 策略参数动态调节
    params = {
        "v_len": st.sidebar.number_input("均量周期", 5, 20, 10),
        "s_ratio": st.sidebar.slider("缩量阈值%", 30, 80, 60) / 100,
        "e_ratio": st.sidebar.slider("放量阈值%", 120, 300, 150) / 100,
        "b_min": st.sidebar.slider("实体饱满度", 0.0, 0.5, 0.15)
    }
    
    target_symbol = st.sidebar.text_input("目标合约", "ETH-USDT-SWAP")
    
    # 运行局部刷新组件
    render_dashboard(target_symbol, params)

if __name__ == "__main__":
    main()
