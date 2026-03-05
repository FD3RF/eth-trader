import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time

# ---------- 1. 页面配置与黑金视觉样式 ----------
st.set_page_config(layout="wide", page_title="ETH Warrior 稳定版", page_icon="⚔️")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #f0c05a; font-family: monospace; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------- 2. OKX 数据引擎 (带异常缓冲) ----------
class OKXEngine:
    def __init__(self):
        self.url = "https://www.okx.com/api/v5/market/candles"

    async def fetch(self, instId):
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            try:
                params = {"instId": instId, "bar": "5m", "limit": "150"}
                resp = await client.get(self.url, params=params)
                if resp.status_code == 200:
                    data = resp.json().get('data', [])
                    if not data or len(data) < 20: return None
                    
                    # 规范化列名与类型
                    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].apply(pd.to_numeric)
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    df = df.rename(columns={'ts':'time','o':'open','h':'high','l':'low','c':'close','v':'volume'})
                    return df.sort_values('time').reset_index(drop=True)
            except Exception:
                return None
        return None

engine = OKXEngine()

# ---------- 3. 策略逻辑模块 ----------
def apply_strategy(df, s_ratio, e_ratio, v_len, b_min):
    df = df.copy()
    df['v_ma'] = df['volume'].rolling(v_len).mean()
    df['h_ref'] = df['high'].rolling(20).max().shift(1)
    df['l_ref'] = df['low'].rolling(20).min().shift(1)
    
    # 判定逻辑
    df['is_shrink'] = df['volume'] < (df['v_ma'] * s_ratio)
    df['is_expand'] = df['volume'] > (df['v_ma'] * e_ratio)
    
    body = (df['close'] - df['open']).abs()
    range_val = (df['high'] - df['low']) + 1e-9
    df['body_pct'] = body / range_val
    
    df['signal'] = 0.0
    df.loc[df['is_shrink'] & (df['low'] <= df['l_ref'] * 1.002), 'signal'] = 0.5
    df.loc[df['is_shrink'] & (df['high'] >= df['h_ref'] * 0.998), 'signal'] = -0.5
    df.loc[df['is_expand'] & (df['close'] > df['open']) & (df['body_pct'] > b_min), 'signal'] = 1.0
    df.loc[df['is_expand'] & (df['close'] < df['open']) & (df['body_pct'] > b_min), 'signal'] = -1.0
    return df

# ---------- 4. 局部刷新 UI 组件 ----------
@st.fragment(run_every="5s")
def render_warrior_ui(symbol, params):
    try:
        # 使用 asyncio.run 简化调用过程
        df_raw = asyncio.run(engine.fetch(symbol))
    except Exception:
        df_raw = None
    
    # 【修复 1】严格的长度检测，防止 IndexError
    if df_raw is not None and len(df_raw) > 30:
        df = apply_strategy(df_raw, **params)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 指标看板
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ETH 现价", f"${last['close']:.2f}", f"{last['close']-prev['close']:.2f}")
        c2.metric("当前量能比", f"{(last['volume']/last['v_ma']):.2f}x")
        
        status_map = {1.0: "🔥 放量起涨", -1.0: "📉 放量杀跌", 0.5: "👀 缩量探底", -0.5: "👀 缩量摸顶", 0.0: "💎 震荡蓄势"}
        c3.metric("实时战报", status_map.get(last['signal'], "等待信号"))
        c4.metric("心跳刷新", f"{int(time.time()%60)}s")

        # 高级交互 K 线图
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        
        # 买卖信号 (Scattergl 加速)
        buys = df[df['signal'] == 1.0]
        sells = df[df['signal'] == -1.0]
        fig.add_trace(go.Scattergl(x=buys['time'], y=buys['low']*0.998, mode='markers', 
                                   marker=dict(symbol='triangle-up', size=15, color='#00ff00'), name="买入"), row=1, col=1)
        fig.add_trace(go.Scattergl(x=sells['time'], y=sells['high']*1.002, mode='markers', 
                                   marker=dict(symbol='triangle-down', size=15, color='#ff4b4b'), name="卖出"), row=1, col=1)

        # 【修复 2】颜色映射逻辑优化，防止 Value Error
        v_clrs = np.where(df['close'] >= df['open'], '#00cc96', '#ff4b4b')
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=v_clrs, opacity=0.5), row=2, col=1)
        fig.add_trace(go.Scattergl(x=df['time'], y=df['v_ma'], line=dict(color='orange', width=1.5)), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("⚔️ Warrior 正在同步数据流，请确保网络连接正常...")

# ---------- 5. 主控制台 ----------
def main():
    st.sidebar.title("⚔️ Warrior 控制中心")
    with st.sidebar.expander("策略参数调节", expanded=True):
        params = {
            "v_len": st.number_input("均量周期", 5, 30, 10),
            "s_ratio": st.slider("缩量判定 (%)", 30, 80, 60) / 100,
            "e_ratio": st.slider("放量判定 (%)", 120, 300, 150) / 100,
            "b_min": st.slider("突破实体比", 0.0, 0.5, 0.20)
        }
    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    st.sidebar.divider()
    st.sidebar.caption("不灭大衍系统 · 纯净量价流")
    render_warrior_ui(symbol, params)

if __name__ == "__main__":
    main()
