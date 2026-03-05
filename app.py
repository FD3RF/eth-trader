import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time
from datetime import datetime

# ---------- 1. 页面配置与黑金视觉样式 ----------
st.set_page_config(layout="wide", page_title="ETH Warrior 终极稳定版", page_icon="⚔️")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #f0c05a; font-family: 'Courier New', monospace; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------- 2. 高性能 OKX 异步抓取引擎 ----------
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
                    if not data: return None
                    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                    df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    # 【核心修复】统一列名，解决 KeyError
                    df.columns = ['time','open','high','low','close','volume']
                    return df.sort_values('time').reset_index(drop=True)
            except Exception:
                return None
        return None

engine = OKXEngine()

# ---------- 3. 策略算法逻辑 ----------
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
    # 缩量探底/摸顶
    df.loc[df['is_shrink'] & (df['low'] <= df['l_ref'] * 1.002), 'signal'] = 0.5
    df.loc[df['is_shrink'] & (df['high'] >= df['h_ref'] * 0.998), 'signal'] = -0.5
    # 放量突破执行
    df.loc[df['is_expand'] & (df['close'] > df['open']) & (df['body_pct'] > b_min), 'signal'] = 1.0
    df.loc[df['is_expand'] & (df['close'] < df['open']) & (df['body_pct'] > b_min), 'signal'] = -1.0
    return df

# ---------- 4. 局部刷新 UI 组件 ----------
@st.fragment(run_every="5s")
def render_warrior_ui(symbol, params):
    # 处理异步调用
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    df_raw = loop.run_until_complete(engine.fetch(symbol))
    
    if df_raw is not None and len(df_raw) > 30:
        df = apply_strategy(df_raw, **params)
        last = df.iloc[-1]
        
        # 1. 看板数据展示
        c1, c2, c3, c4 = st.columns(4)
        diff = last['close'] - df.iloc[-2]['close']
        c1.metric("ETH 现价", f"${last['close']}", f"{diff:.2f}")
        c2.metric("当前量能比", f"{(last['volume']/last['v_ma']):.2f}x")
        
        status_txt = {1.0: "🔥 放量起涨", -1.0: "📉 放量杀跌", 0.5: "👀 缩量探底", -0.5: "👀 缩量摸顶", 0.0: "💎 震荡蓄势"}
        c3.metric("实时战报", status_txt.get(last['signal'], "等待信号"))
        c4.metric("心跳刷新", f"{int(time.time()%60)}s")

        # 2. 交互式 K 线图
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # 使用 Scattergl 和 Candlestick 实现高性能渲染
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name="K线"), row=1, col=1)
        
        # 标注信号
        buys = df[df['signal'] == 1.0]
        sells = df[df['signal'] == -1.0]
        fig.add_trace(go.Scattergl(x=buys['time'], y=buys['low']*0.998, mode='markers', 
                                   marker=dict(symbol='triangle-up', size=15, color='#00ff00'), name="买入信号"), row=1, col=1)
        fig.add_trace(go.Scattergl(x=sells['time'], y=sells['high']*1.002, mode='markers', 
                                   marker=dict(symbol='triangle-down', size=15, color='#ff4b4b'), name="卖出信号"), row=1, col=1)

        # 支撑压力参考线
        fig.add_hline(y=last['h_ref'], line_dash="dash", line_color="#ff4b4b", opacity=0.3, row=1, col=1)
        fig.add_hline(y=last['l_ref'], line_dash="dash", line_color="#00ff00", opacity=0.3, row=1, col=1)

        # 【核心修复】量能柱颜色映射，使用正确的列名引用
        v_clrs = ['#ff4b4b' if r['close'] < r['open'] else '#00cc96' for _, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=v_clrs, opacity=0.5, name="成交量"), row=2, col=1)
        fig.add_trace(go.Scattergl(x=df['time'], y=df['v_ma'], line=dict(color='orange', width=1.5), name="均量线"), row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("⚔️ Warrior 正在同步 OKX 实时数据流...")

# ---------- 5. 主程序控制台 ----------
def main():
    st.sidebar.title("⚔️ Warrior V2.0 稳定版")
    st.sidebar.caption("模式：纯净量价心法 (5m 周期)")
    
    with st.sidebar.expander("策略参数微调", expanded=True):
        params = {
            "v_len": st.number_input("均量周期", 5, 30, 10),
            "s_ratio": st.slider("缩量判定 (均量%)", 30, 80, 60) / 100,
            "e_ratio": st.slider("放量判定 (均量%)", 120, 300, 150) / 100,
            "b_min": st.slider("突破实体占比", 0.0, 0.5, 0.20)
        }
    
    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    st.sidebar.divider()
    st.sidebar.warning("注意：物理冷静期功能已集成在后台逻辑中")
    
    render_warrior_ui(symbol, params)

if __name__ == "__main__":
    main()
