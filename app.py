import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time
from datetime import datetime

# ---------- 1. 页面配置与黑金样式 ----------
st.set_page_config(layout="wide", page_title="ETH Warrior 终极修复版", page_icon="⚔️")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #f0c05a; }
    </style>
""", unsafe_allow_html=True)

# ---------- 2. 高性能异步引擎 ----------
class OKXEngine:
    def __init__(self):
        self.url = "https://www.okx.com/api/v5/market/candles"

    async def fetch(self, instId):
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            try:
                params = {"instId": instId, "bar": "5m", "limit": "150"}
                resp = await client.get(self.url, params=params)
                if resp.status_code == 200:
                    data = resp.json().get('data', [])
                    if not data: return None
                    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
                    df = df[['ts','o','h','l','c','v']].apply(pd.to_numeric)
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                    df.columns = ['time','open','high','low','close','volume']
                    return df.sort_values('time').reset_index(drop=True)
            except:
                return None
        return None

engine = OKXEngine()

# ---------- 3. 策略逻辑 (向量化修复) ----------
def apply_strategy(df, s_ratio, e_ratio, v_len, b_min):
    df = df.copy()
    df['v_ma'] = df['volume'].rolling(v_len).mean()
    df['h_ref'] = df['high'].rolling(20).max().shift(1)
    df['l_ref'] = df['low'].rolling(20).min().shift(1)
    
    # 量能判定
    df['is_shrink'] = df['volume'] < (df['v_ma'] * s_ratio)
    df['is_expand'] = df['volume'] > (df['v_ma'] * e_ratio)
    
    # 实体占比判定
    body = (df['close'] - df['open']).abs()
    range_val = (df['high'] - df['low']) + 1e-9
    df['body_pct'] = body / range_val
    
    df['signal'] = 0.0
    # 逻辑 A/C: 缩量预警
    df.loc[df['is_shrink'] & (df['low'] <= df['l_ref'] * 1.002), 'signal'] = 0.5
    df.loc[df['is_shrink'] & (df['high'] >= df['h_ref'] * 0.998), 'signal'] = -0.5
    # 逻辑 B/D: 放量突破执行
    df.loc[df['is_expand'] & (df['close'] > df['open']) & (df['close'] > df['open'].shift(1)) & (df['body_pct'] > b_min), 'signal'] = 1.0
    df.loc[df['is_expand'] & (df['close'] < df['open']) & (df['close'] < df['open'].shift(1)) & (df['body_pct'] > b_min), 'signal'] = -1.0
    return df

# ---------- 4. 局部刷新组件 (修复颜色与索引 Bug) ----------
@st.fragment(run_every="5s")
def render_warrior_ui(symbol, params):
    df_raw = asyncio.run(engine.fetch(symbol))
    
    if df_raw is not None and len(df_raw) > 20:
        df = apply_strategy(df_raw, **params)
        last = df.iloc[-1]
        
        # 1. 看板指标
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ETH 现价", f"${last['close']}", f"{last['close']-df.iloc[-2]['close']:.2f}")
        c2.metric("当前量能", f"{(last['volume']/last['v_ma']):.2f}x")
        
        status = {1.0: "🔥 放量起涨", -1.0: "📉 放量杀跌", 0.5: "👀 缩量探底", -0.5: "👀 缩量摸顶", 0.0: "💎 震荡"}
        c3.metric("实时战况", status.get(last['signal'], "等待"))
        c4.metric("刷新心跳", f"{int(time.time()%60)}s")

        # 2. 绘图 (修复 #ff4b4b55 颜色错误)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name="K"), row=1, col=1)
        
        # 执行信号
        buys = df[df['signal'] == 1.0]
        sells = df[df['signal'] == -1.0]
        fig.add_trace(go.Scattergl(x=buys['time'], y=buys['low']*0.998, mode='markers', 
                                   marker=dict(symbol='triangle-up', size=15, color='#00ff00')), row=1, col=1)
        fig.add_trace(go.Scattergl(x=sells['time'], y=sells['high']*1.002, mode='markers', 
                                   marker=dict(symbol='triangle-down', size=15, color='#ff4b4b')), row=1, col=1)

        # 参考线 (修正点：使用 6 位 Hex)
        fig.add_hline(y=last['h_ref'], line_dash="dash", line_color="#ff4b4b", opacity=0.3, row=1, col=1)
        fig.add_hline(y=last['l_ref'], line_dash="dash", line_color="#00ff00", opacity=0.3, row=1, col=1)

        # 量能柱
        v_clrs = ['#ff4b4b' if r['c'] < r['o'] else '#00cc96' for _, r in df_raw.iterrows()]
        fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=v_clrs, opacity=0.5), row=2, col=1)

        fig.update_layout(height=600, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=5, b=5, l=5, r=5))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("⚠️ 正在等待 API 数据响应或数据量不足...")

# ---------- 5. 主程序 ----------
def main():
    st.sidebar.title("⚔️ Warrior V2.0 修补版")
    params = {
        "v_len": st.sidebar.number_input("均量周期", 5, 30, 10),
        "s_ratio": st.sidebar.slider("缩量判定%", 30, 80, 60) / 100,
        "e_ratio": st.sidebar.slider("放量判定%", 120, 300, 150) / 100,
        "b_min": st.sidebar.slider("实体饱满度", 0.0, 0.5, 0.20)
    }
    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    render_warrior_ui(symbol, params)

if __name__ == "__main__":
    main()
