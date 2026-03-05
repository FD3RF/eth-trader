import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import asyncio
import time

# -----------------------------
# 1. 页面配置：黑金视觉
# -----------------------------
st.set_page_config(layout="wide", page_title="ETH Warrior", page_icon="⚔️")

st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #f0c05a; font-family: monospace; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. OKX 数据引擎：增强容错
# -----------------------------
class OKXEngine:
    def __init__(self):
        self.url = "https://www.okx.com/api/v5/market/candles"

    async def fetch(self, instId):
        # 使用 follow_redirects=True 确保稳定性
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            try:
                params = {"instId": instId, "bar": "5m", "limit": "150"}
                resp = await client.get(self.url, params=params)
                if resp.status_code != 200: return None
                
                data = resp.json().get("data", [])
                if not data: return None

                # 快速清洗
                df = pd.DataFrame(data).iloc[:, 0:6]
                df.columns = ["ts", "open", "high", "low", "close", "volume"]
                df = df.apply(pd.to_numeric) # 一键转换所有列
                df["time"] = pd.to_datetime(df["ts"], unit="ms")
                return df.sort_values("time").reset_index(drop=True)
            except Exception:
                return None

engine = OKXEngine()

# -----------------------------
# 3. 策略算法：不灭大衍逻辑
# -----------------------------
def apply_strategy(df, s_ratio, e_ratio, v_len, b_min):
    df = df.copy()
    df["v_ma"] = df["volume"].rolling(v_len).mean()
    df["h_ref"] = df["high"].rolling(20).max().shift(1)
    df["l_ref"] = df["low"].rolling(20).min().shift(1)

    df["is_shrink"] = df["volume"] < df["v_ma"] * s_ratio
    df["is_expand"] = df["volume"] > df["v_ma"] * e_ratio

    body = (df["close"] - df["open"]).abs()
    range_val = (df["high"] - df["low"]) + 1e-9
    df["body_pct"] = body / range_val

    df["signal"] = 0.0
    # 缩量探底 (0.5) / 缩量摸顶 (-0.5)
    df.loc[(df["is_shrink"]) & (df["low"] <= df["l_ref"] * 1.002), "signal"] = 0.5
    df.loc[(df["is_shrink"]) & (df["high"] >= df["h_ref"] * 0.998), "signal"] = -0.5
    # 放量突破起涨 (1.0) / 杀跌 (-1.0)
    df.loc[(df["is_expand"]) & (df["close"] > df["open"]) & (df["body_pct"] > b_min), "signal"] = 1.0
    df.loc[(df["is_expand"]) & (df["close"] < df["open"]) & (df["body_pct"] > b_min), "signal"] = -1.0
    return df

# -----------------------------
# 4. 局部刷新 UI
# -----------------------------
@st.fragment(run_every="5s")
def render(symbol, params):
    # 异步抓取数据
    try:
        df_raw = asyncio.run(engine.fetch(symbol))
    except Exception:
        df_raw = None

    if df_raw is None or len(df_raw) < 30:
        st.warning("⚔️ 战士正在同步 OKX 实时数据流，或检测合约代码是否有误...")
        return

    df = apply_strategy(df_raw, **params)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ---- 看板指标
    c1, c2, c3, c4 = st.columns(4)
    diff = last["close"] - prev["close"]
    c1.metric("ETH 现价", f"${last['close']:.2f}", f"{diff:.2f}")
    c2.metric("量能比", f"{(last['volume'] / last['v_ma']):.2f}x")
    
    signal_map = {1.0: "🔥 放量起涨", -1.0: "📉 放量杀跌", 0.5: "👀 缩量探底", -0.5: "👀 缩量摸顶", 0.0: "💎 蓄势中"}
    c3.metric("实时战报", signal_map.get(last["signal"], "💎 震荡蓄势"))
    c4.metric("刷新心跳", f"{int(time.time() % 60)}s")

    # ---- 高级绘图 (使用 Scattergl 提升性能)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

    # 主 K 线
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="K线", increasing_line_color='#00cc96', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)

    # 信号标注
    buy = df[df["signal"] == 1.0]
    sell = df[df["signal"] == -1.0]
    fig.add_trace(go.Scattergl(x=buy["time"], y=buy["low"] * 0.998, mode="markers", 
                               marker=dict(symbol="triangle-up", size=14, color="#00ff00"), name="买入信号"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=sell["time"], y=sell["high"] * 1.002, mode="markers", 
                               marker=dict(symbol="triangle-down", size=14, color="#ff4b4b"), name="卖出信号"), row=1, col=1)

    # 成交量
    clrs = ['#00cc96' if c >= o else '#ff4b4b' for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["time"], y=df["volume"], marker_color=clrs, name="成交量", opacity=0.8), row=2, col=1)
    fig.add_trace(go.Scattergl(x=df["time"], y=df["v_ma"], line=dict(color="orange", width=1.5), name="均量线"), row=2, col=1)

    fig.update_layout(height=650, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# -----------------------------
# 5. 主程序
# -----------------------------
def main():
    st.sidebar.title("⚔️ Warrior 控制台")
    
    with st.sidebar.expander("策略心法调节", expanded=True):
        v_len = st.slider("均量参考周期", 5, 30, 10)
        s_ratio = st.slider("缩量判定 (均量%)", 30, 80, 60) / 100
        e_ratio = st.slider("放量判定 (均量%)", 120, 300, 150) / 100
        b_min = st.slider("突破实体比例", 0.0, 0.5, 0.2)

    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")
    st.sidebar.info("Warrior V2.0 · 纯净量价版")
    
    # 启动渲染
    render(symbol, {"v_len": v_len, "s_ratio": s_ratio, "e_ratio": e_ratio, "b_min": b_min})

if __name__ == "__main__":
    main()
