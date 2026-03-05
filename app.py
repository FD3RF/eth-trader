import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime

# ==========================================
# 1. 页面初始化
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Warrior Sniper V6.2",
    page_icon="⚔️"
)

# 初始化 HTTP Client（持久连接减少延迟）
if "http_client" not in st.session_state:
    st.session_state.http_client = httpx.Client(
        timeout=httpx.Timeout(15.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        follow_redirects=True
    )

# ==========================================
# 2. CSS (抗闪烁 + UI 稳定)
# ==========================================
st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    max-width:100%;
}

/* Metric 卡片固定高度 */
[data-testid="stMetric"]{
    background:#0e1117;
    border:1px solid #1f2937;
    border-radius:12px;
    padding:12px;
    min-height:90px;
}

/* 状态卡 */
.status-card{
    background:#111827;
    border-radius:12px;
    border:1px solid #1f2937;
    padding:16px;
}

/* Plotly 防抖 */
.js-plotly-plot{
    transition:opacity 0.25s ease;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 策略核心逻辑（完全保持）
# ==========================================
def apply_warrior_logic(df, p):

    df = df.dropna().reset_index(drop=True)

    df["ma_v"] = df["v"].rolling(p["ma_len"]).mean()

    df["body_size"] = abs(df["c"] - df["o"])
    df["total_size"] = (df["h"] - df["l"]).replace(0, 0.001)

    df["body_ratio"] = df["body_size"] / df["total_size"]
    df["vol_ratio"] = df["v"] / df["ma_v"].replace(0, 1e-9)

    df["is_expand"] = df["v"] > df["ma_v"] * (p["expand_p"] / 100)

    df["buy_sig"] = (
        df["is_expand"]
        & (df["c"] > df["o"])
        & (df["body_ratio"] > p["body_r"])
    )

    df["sell_sig"] = (
        df["is_expand"]
        & (df["c"] < df["o"])
        & (df["body_ratio"] > p["body_r"])
    )

    window = df.tail(30)

    v_max_down = window[window["c"] < window["o"]]
    v_max_up = window[window["c"] > window["o"]]

    anchors = {

        "upper":
        v_max_down.nlargest(1,"v")["h"].values[0]
        if not v_max_down.empty
        else window["h"].max(),

        "lower":
        v_max_up.nlargest(1,"v")["l"].values[0]
        if not v_max_up.empty
        else window["l"].min()

    }

    return df, anchors


# ==========================================
# 4. 主界面
# ==========================================
def main():

    st.sidebar.title("⚔️ Warrior Sniper")

    with st.sidebar.expander("🏹 狙击核心校准", expanded=True):

        ma_len = st.number_input("均量周期",5,50,10)

        expand_p = st.slider("放量判定 (%)",100,300,150)

        body_r = st.slider("实体比率",0.05,0.90,0.20)

        rr_ratio = st.slider("盈亏比 (1:X)",1.0,5.0,1.5,step=0.1)

        params = {
            "ma_len":ma_len,
            "expand_p":expand_p,
            "body_r":body_r,
            "rr_ratio":rr_ratio
        }

    symbol = st.sidebar.text_input("合约代码","ETH-USDT-SWAP")

    header_area = st.empty()
    metric_area = st.empty()
    chart_area = st.empty()

# ==========================================
# 5. 实时刷新
# ==========================================
    @st.fragment(run_every="10s")
    def update_dashboard():

        try:

            url = "https://www.okx.com/api/v5/market/candles"

            r = st.session_state.http_client.get(
                url,
                params={
                    "instId":symbol,
                    "bar":"5m",
                    "limit":"100"
                }
            )

            if r.status_code != 200:
                return

            data = r.json()["data"]

            df = pd.DataFrame(
                data,
                columns=[
                    "ts","o","h","l","c","v",
                    "volCcy","volCcyQuote","confirm"
                ]
            )

            for col in ["o","h","l","c","v"]:
                df[col] = pd.to_numeric(df[col])

            df["time"] = pd.to_datetime(df["ts"].astype(int), unit="ms")

            # OKX 返回倒序 → 修复
            df = df.sort_values("time").reset_index(drop=True)

            df, anchors = apply_warrior_logic(df, params)

            curr = df.iloc[-1]

            upper = anchors["upper"]
            lower = anchors["lower"]

# ==========================================
# 战报
# ==========================================
            with header_area.container():

                if curr["buy_sig"] or curr["c"] > upper:

                    status = "🚀 多头总攻"
                    color = "#10b981"

                elif curr["sell_sig"] or curr["c"] < lower:

                    status = "❄️ 空头突袭"
                    color = "#ef4444"

                else:

                    status = "💎 窄幅震荡"
                    color = "#3b82f6"

                st.markdown(
                    f"""
                    <div class="status-card"
                    style="border-left:8px solid {color};">

                    <h2 style="color:{color};margin:0;">
                    {status} | ETH: ${curr["c"]:.2f}
                    </h2>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ==========================================
# 指标
# ==========================================
            with metric_area.container():

                c1,c2,c3,c4 = st.columns(4)

                c1.metric("当前现价",f"${curr['c']:.2f}")

                c2.metric("放量系数",f"{curr['vol_ratio']:.2f}x")

                c3.metric("多头锚点",f"${lower:.2f}")

                c4.metric("空头锚点",f"${upper:.2f}")

# ==========================================
# 图表
# ==========================================
            with chart_area.container():

                df_p = df.tail(60)

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[0.75,0.25],
                    vertical_spacing=0.02
                )

                fig.add_trace(
                    go.Candlestick(
                        x=df_p["time"],
                        open=df_p["o"],
                        high=df_p["h"],
                        low=df_p["l"],
                        close=df_p["c"],
                        name="K"
                    ),
                    row=1,
                    col=1
                )

                buys = df_p[df_p["buy_sig"]]
                sells = df_p[df_p["sell_sig"]]

                fig.add_trace(
                    go.Scatter(
                        x=buys["time"],
                        y=buys["l"]*0.998,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up",
                            size=14,
                            color="#10b981"
                        )
                    ),
                    row=1,col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=sells["time"],
                        y=sells["h"]*1.002,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=14,
                            color="#ef4444"
                        )
                    ),
                    row=1,col=1
                )

                fig.add_hline(
                    y=upper,
                    line_dash="dash",
                    line_color="#ef4444"
                )

                fig.add_hline(
                    y=lower,
                    line_dash="dash",
                    line_color="#10b981"
                )

                v_colors = [
                    "#10b981" if c>=o else "#ef4444"
                    for c,o in zip(df_p["c"],df_p["o"])
                ]

                fig.add_trace(
                    go.Bar(
                        x=df_p["time"],
                        y=df_p["v"],
                        marker_color=v_colors,
                        opacity=0.6
                    ),
                    row=2,col=1
                )

                fig.update_layout(

                    height=600,

                    template="plotly_dark",

                    showlegend=False,

                    xaxis_rangeslider_visible=False,

                    margin=dict(t=0,b=0,l=10,r=30),

                    hovermode="x unified",

                    uirevision="constant"  # 防止刷新重绘
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar":False}
                )

        except Exception:
            pass

    update_dashboard()


if __name__ == "__main__":
    main()
