import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime

# ==========================================
# 页面初始化
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Warrior Sniper V6.2",
    page_icon="⚔️"
)

# ==========================================
# HTTP Client（稳定连接池）
# ==========================================
if "http_client" not in st.session_state:
    st.session_state.http_client = httpx.Client(
        timeout=httpx.Timeout(10.0),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10
        ),
        follow_redirects=True
    )

# ==========================================
# CSS（精细化 UI + 图表美化）
# ==========================================
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    max-width:100%;
}

/* 指标卡：现代化轻质感 */
[data-testid="stMetric"]{
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 14px;
    min-height: 92px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

/* 状态卡：强调左边缘 */
.status-card{
    background:#111827;
    border-radius:14px;
    border:1px solid #1f2937;
    padding:16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

/* Plotly 过渡与圆角 */
.js-plotly-plot{
    transition:opacity .2s ease;
    border-radius:14px;
}

/* 信息框美化 */
div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 数据获取
# ==========================================
@st.cache_data(ttl=8)
def fetch_data(symbol):

    try:
        r = st.session_state.http_client.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": symbol, "bar": "5m", "limit": "120"}
        )
    except Exception:
        return None

    if r.status_code != 200:
        return None

    payload = r.json()
    if not isinstance(payload, dict) or "data" not in payload:
        return None

    data = payload["data"]
    if not data:
        return None

    try:
        df = pd.DataFrame(
            data,
            columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"]
        )
    except Exception:
        return None

    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["ts"], unit="ms", errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# ==========================================
# 策略核心（量价 + 缩量回踩）
# ==========================================
def apply_warrior_logic(df, p):

    df = df.copy().dropna().reset_index(drop=True)

    # 均量与放量
    df["ma_v"] = df["v"].rolling(p["ma_len"], min_periods=1).mean()
    df["is_expand"] = df["v"] > df["ma_v"] * (p["expand_p"] / 100)

    # 实体比率
    df["body_size"] = abs(df["c"] - df["o"])
    df["total_size"] = (df["h"] - df["l"]).replace(0, 1e-9)
    df["body_ratio"] = df["body_size"] / df["total_size"]

    # 缩量回踩判定
    df["is_shrink"] = df["v"] < df["ma_v"] * 0.8
    df["is_pinbar"] = (df["l"] < df["o"]) & (df["l"] < df["c"]) & ((df["c"] - df["l"]) > (df["h"] - df["c"]))

    # 做多信号：缩量 + 企稳 + 下影
    df["buy_sig"] = (
        df["is_shrink"] &
        df["is_pinbar"] &
        (df["c"] > df["o"])
    )

    # 做空信号：放量下跌
    df["sell_sig"] = (
        df["is_expand"] &
        (df["c"] < df["o"]) &
        (df["body_ratio"] > p["body_r"])
    )

    # 锚点：30周期高低与放量节点
    window = df.tail(30)
    upper = window["h"].max()
    lower = window["l"].min()

    down = window[window["c"] < window["o"]]
    up = window[window["c"] > window["o"]]

    if not down.empty:
        upper = down.loc[down["v"].idxmax(), "h"]

    if not up.empty:
        lower = up.loc[up["v"].idxmax(), "l"]

    return df, dict(upper=upper, lower=lower)


# ==========================================
# 语音与战报（口诀对应）
# ==========================================
def build_report(curr, upper, lower):

    # 多头突破
    if curr["c"] > upper:
        return {
            "status": "🚀 多头总攻",
            "voice": "放量突破前高，多头总攻，考虑做多"
        }

    # 空头跌破
    if curr["c"] < lower:
        return {
            "status": "❄️ 空头突袭",
            "voice": "跌破低点，空头突袭，观望或反手"
        }

    # 缩量回踩企稳（口诀核心）
    if curr.get("buy_sig", False):
        return {
            "status": "🟢 缩量回踩企稳",
            "voice": "缩量回踩低点不破，下影企稳，准备动手做多"
        }

    # 放量下跌
    if curr.get("sell_sig", False):
        return {
            "status": "🔴 放量下跌",
            "voice": "放量下跌，空头占优，暂不做多"
        }

    return {
        "status": "💎 窄幅震荡",
        "voice": "窄幅震荡，无方向，等待缩量回踩"
    }


# ==========================================
# 主界面
# ==========================================
def main():

    st.sidebar.title("⚔️ Warrior Sniper")

    with st.sidebar.expander("🏹 狙击核心校准", expanded=True):

        ma_len = st.number_input("均量周期", 5, 50, 10)
        expand_p = st.slider("放量判定 (%)", 150, 300, 200)
        body_r = st.slider("实体比率", 0.05, 0.90, 0.20)
        rr_ratio = st.slider("盈亏比 (1:X)", 1.0, 5.0, 1.5, step=0.1)

        params = dict(
            ma_len=ma_len,
            expand_p=expand_p,
            body_r=body_r,
            rr_ratio=rr_ratio
        )

    symbol = st.sidebar.text_input("合约代码", "ETH-USDT-SWAP")

    header_area = st.empty()
    metric_area = st.empty()
    chart_area = st.empty()
    voice_area = st.empty()

    # ==========================================
    # 实时刷新
    # ==========================================
    @st.fragment(run_every="10s")
    def update_dashboard():

        try:
            df = fetch_data(symbol)
            if df is None or df.empty:
                return

            df, anchors = apply_warrior_logic(df, params)
            curr = df.iloc[-1]

            upper = anchors["upper"]
            lower = anchors["lower"]

            report = build_report(curr, upper, lower)
            status = report["status"]
            voice = report["voice"]

            # 战报头
            with header_area.container():
                color = "#10b981" if "多头" in status else "#ef4444" if "空头" in status else "#3b82f6"

                st.markdown(f"""
                <div class="status-card"
                style="border-left:8px solid {color};">

                <h2 style="color:{color};margin:0;">
                {status} | 现价: ${curr["c"]:.2f}
                </h2>

                </div>
                """, unsafe_allow_html=True)

            # 语音战报
            with voice_area.container():
                st.info(f"语音战报：{voice}")

            # 指标卡
            with metric_area.container():
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("当前现价", f"${curr['c']:.2f}")
                c2.metric("放量系数", f"{(curr['v']/curr['ma_v'] if curr['ma_v']>0 else 0):.2f}x")
                c3.metric("多头锚点", f"${lower:.2f}")
                c4.metric("空头锚点", f"${upper:.2f}")

            # 图表
            with chart_area.container():

                df_p = df.tail(60)

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    vertical_spacing=0.02
                )

                fig.add_trace(
                    go.Candlestick(
                        x=df_p["time"],
                        open=df_p["o"],
                        high=df_p["h"],
                        low=df_p["l"],
                        close=df_p["c"]
                    ),
                    row=1, col=1
                )

                buys = df_p[df_p["buy_sig"]]
                sells = df_p[df_p["sell_sig"]]

                fig.add_trace(
                    go.Scatter(
                        x=buys["time"],
                        y=buys["l"] * 0.998,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=14)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=sells["time"],
                        y=sells["h"] * 1.002,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=14)
                    ),
                    row=1, col=1
                )

                fig.add_hline(y=upper, line_dash="dash")
                fig.add_hline(y=lower, line_dash="dash")

                fig.add_trace(
                    go.Bar(
                        x=df_p["time"],
                        y=df_p["v"],
                        opacity=0.6
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    height=620,
                    template="plotly_dark",
                    showlegend=False,
                    xaxis_rangeslider_visible=False,
                    margin=dict(t=10, b=10, l=10, r=30),
                    hovermode="x unified",
                    uirevision="chart-lock"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        except Exception as e:
            st.error(f"数据更新异常: {e}")

    update_dashboard()


if __name__ == "__main__":
    main()
