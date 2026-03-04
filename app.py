import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ===== 配置（实盘需填）=====
BASE_URL = "https://www.okx.com"
# 实盘：API_KEY / SECRET / PASSPHRASE

st.set_page_config(layout="wide", page_title="手动K线交易看盘", page_icon="📈")
st.title("📊 手动看盘 · 无策略框架")

# ===== 获取K线 =====
@st.cache_data(ttl=30)
def get_candles(limit=200, bar="15m"):
    try:
        url = f"{BASE_URL}/api/v5/market/candles"
        res = requests.get(
            url,
            params={"instId": "ETH-USDT", "limit": limit, "bar": bar},
            timeout=6
        ).json()

        if res.get("code") != "0":
            st.warning(res.get("msg", "K线获取失败"))
            return pd.DataFrame()

        df = pd.DataFrame(
            res["data"],
            columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"]
        )[::-1]

        df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        return df

    except Exception as e:
        st.error(f"获取失败: {e}")
        return pd.DataFrame()


# ===== 下单占位 =====
def place_order(side, size):
    st.info(f"模拟下单：{side} {size} ETH（未实盘）")
    # 实盘：调用 OKX 下单接口并签名
    return True


# ===== 主界面 =====
def main():
    df = get_candles()
    if df.empty:
        st.error("暂无K线数据")
        return

    # K线图
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"],
        name="K线"
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 手动操作
    st.subheader("✋ 手动操作")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 买入（示例）"):
            place_order("buy", 0.01)
    with col2:
        if st.button("📉 卖出（示例）"):
            place_order("sell", 0.01)

    st.info(
        "本框架仅看盘与模拟下单，**无自动策略**。\n"
        "实盘需接入 API 并自行承担风险。"
    )


if __name__ == "__main__":
    main()
