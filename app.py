import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ===== OKX 配置（实盘需填）=====
API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"
PASSPHRASE = "YOUR_PASSPHRASE"
BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="手动K线交易看盘", page_icon="📈")


# ===== 获取K线 =====
def get_candles(limit=200, bar="15m"):
    try:
        url = f"{BASE_URL}/api/v5/market/candles"
        res = requests.get(url, params={"instId": "ETH-USDT", "limit": limit, "bar": bar}, timeout=6).json()
        if res.get("code") != "0":
            return pd.DataFrame()

        df = pd.DataFrame(res["data"], columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"])[::-1]
        df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
        for c in ["o","h","l","c","v"]:
            df[c] = df[c].astype(float)
        return df
    except:
        return pd.DataFrame()


# ===== 下单示例（实盘需签名）=====
def place_order(side, size):
    st.info(f"模拟下单：{side} {size} ETH（未实盘）")
    # 实盘需实现 OKX 下单签名与 API
    return True


# ===== UI =====
def main():
    st.title("📊 手动K线看盘交易")

    df = get_candles()
    if df.empty:
        st.error("K线获取失败")
        return

    # K线图
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["o"],
        high=df["h"],
        low=df["l"],
        close=df["c"]
    )])
    st.plotly_chart(fig, use_container_width=True)

    # 手动下单区
    st.subheader("✋ 手动下单")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 买入（示例）"):
            place_order("buy", 0.01)
    with col2:
        if st.button("📉 卖出（示例）"):
            place_order("sell", 0.01)

    st.info("本框架**不含自动策略**，仅用于看盘与手动操作。实盘需自行接 OKX API 并遵守风控。")


if __name__ == "__main__":
    main()
