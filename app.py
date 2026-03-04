import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time

st.set_page_config(page_title="OKX ETH 永续 5m K线", layout="wide")

st.title("OKX ETH 永续合约 5分钟 K 线")

# ====== 配置 ======
symbol = "ETH-USDT-SWAP"
bar = "5m"
limit = 100

# ====== 拉取 OKX 公共 API ======
def fetch_candles():
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    resp = requests.get(url)
    data = resp.json()
    return data.get("data")

data = fetch_candles()

if data:
    df = pd.DataFrame(data, columns=[
        "ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"
    ])

    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
    df = df.sort_values("ts")

    df[["o", "h", "l", "c"]] = df[["o", "h", "l", "c"]].astype(float)

    st.subheader("最近K线数据（部分）")
    st.dataframe(df.tail())

    fig = go.Figure(data=[
        go.Candlestick(
            x=df["ts"],
            open=df["o"],
            high=df["h"],
            low=df["l"],
            close=df["c"]
        )
    ])

    fig.update_layout(
        title="ETH 永续 5 分钟 K 线",
        xaxis_title="时间",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("无法获取数据，请检查 OKX API 或网络。")

# ====== 自动刷新（可选）======
refresh = st.checkbox("自动刷新（每60秒）")
if refresh:
    time.sleep(60)
    st.experimental_rerun()
