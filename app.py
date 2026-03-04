import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== OKX 配置（实盘需填）=====
API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"
PASSPHRASE = "YOUR_PASSPHRASE"
BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="5分钟ETH合约策略看盘", page_icon="📈")


# ===== 获取K线（带缓存）=====
@st.cache_data(ttl=30)  # 缓存30秒，避免频繁请求
def get_candles(instId="ETH-USDT", bar="5m", limit=200):
    """从OKX获取K线数据"""
    try:
        url = f"{BASE_URL}/api/v5/market/candles"
        res = requests.get(
            url,
            params={"instId": instId, "limit": limit, "bar": bar},
            timeout=6
        ).json()
        if res.get("code") != "0":
            return pd.DataFrame()

        df = pd.DataFrame(
            res["data"],
            columns=["ts", "o", "h", "l", "c", "v", "volCcy", "volCcyQuote", "confirm"]
        )[::-1]  # 倒序变正序
        df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"获取K线失败: {e}")
        return pd.DataFrame()


# ===== 策略计算 =====
def calculate_signals(df, short_ma=5, long_ma=20):
    """计算双均线策略信号，返回带信号列的DataFrame"""
    df = df.copy()
    df['MA_short'] = df['c'].rolling(window=short_ma).mean()
    df['MA_long'] = df['c'].rolling(window=long_ma).mean()
    
    # 金叉：短期均线上穿长期均线
    df['signal'] = 0
    df.loc[(df['MA_short'] > df['MA_long']) & (df['MA_short'].shift(1) <= df['MA_long'].shift(1)), 'signal'] = 1   # 买入信号
    df.loc[(df['MA_short'] < df['MA_long']) & (df['MA_short'].shift(1) >= df['MA_long'].shift(1)), 'signal'] = -1  # 卖出信号
    
    # 最新信号状态（当前持仓方向）
    df['position'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'position'] = 1   # 持多
    df.loc[df['MA_short'] < df['MA_long'], 'position'] = -1  # 持空
    return df


# ===== 模拟下单 =====
def place_order(side, size):
    st.info(f"模拟下单：{side} {size} ETH（未实盘）")
    # 实盘需实现 OKX 下单签名与 API
    return True


# ===== 主界面 =====
def main():
    st.title("📊 5分钟以太坊合约 · 双均线策略看盘")

    # ---- 侧边栏参数 ----
    with st.sidebar:
        st.header("策略参数")
        short_ma = st.number_input("短期均线长度", min_value=1, max_value=50, value=5, step=1)
        long_ma = st.number_input("长期均线长度", min_value=2, max_value=100, value=20, step=1)
        if short_ma >= long_ma:
            st.error("短期均线必须小于长期均线！")
            st.stop()
        
        st.header("交易设置")
        trade_size = st.number_input("下单数量 (ETH)", min_value=0.001, value=0.01, step=0.001, format="%.3f")
        mode = st.radio("模式", ["模拟", "实盘（需配置密钥）"], index=0, disabled=True)  # 暂时禁用实盘
        st.markdown("---")
        st.caption("数据源: OKX ETH-USDT 永续合约")

    # ---- 获取数据 ----
    df = get_candles(bar="5m", limit=200)
    if df.empty:
        st.error("无法获取K线数据，请稍后重试。")
        return

    # ---- 计算策略信号 ----
    df = calculate_signals(df, short_ma, long_ma)

    # ---- 绘制K线图 + 均线 + 买卖点 ----
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("ETH-USDT 价格", "成交量")
    )

    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["o"],
            high=df["h"],
            low=df["l"],
            close=df["c"],
            name="K线"
        ),
        row=1, col=1
    )

    # 均线
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["MA_short"], line=dict(color="blue", width=1), name=f"MA{short_ma}"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["MA_long"], line=dict(color="orange", width=1), name=f"MA{long_ma}"),
        row=1, col=1
    )

    # 买卖信号点
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]
    fig.add_trace(
        go.Scatter(
            x=buy_signals["time"],
            y=buy_signals["c"] * 0.98,  # 略低于收盘价，避免遮挡K线
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="green"),
            name="买入信号"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sell_signals["time"],
            y=sell_signals["c"] * 1.02,  # 略高于收盘价
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="red"),
            name="卖出信号"
        ),
        row=1, col=1
    )

    # 成交量
    colors = ['red' if row['o'] > row['c'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df["time"], y=df["v"], name="成交量", marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ---- 当前信号提示 ----
    last_row = df.iloc[-1]
    signal_text = ""
    if last_row["position"] == 1:
        signal_text = "📈 当前处于多头持仓状态 (MA短 > MA长)"
    elif last_row["position"] == -1:
        signal_text = "📉 当前处于空头持仓状态 (MA短 < MA长)"
    else:
        signal_text = "⏸️ 无明确持仓信号 (均线粘合)"

    st.info(f"**最新策略状态**：{signal_text}")

    # 最近一次信号时间
    last_buy = buy_signals["time"].max() if not buy_signals.empty else None
    last_sell = sell_signals["time"].max() if not sell_signals.empty else None
    col1, col2 = st.columns(2)
    with col1:
        if last_buy:
            st.success(f"最近买入信号: {last_buy.strftime('%Y-%m-%d %H:%M')}")
    with col2:
        if last_sell:
            st.error(f"最近卖出信号: {last_sell.strftime('%Y-%m-%d %H:%M')}")

    # ---- 手动下单区 ----
    st.subheader("✋ 手动交易")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 买入"):
            place_order("buy", trade_size)
    with col2:
        if st.button("📉 卖出"):
            place_order("sell", trade_size)

    st.caption("本工具仅供学习参考，不构成投资建议。实盘需自行承担风险。")


if __name__ == "__main__":
    main()
