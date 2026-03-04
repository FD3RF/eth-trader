import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== OKX 配置 =====
API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"
PASSPHRASE = "YOUR_PASSPHRASE"
BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="5分钟ETH突破策略看盘", page_icon="📈")


# ===== 获取K线 =====
@st.cache_data(ttl=30)
def get_candles(instId="ETH-USDT", bar="5m", limit=200):
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
        )[::-1]
        df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"获取K线失败: {e}")
        return pd.DataFrame()


# ===== 策略计算（低点不破 + 高点突破 + 成交量条件）=====
def calculate_signals(df, lookback=1, volume_type="放量", vol_ma_period=15, vol_multiplier=1.2):
    """
    根据条件生成信号：
    - 买入：当前最低价 >= 前 lookback 根K线的最低价 且 当前最高价 > 前 lookback 根K线的最高价
    - 卖出：当前最高价 <= 前 lookback 根K线的最高价 且 当前最低价 < 前 lookback 根K线的最低价
    成交量条件：
      - "放量": 成交量 > 均线 * vol_multiplier
      - "缩量": 成交量 < 均线 * vol_multiplier
      - "无": 不检查成交量
    """
    df = df.copy()
    # 前 lookback 根的最高、最低
    df['prev_high'] = df['h'].shift(lookback)
    df['prev_low'] = df['l'].shift(lookback)

    # 基础条件
    buy_base = (df['l'] >= df['prev_low']) & (df['h'] > df['prev_high'])
    sell_base = (df['h'] <= df['prev_high']) & (df['l'] < df['prev_low'])

    # 成交量条件
    if volume_type != "无":
        df['vol_ma'] = df['v'].rolling(window=vol_ma_period).mean()
        if volume_type == "放量":
            vol_condition = df['v'] > df['vol_ma'] * vol_multiplier
        else:  # 缩量
            vol_condition = df['v'] < df['vol_ma'] * vol_multiplier
    else:
        vol_condition = True  # 始终满足

    df['signal'] = 0
    df.loc[buy_base & vol_condition, 'signal'] = 1
    df.loc[sell_base & vol_condition, 'signal'] = -1

    # 持仓状态：根据最近一次信号维持
    df['position'] = df['signal'].replace(0, method='ffill').fillna(0)
    return df


# ===== 模拟下单 =====
def place_order(side, size):
    st.info(f"模拟下单：{side} {size} ETH（未实盘）")
    return True


# ===== 主界面 =====
def main():
    st.title("📊 5分钟以太坊合约 · 低点不破+高点突破策略看盘")

    # ---- 侧边栏参数（新增成交量类型）----
    with st.sidebar:
        st.header("突破策略参数")
        lookback = st.number_input("比较前几根K线", min_value=1, max_value=10, value=1, step=1,
                                   help="例如1表示与前一根K线比较")

        # 成交量条件选择
        volume_type = st.selectbox("成交量条件", ["无", "放量", "缩量"], index=1,
                                   help="放量：成交量大于均线倍数；缩量：小于均线倍数")
        if volume_type != "无":
            vol_ma_period = st.number_input("成交量均线周期", min_value=5, max_value=50, value=15, step=1)
            vol_multiplier = st.number_input("成交量倍数", min_value=0.1, value=1.2, step=0.1,
                                             help="成交量与均线的比值阈值")
        else:
            vol_ma_period = 15   # 占位，实际不使用
            vol_multiplier = 1.0

        st.header("交易设置")
        trade_size = st.number_input("下单数量 (ETH)", min_value=0.001, value=0.01, step=0.001, format="%.3f")
        mode = st.radio("模式", ["模拟", "实盘（需配置密钥）"], index=0, disabled=True)
        st.markdown("---")
        st.caption("数据源: OKX ETH-USDT 永续合约")

    # ---- 获取数据 ----
    df = get_candles(bar="5m", limit=200)
    if df.empty:
        st.error("无法获取K线数据，请稍后重试。")
        return

    # ---- 计算策略信号 ----
    df = calculate_signals(df, lookback, volume_type, vol_ma_period, vol_multiplier)

    # ---- 绘制K线图 + 信号点 ----
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

    # 买卖信号点
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]
    fig.add_trace(
        go.Scatter(
            x=buy_signals["time"],
            y=buy_signals["l"] * 0.99,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="green"),
            name="买入信号"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sell_signals["time"],
            y=sell_signals["h"] * 1.01,
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

    # 如果启用了成交量条件，显示成交量均线
    if volume_type != "无":
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["vol_ma"], line=dict(color="purple", width=1), name="成交量MA"),
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
    if last_row["position"] == 1:
        signal_text = "📈 当前处于多头持仓状态 (最后信号为买入)"
    elif last_row["position"] == -1:
        signal_text = "📉 当前处于空头持仓状态 (最后信号为卖出)"
    else:
        signal_text = "⏸️ 无持仓 (最近无信号)"

    st.info(f"**最新策略状态**：{signal_text}")

    # 最近一次信号时间
    last_buy = buy_signals["time"].max() if not buy_signals.empty else None
    last_sell = sell_signals["time"].max() if not buy_signals.empty else None
    col1, col2 = st.columns(2)
    with col1:
        if last_buy:
            st.success(f"最近买入信号: {last_buy.strftime('%Y-%m-%d %H:%M')}")
    with col2:
        if last_sell:
            st.error(f"最近卖出信号: {last_sell.strftime('%Y-%m-%d %H:%M')}")

    # ---- 策略历史表现亮点（您提供的回测结果）----
    st.subheader("🏆 策略历史表现亮点（90天回测）")
    
    train_metrics = {
        "交易数": 407,
        "总盈利": 4066.62,
        "胜率": "82.31%",
        "盈亏比": 1.47,
        "夏普比率": 0.85,
        "年化收益率": "464.70%"
    }
    test_metrics = {
        "交易数": 136,
        "总盈利": 939.18,
        "胜率": "78.68%",
        "盈亏比": 1.69,
        "夏普比率": 0.88,
        "年化收益率": "518.21%"
    }

    col_left, col_mid, col_right = st.columns(3)
    with col_left:
        st.markdown("##### 🏋️ 训练集")
        st.metric("胜率", train_metrics["胜率"])
        st.metric("总盈利", f"{train_metrics['总盈利']:.2f}")
        st.metric("夏普比率", train_metrics["夏普比率"])
    with col_mid:
        st.markdown("##### 🧪 测试集")
        st.metric("胜率", test_metrics["胜率"])
        st.metric("总盈利", f"{test_metrics['总盈利']:.2f}")
        st.metric("夏普比率", test_metrics["夏普比率"])
    with col_right:
        st.markdown("##### 📊 其他指标")
        st.metric("训练集交易数", train_metrics["交易数"])
        st.metric("测试集交易数", test_metrics["交易数"])
        st.metric("训练集年化", train_metrics["年化收益率"])

    st.caption("注：以上数据基于历史90天5分钟K线回测，仅供参考。")

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
