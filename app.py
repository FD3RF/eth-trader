import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ===== OKX 配置 =====
API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"
PASSPHRASE = "YOUR_PASSPHRASE"
BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="5分钟ETH突破策略看盘+回测数据", page_icon="📈")


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
    df['prev_high'] = df['h'].shift(lookback)
    df['prev_low'] = df['l'].shift(lookback)

    buy_base = (df['l'] >= df['prev_low']) & (df['h'] > df['prev_high'])
    sell_base = (df['h'] <= df['prev_high']) & (df['l'] < df['prev_low'])

    if volume_type != "无":
        df['vol_ma'] = df['v'].rolling(window=vol_ma_period).mean()
        if volume_type == "放量":
            vol_condition = df['v'] > df['vol_ma'] * vol_multiplier
        else:
            vol_condition = df['v'] < df['vol_ma'] * vol_multiplier
    else:
        vol_condition = True

    df['signal'] = 0
    df.loc[buy_base & vol_condition, 'signal'] = 1
    df.loc[sell_base & vol_condition, 'signal'] = -1

    df['position'] = df['signal'].replace(0, method='ffill').fillna(0)
    return df


# ===== 模拟下单 =====
def place_order(side, size):
    st.info(f"模拟下单：{side} {size} ETH（未实盘）")
    return True


# ===== 加载回测数据（内置CSV或上传）=====
@st.cache_data
def load_backtest_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"读取文件失败: {e}")
            return pd.DataFrame()
    else:
        # 内置数据（从您提供的CSV内容复制）
        data = """body_threshold,vol_ma_period,break_threshold,交易数,多头交易数,空头交易数,多头胜率,空头胜率,多头盈利,空头盈利,胜率,总盈利,最大回撤,夏普比率,盈亏比
0.15,15,0.001,235,116,119,66.37931034482759,68.0672268907563,470.15512826847237,558.4718197276695,67.23404255319149,1028.6269479961416,0.4165299098957239,0.8606266549250197,1.7961553750939152
0.15,15,0.0008,258,125,133,65.60000000000001,65.41353383458647,466.079787761337,561.6676983829412,65.50387596899225,1027.7474861442784,0.4165299098957239,0.8493591728762571,1.760506690206602
0.15,15,0.0012,217,105,112,71.42857142857143,70.53571428571429,485.9436610909964,574.5544817950911,70.96774193548387,1060.4981428860874,0.3969425913510632,0.9059610043479966,1.7617952052434003
0.15,10,0.001,233,115,118,66.08695652173913,67.79661016949152,461.05667668061426,549.7809932020772,66.95278969957081,1010.8376698826912,0.4165299098957239,0.8469443330215395,1.7962579357153006
0.15,10,0.0008,257,124,133,65.32258064516128,65.41353383458647,456.9813361734789,554.3537055882366,65.36964980544747,1011.3350417617157,0.4165299098957239,0.8366713249976645,1.751164374377886
0.15,10,0.0012,215,104,111,71.15384615384616,70.27027027027027,476.8452095031383,565.8636552694987,70.69767441860465,1042.7088647726368,0.3969425913510632,0.8920782446041523,1.761987279920726
0.15,20,0.001,236,117,119,66.66666666666666,68.0672268907563,488.20047233781236,562.6633965616921,67.37288135593221,1050.8638688995043,0.4165299098957239,0.8780882160300897,1.8276766128866202
0.15,20,0.0008,260,126,134,65.87301587301587,65.67164179104478,484.125131830677,567.2361089478518,65.76923076923077,1051.361240778529,0.4165299098957239,0.8674411025233326,1.7799696442583801
0.15,20,0.0012,218,106,112,71.69811320754717,70.53571428571429,503.9890051603364,578.7460586291139,71.10091743119266,1082.7350637894501,0.3969425913510632,0.9237215869857187,1.7967803489629814
0.12,15,0.001,235,116,119,66.37931034482759,68.0672268907563,470.15512826847237,558.4718197276695,67.23404255319149,1028.6269479961416,0.4165299098957239,0.8606266549250197,1.7961553750939152
0.12,15,0.0008,258,125,133,65.60000000000001,65.41353383458647,466.079787761337,561.6676983829412,65.50387596899225,1027.7474861442784,0.4165299098957239,0.8493591728762571,1.760506690206602
0.12,15,0.0012,217,105,112,71.42857142857143,70.53571428571429,485.9436610909964,574.5544817950911,70.96774193548387,1060.4981428860874,0.3969425913510632,0.9059610043479966,1.7617952052434003
0.12,10,0.001,233,115,118,66.08695652173913,67.79661016949152,461.05667668061426,549.7809932020772,66.95278969957081,1010.8376698826912,0.4165299098957239,0.8469443330215395,1.7962579357153006
0.12,10,0.0008,257,124,133,65.32258064516128,65.41353383458647,456.9813361734789,554.3537055882366,65.36964980544747,1011.3350417617157,0.4165299098957239,0.8366713249976645,1.751164374377886
0.12,10,0.0012,215,104,111,71.15384615384616,70.27027027027027,476.8452095031383,565.8636552694987,70.69767441860465,1042.7088647726368,0.3969425913510632,0.8920782446041523,1.761987279920726
0.12,20,0.001,236,117,119,66.66666666666666,68.0672268907563,488.20047233781236,562.6633965616921,67.37288135593221,1050.8638688995043,0.4165299098957239,0.8780882160300897,1.8276766128866202
0.12,20,0.0008,260,126,134,65.87301587301587,65.67164179104478,484.125131830677,567.2361089478518,65.76923076923077,1051.361240778529,0.4165299098957239,0.8674411025233326,1.7799696442583801
0.12,20,0.0012,218,106,112,71.69811320754717,70.53571428571429,503.9890051603364,578.7460586291139,71.10091743119266,1082.7350637894501,0.3969425913510632,0.9237215869857187,1.7967803489629814
0.18,15,0.001,235,116,119,66.37931034482759,68.0672268907563,470.15512826847237,558.4718197276695,67.23404255319149,1028.6269479961416,0.4165299098957239,0.8606266549250197,1.7961553750939152
0.18,15,0.0008,258,125,133,65.60000000000001,65.41353383458647,466.079787761337,561.6676983829412,65.50387596899225,1027.7474861442784,0.4165299098957239,0.8493591728762571,1.760506690206602
0.18,15,0.0012,217,105,112,71.42857142857143,70.53571428571429,485.9436610909964,574.5544817950911,70.96774193548387,1060.4981428860874,0.3969425913510632,0.9059610043479966,1.7617952052434003
0.18,10,0.001,233,115,118,66.08695652173913,67.79661016949152,461.05667668061426,549.7809932020772,66.95278969957081,1010.8376698826912,0.4165299098957239,0.8469443330215395,1.7962579357153006
0.18,10,0.0008,257,124,133,65.32258064516128,65.41353383458647,456.9813361734789,554.3537055882366,65.36964980544747,1011.3350417617157,0.4165299098957239,0.8366713249976645,1.751164374377886
0.18,10,0.0012,215,104,111,71.15384615384616,70.27027027027027,476.8452095031383,565.8636552694987,70.69767441860465,1042.7088647726368,0.3969425913510632,0.8920782446041523,1.761987279920726
0.18,20,0.001,236,117,119,66.66666666666666,68.0672268907563,488.20047233781236,562.6633965616921,67.37288135593221,1050.8638688995043,0.4165299098957239,0.8780882160300897,1.8276766128866202
0.18,20,0.0008,260,126,134,65.87301587301587,65.67164179104478,484.125131830677,567.2361089478518,65.76923076923077,1051.361240778529,0.4165299098957239,0.8674411025233326,1.7799696442583801
0.18,20,0.0012,218,106,112,71.69811320754717,70.53571428571429,503.9890051603364,578.7460586291139,71.10091743119266,1082.7350637894501,0.3969425913510632,0.9237215869857187,1.7967803489629814"""
        df = pd.read_csv(io.StringIO(data))
    return df


# ===== 主界面 =====
def main():
    st.title("📊 5分钟以太坊合约 · 低点不破+高点突破策略看盘 + 回测数据")

    # ---- 侧边栏参数 ----
    with st.sidebar:
        st.header("突破策略参数")
        lookback = st.number_input("比较前几根K线", min_value=1, max_value=10, value=1, step=1,
                                   help="例如1表示与前一根K线比较")

        volume_type = st.selectbox("成交量条件", ["无", "放量", "缩量"], index=1,
                                   help="放量：成交量大于均线倍数；缩量：小于均线倍数")
        if volume_type != "无":
            vol_ma_period = st.number_input("成交量均线周期", min_value=5, max_value=50, value=15, step=1)
            vol_multiplier = st.number_input("成交量倍数", min_value=0.1, value=1.2, step=0.1,
                                             help="成交量与均线的比值阈值")
        else:
            vol_ma_period = 15
            vol_multiplier = 1.0

        st.header("交易设置")
        trade_size = st.number_input("下单数量 (ETH)", min_value=0.001, value=0.01, step=0.001, format="%.3f")
        mode = st.radio("模式", ["模拟", "实盘（需配置密钥）"], index=0, disabled=True)
        st.markdown("---")
        st.caption("数据源: OKX ETH-USDT 永续合约")

        # ---- 新增：回测数据上传折叠区 ----
        with st.expander("📁 回测数据 (CSV)"):
            uploaded_file = st.file_uploader("上传回测CSV文件", type=["csv"])
            st.caption("若未上传，使用内置90天回测数据（27组参数）")

    # ---- 加载回测数据 ----
    backtest_df = load_backtest_data(uploaded_file)

    # ---- 获取实时K线数据 ----
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

    colors = ['red' if row['o'] > row['c'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df["time"], y=df["v"], name="成交量", marker_color=colors),
        row=2, col=1
    )

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

    last_buy = buy_signals["time"].max() if not buy_signals.empty else None
    last_sell = sell_signals["time"].max() if not sell_signals.empty else None
    col1, col2 = st.columns(2)
    with col1:
        if last_buy:
            st.success(f"最近买入信号: {last_buy.strftime('%Y-%m-%d %H:%M')}")
    with col2:
        if last_sell:
            st.error(f"最近卖出信号: {last_sell.strftime('%Y-%m-%d %H:%M')}")

    # ---- 回测数据展示 ----
    st.subheader("📊 回测参数表现总览")
    if not backtest_df.empty:
        # 选择要显示的列（排除一些不常用的）
        display_cols = ["body_threshold", "vol_ma_period", "break_threshold", 
                        "交易数", "胜率", "总盈利", "最大回撤", "夏普比率", "盈亏比"]
        available_cols = [col for col in display_cols if col in backtest_df.columns]
        
        # 添加排序功能
        sort_col = st.selectbox("排序依据", available_cols, index=available_cols.index("总盈利") if "总盈利" in available_cols else 0)
        sort_asc = st.checkbox("升序", value=False)
        
        sorted_df = backtest_df.sort_values(by=sort_col, ascending=sort_asc)
        
        # 高亮前三名（按总盈利，如果总盈利存在）
        def highlight_top3(s):
            if sort_col == "总盈利" and "总盈利" in sorted_df.columns:
                top3_values = sorted_df["总盈利"].nlargest(3).values
                return ['background-color: rgba(255, 255, 0, 0.2)' if s.name == "总盈利" and v in top3_values else '' for v in s]
            return ['' for _ in s]
        
        st.dataframe(sorted_df[available_cols].style.apply(highlight_top3, axis=0), use_container_width=True)
        
        # 显示最优参数（按总盈利）
        if "总盈利" in backtest_df.columns:
            best_row = backtest_df.loc[backtest_df["总盈利"].idxmax()]
            st.success(f"🏆 最佳参数（按总盈利）：body_threshold={best_row['body_threshold']}, vol_ma_period={best_row['vol_ma_period']}, break_threshold={best_row['break_threshold']}，总盈利={best_row['总盈利']:.2f}，胜率={best_row['胜率']:.2f}%")
    else:
        st.warning("无回测数据可显示")

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
