import streamlit as st
import requests
import pandas as pd
import ta  # 技术分析库，用于计算常用指标
import plotly.graph_objects as go  # 绘图库，用于绘制K线图
from streamlit_autorefresh import st_autorefresh  # 页面自动刷新组件
import os
from datetime import datetime
import time

# ==========================
# 基础配置
# ==========================
st.set_page_config(layout="wide")  # 设置页面为宽屏模式
st.title("顶级模型 | 高质量单（胜率优先）")  # 页面主标题

# 交易对符号（OKEx永续合约）
SYMBOL = "ETH-USDT-SWAP"

# 历史记录文件（用于存储已触发的信号订单）
HISTORY_FILE = "history.csv"

# 上一次信号的记录文件（用于防重复触发）
LAST_SIGNAL_FILE = "last_signal.txt"

# 设置页面自动刷新间隔（毫秒），这里每8秒刷新一次
st_autorefresh(interval=8000, key="refresh")


# ==========================
# 数据获取函数（带缓存和重试机制）
# ==========================
@st.cache_data(ttl=5)  # 缓存数据5秒，避免每8秒都重复请求，减轻API压力
def get_data():
    """
    从OKEx API获取ETH永续合约的5分钟K线数据
    返回包含时间、开高低收、成交量的DataFrame，按时间升序排列
    若获取失败则返回空DataFrame
    """
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": SYMBOL, "bar": "5m", "limit": 300}  # 获取最近300根K线
    retries = 3  # 重试次数
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=5, params=params)
            j = r.json()
            if "data" in j:
                break
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"数据获取失败: {e}")
                return pd.DataFrame()
            time.sleep(1)  # 等待1秒后重试
    else:
        return pd.DataFrame()  # 所有重试均失败

    if "data" not in j:
        return pd.DataFrame()

    # 将API返回的数据转换为DataFrame
    df = pd.DataFrame(j["data"], columns=[
        "ts", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ])
    # 转换时间戳为datetime类型
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    # 将价格和成交量转换为浮点数
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    # 按时间升序排列（API返回可能是降序）
    return df.sort_values("ts").reset_index(drop=True)


# 获取数据
df = get_data()
if df.empty:
    st.error("无法获取数据，请检查网络或API服务")
    st.stop()  # 停止后续执行

# 确保数据量足够计算指标（至少需要60根K线）
if len(df) < 60:
    st.warning("数据量不足，可能影响指标计算")
    # 继续执行，但可能指标会有NaN


# ==========================
# 技术指标计算
# ==========================
# 指数移动平均线
df["EMA20"] = ta.trend.ema_indicator(df["close"], window=20)
df["EMA60"] = ta.trend.ema_indicator(df["close"], window=60)

# 相对强弱指标（RSI）
df["RSI"] = ta.momentum.rsi(df["close"], window=14)

# 平均真实波幅（ATR）
df["ATR"] = ta.volatility.average_true_range(
    df["high"], df["low"], df["close"], window=14
)

# 成交量简单移动平均（20周期）
df["VOL_MA20"] = df["volume"].rolling(window=20).mean()

# 删除含有NaN值的行，确保后续计算无误
df = df.dropna().reset_index(drop=True)

# 最新一根K线的数据
latest = df.iloc[-1]


# ==========================
# 趋势判断（基于EMA20与EMA60的相对位置）
# ==========================
trend = "无"
# 要求最近3根K线的EMA20全部大于EMA60，定义为多头趋势
if (df["EMA20"].iloc[-3:] > df["EMA60"].iloc[-3:]).all():
    trend = "多"
# 要求最近3根K线的EMA20全部小于EMA60，定义空头趋势
elif (df["EMA20"].iloc[-3:] < df["EMA60"].iloc[-3:]).all():
    trend = "空"


# ==========================
# 结构与回调识别（高胜率核心逻辑）
# ==========================
# 关键结构：最近20根K线的最高价（阻力）和最低价（支撑）
resistance = df["high"].iloc[-20:].max()
support = df["low"].iloc[-20:].min()

def is_retest(price, level):
    """
    判断价格是否回踩/反抽到关键水平附近（允许±0.2%的误差）
    用于确认回调是否到位
    """
    return price <= level * 1.002 and price >= level * 0.998

def big_shadow(row):
    """
    假突破过滤：检查最新K线是否存在过长的上影线
    若上影线长度超过实体长度的1.2倍，视为假突破，结构无效
    """
    body = abs(row["close"] - row["open"])
    upper = row["high"] - max(row["close"], row["open"])
    # 上影线显著长于实体
    return upper > body * 1.2

# 判断当前K线是否具有过长上影线（假突破信号）
if big_shadow(latest):
    structure_ok = False  # 结构无效
else:
    structure_ok = True   # 结构有效

# 动能过滤：根据趋势方向，RSI需处于合理区间，避免过热或弱势
rsi = latest["RSI"]
if trend == "多":
    rsi_ok = (45 <= rsi <= 65)   # 多头趋势下RSI不宜过高或过低
elif trend == "空":
    rsi_ok = (35 <= rsi <= 55)   # 空头趋势下RSI应在偏弱区间
else:
    rsi_ok = False

# 盈亏比计算（基于ATR）
entry = latest["close"]
atr = latest["ATR"]
if trend == "多":
    stop = entry - atr            # 做多止损 = 入场价 - ATR
    tp = entry + atr * 2          # 做多止盈 = 入场价 + 2倍ATR
else:
    stop = entry + atr            # 做空止损 = 入场价 + ATR
    tp = entry - atr * 2          # 做空止盈 = 入场价 - 2倍ATR

# 计算盈亏比（绝对值）
if (entry - stop) != 0:
    rr = abs((tp - entry) / (entry - stop))
else:
    rr = 0

# 成交量过滤：最新成交量需大于20周期均量的1.3倍
volume_ok = latest["volume"] > df["VOL_MA20"].iloc[-1] * 1.3

# 最终信号生成（所有条件需同时满足）
signal = None
if trend != "无" and structure_ok and volume_ok and rsi_ok and rr >= 2:
    # 根据趋势方向，判断价格是否回踩到关键水平
    if trend == "多" and is_retest(entry, support):
        signal = "多"
    elif trend == "空" and is_retest(entry, resistance):
        signal = "空"

# 信号质量标记（仅用于显示）
quality = "高" if signal else "无"


# ==========================
# 防重复触发机制（避免同一价位连续触发）
# ==========================
def already_signaled(sig, price):
    """
    检查本次信号是否与上一次记录相同且价格相近（误差小于0.1%）
    防止短时间内重复触发
    """
    if not os.path.exists(LAST_SIGNAL_FILE):
        return False
    try:
        with open(LAST_SIGNAL_FILE, "r") as f:
            content = f.read().split(",")
            last_sig = content[0]
            last_price = float(content[1])
            # 价格相差小于0.1%视为同一价位
            price_diff_pct = abs(price - last_price) / last_price
            return last_sig == sig and price_diff_pct < 0.001
    except (IndexError, ValueError, FileNotFoundError):
        return False

def mark_signal(sig, price):
    """
    将本次信号记录到文件，供下次检查
    """
    with open(LAST_SIGNAL_FILE, "w") as f:
        f.write(f"{sig},{price}")


# ==========================
# 历史记录管理
# ==========================
def load_history():
    """
    从CSV文件加载历史订单记录
    若文件不存在则返回空DataFrame
    """
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["time", "direction", "entry", "stop", "tp", "result"])

def save_history_record(row):
    """
    将新订单追加到历史记录文件，并保留最近5000条
    """
    dfh = load_history()
    dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
    dfh = dfh.tail(5000)  # 限制历史记录长度
    dfh.to_csv(HISTORY_FILE, index=False)

def update_history_results():
    """
    遍历所有未平仓订单，根据最新价格更新其盈亏状态（win/lose）
    若价格触及止盈或止损，则标记结果
    """
    dfh = load_history()
    if dfh.empty:
        return dfh
    current_price = latest["close"]
    for idx, row in dfh.iterrows():
        # 只处理未标记结果的订单（result为空）
        if pd.isna(row.get("result")) or row["result"] == "":
            if row["direction"] == "多":
                if current_price >= row["tp"]:
                    dfh.at[idx, "result"] = "win"
                elif current_price <= row["stop"]:
                    dfh.at[idx, "result"] = "lose"
            else:  # 空头
                if current_price <= row["tp"]:
                    dfh.at[idx, "result"] = "win"
                elif current_price >= row["stop"]:
                    dfh.at[idx, "result"] = "lose"
    # 保存更新后的历史记录
    dfh.to_csv(HISTORY_FILE, index=False)
    return dfh

# 更新历史记录并获取最新DataFrame
history_df = update_history_results()

# ==========================
# 生成新信号（若满足条件且未重复）
# ==========================
if signal and not already_signaled(signal, entry):
    # 构造新订单记录
    new_record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": signal,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp": round(tp, 4),
        "result": ""  # 初始为空，表示未平仓
    }
    save_history_record(new_record)
    mark_signal(signal, entry)  # 记录本次信号，用于防重复


# ==========================
# 绘制K线图（含EMA线及关键水平）
# ==========================
fig = go.Figure()

# 添加K线图
fig.add_trace(go.Candlestick(
    x=df["ts"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="K线"
))

# 添加EMA20线
fig.add_trace(go.Scatter(
    x=df["ts"],
    y=df["EMA20"],
    line=dict(color="blue", width=1),
    name="EMA20"
))

# 添加EMA60线
fig.add_trace(go.Scatter(
    x=df["ts"],
    y=df["EMA60"],
    line=dict(color="orange", width=1),
    name="EMA60"
))

# 添加水平线：支撑和阻力
fig.add_hline(y=support, line_dash="dash", line_color="green", annotation_text="支撑")
fig.add_hline(y=resistance, line_dash="dash", line_color="red", annotation_text="阻力")

# 如果当前有信号，在最新K线处标记入场位
if signal:
    fig.add_trace(go.Scatter(
        x=[latest["ts"]],
        y=[entry],
        mode="markers",
        marker=dict(symbol="triangle-up" if signal=="多" else "triangle-down", size=15, color="gold"),
        name="信号入场"
    ))

# 设置图表布局
fig.update_layout(
    title=f"{SYMBOL} 5分钟K线图",
    xaxis_title="时间",
    yaxis_title="价格",
    template="plotly_dark",
    height=600
)

# 在Streamlit中显示图表
st.plotly_chart(fig, use_container_width=True)


# ==========================
# 状态面板与交易提示（优化显示）
# ==========================
st.subheader("📊 当前市场状态")

# 使用三列布局展示关键指标
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("趋势方向", trend)
with col2:
    st.metric("RSI(14)", f"{rsi:.2f}")
with col3:
    st.metric("ATR", f"{atr:.4f}")
with col4:
    st.metric("信号质量", quality)

# 显示入场、止损、止盈建议
st.subheader("💰 交易建议（仅当信号出现时有效）")
if signal:
    # 根据多空显示不同的颜色和图标
    if signal == "多":
        st.success("📈 高质量做多信号")
    else:
        st.error("📉 高质量做空信号")

    # 使用列布局展示具体价位
    col_e, col_s, col_t = st.columns(3)
    with col_e:
        st.metric("入场价", f"{entry:.4f}")
    with col_s:
        st.metric("止损价", f"{stop:.4f}", delta=f"{-atr:.4f}" if signal=="多" else f"{atr:.4f}")
    with col_t:
        st.metric("止盈价", f"{tp:.4f}", delta=f"{atr*2:.4f}" if signal=="多" else f"{-atr*2:.4f}")

    # 显示盈亏比
    st.info(f"**预期盈亏比**：{rr:.2f} : 1")

    # 显示信号触发理由
    reasons = []
    if trend != "无":
        reasons.append(f"趋势方向：{trend}")
    if is_retest(entry, support if trend=="多" else resistance):
        reasons.append("价格回踩关键水平")
    if volume_ok:
        reasons.append("成交量放大")
    if rsi_ok:
        reasons.append(f"RSI处于健康区间 ({rsi:.1f})")
    if structure_ok:
        reasons.append("无长上影线假突破")
    st.write("**信号理由**：" + "；".join(reasons))

else:
    st.warning("⏳ 暂无高质量交易机会，请耐心等待")

    # 即使无信号，也展示当前是否接近条件，帮助用户观察
    st.write("**当前条件检查**：")
    checks = {
        "趋势明确": trend != "无",
        "回踩关键位": is_retest(entry, support) or is_retest(entry, resistance),
        "成交量放大": volume_ok,
        "RSI合适": rsi_ok,
        "结构有效": structure_ok,
        "盈亏比达标": rr >= 2
    }
    for check_name, check_result in checks.items():
        icon = "✅" if check_result else "❌"
        st.write(f"{icon} {check_name}")


# ==========================
# 历史记录与胜率统计
# ==========================
st.subheader("📜 历史信号记录")

# 计算胜率等统计信息
if not history_df.empty:
    total_trades = len(history_df)
    completed = history_df[history_df["result"].isin(["win", "lose"])]
    wins = len(completed[completed["result"] == "win"])
    losses = len(completed[completed["result"] == "lose"])
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    # 显示统计卡片
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("总信号数", total_trades)
    with col_stat2:
        st.metric("已完成交易", len(completed))
    with col_stat3:
        st.metric("胜率", f"{win_rate:.1f}%" if not pd.isna(win_rate) else "N/A")
    with col_stat4:
        st.metric("当前持仓", len(history_df[history_df["result"] == ""]))

    # 显示最近20条记录
    st.dataframe(
        history_df.tail(20)[["time", "direction", "entry", "stop", "tp", "result"]],
        use_container_width=True
    )
else:
    st.info("暂无历史记录")
