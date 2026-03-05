import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# ---------- 页面配置 ----------
st.set_page_config(layout="wide", page_title="ETH 5min 量价心法")
st.title("📈 以太坊 5分钟 多空精准口诀策略监控 (增强版)")

# ---------- 侧边栏参数 ----------
with st.sidebar:
    st.header("⚙️ 系统设置")
    proxy = st.text_input("代理服务器 (可选)", "", help="例: http://127.0.0.1:7890")
    refresh_interval = st.number_input("自动刷新间隔 (秒)", min_value=10, max_value=300, value=30, step=10)
    
    st.divider()
    st.header("📐 策略参数")
    symbol = st.text_input("交易对", "ETHUSDT")
    interval = st.selectbox("K线周期", ["1m", "5m", "15m", "1h"], index=1)
    limit = st.number_input("加载K线数量", min_value=100, max_value=1000, value=300)
    
    st.subheader("核心规则")
    lookback = st.number_input("关键点回溯周期", value=20, min_value=5, max_value=100)
    vol_window = st.number_input("成交量均线周期", value=5, min_value=3, max_value=30)
    shrink_thresh = st.slider("缩量阈值 (< 均线×系数)", 0.1, 1.0, 0.6)
    expand_thresh = st.slider("放量阈值 (> 均线×系数)", 1.0, 5.0, 1.5)
    body_min_ratio = st.slider("最小实体占比", 0.0, 1.0, 0.5, help="实体长度 / 全影线长度，用于过滤十字星")
    touch_tolerance = st.slider("关键点接近容忍度 (%)", 0.1, 2.0, 0.2, help="价格接近前高/前底的百分比") / 100.0
    confirm_bars = st.number_input("信号确认窗口 (K线根数)", value=3, min_value=1, max_value=10)

    st.subheader("止损止盈")
    stop_loss_pct = st.slider("止损偏移 (%)", 0.1, 2.0, 0.5) / 100.0
    risk_reward = st.slider("盈亏比", 1.0, 5.0, 1.5)

# ---------- 数据获取 (多端点+代理) ----------
@st.cache_data(ttl=30)
def fetch_klines(symbol, interval, limit, proxy):
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines"
    ]
    proxies = {"http": proxy, "https": proxy} if proxy else None
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    
    for url in endpoints:
        try:
            resp = requests.get(url, params=params, proxies=proxies, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'q_vol', 'trades', 't_buy_base', 't_buy_quote', 'ignore'
                ])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            continue
    return None

# ---------- 策略引擎 ----------
def generate_signals(df):
    data = df.copy()
    
    # 1. 关键点 (滚动前高/前低，shift 1 避免使用当前K线本身)
    data['prev_high'] = data['high'].rolling(lookback, min_periods=1).max().shift(1)
    data['prev_low']  = data['low'].rolling(lookback, min_periods=1).min().shift(1)
    
    # 2. 成交量均线
    data['vol_ma'] = data['volume'].rolling(vol_window, min_periods=1).mean()
    
    # 3. 实体占比过滤
    data['body'] = abs(data['close'] - data['open'])
    data['range'] = data['high'] - data['low']
    data['body_ratio'] = data['body'] / data['range']
    data['solid'] = data['body_ratio'] >= body_min_ratio
    
    # 4. 观察阶段 (缩量触碰关键位)
    # 做多观察：价格接近前低且不破 (收盘在前低之上)，成交量萎缩
    data['obs_long'] = (
        (data['low'] <= data['prev_low'] * (1 + touch_tolerance)) &
        (data['close'] > data['prev_low'] * 0.999) &  # 未有效跌破
        (data['volume'] < data['vol_ma'] * shrink_thresh)
    )
    
    # 做空观察：价格接近前高且不破 (收盘在前高之下)，成交量萎缩
    data['obs_short'] = (
        (data['high'] >= data['prev_high'] * (1 - touch_tolerance)) &
        (data['close'] < data['prev_high'] * 1.001) &  # 未有效突破
        (data['volume'] < data['vol_ma'] * shrink_thresh)
    )
    
    # 5. 确认信号 (放量突破/跌破)
    data['signal'] = 0
    data['entry'] = np.nan
    data['stop'] = np.nan
    data['target'] = np.nan
    
    for i in range(len(data)):
        # 做多确认：观察后 N 根内出现放量阳线突破前高
        if i >= confirm_bars and any(data['obs_long'].iloc[i-confirm_bars:i]):
            if (data['volume'].iloc[i] > data['vol_ma'].iloc[i] * expand_thresh and
                data['close'].iloc[i] > data['prev_high'].iloc[i] and
                data['solid'].iloc[i] and data['close'].iloc[i] > data['open'].iloc[i]):
                
                data.loc[data.index[i], 'signal'] = 1
                entry = data['close'].iloc[i]
                stop = data['prev_low'].iloc[i] * (1 - stop_loss_pct)
                data.loc[data.index[i], 'entry'] = entry
                data.loc[data.index[i], 'stop'] = stop
                data.loc[data.index[i], 'target'] = entry + (entry - stop) * risk_reward
        
        # 做空确认：观察后 N 根内出现放量阴线跌破前低
        if i >= confirm_bars and any(data['obs_short'].iloc[i-confirm_bars:i]):
            if (data['volume'].iloc[i] > data['vol_ma'].iloc[i] * expand_thresh and
                data['close'].iloc[i] < data['prev_low'].iloc[i] and
                data['solid'].iloc[i] and data['close'].iloc[i] < data['open'].iloc[i]):
                
                data.loc[data.index[i], 'signal'] = -1
                entry = data['close'].iloc[i]
                stop = data['prev_high'].iloc[i] * (1 + stop_loss_pct)
                data.loc[data.index[i], 'entry'] = entry
                data.loc[data.index[i], 'stop'] = stop
                data.loc[data.index[i], 'target'] = entry - (stop - entry) * risk_reward
    
    # 6. 特殊陷阱信号 (可选标记，此处作为观察提示，不单独生成信号)
    data['trap_long'] = (
        (data['volume'] > data['vol_ma'] * expand_thresh * 1.5) &
        (data['high'] >= data['prev_high'] * 0.998) &
        (data['close'] < data['prev_high']) &
        (data['close'] < data['open'])  # 阴线
    )
    data['trap_short'] = (
        (data['volume'] > data['vol_ma'] * expand_thresh * 1.5) &
        (data['low'] <= data['prev_low'] * 1.002) &
        (data['close'] > data['prev_low']) &
        (data['close'] > data['open'])  # 阳线
    )
    
    return data

# ---------- 回测统计 ----------
def backtest_stats(df):
    signals = df[df['signal'] != 0].copy()
    if len(signals) == 0:
        return None
    
    results = []
    for idx, row in signals.iterrows():
        future = df.loc[idx+1:, ['close', 'high', 'low']]
        if len(future) == 0:
            continue
        if row['signal'] == 1:  # 做多
            stop_hit = (future['low'] <= row['stop']).any()
            target_hit = (future['high'] >= row['target']).any()
            if target_hit:
                pnl = (row['target'] - row['entry']) / row['entry'] * 100
                result = 'win'
            elif stop_hit:
                pnl = (row['stop'] - row['entry']) / row['entry'] * 100
                result = 'loss'
            else:
                # 未触及止损止盈，以最新价计算
                last = future['close'].iloc[-1]
                pnl = (last - row['entry']) / row['entry'] * 100
                result = 'open'
        else:  # 做空
            stop_hit = (future['high'] >= row['stop']).any()
            target_hit = (future['low'] <= row['target']).any()
            if target_hit:
                pnl = (row['entry'] - row['target']) / row['entry'] * 100
                result = 'win'
            elif stop_hit:
                pnl = (row['entry'] - row['stop']) / row['entry'] * 100
                result = 'loss'
            else:
                last = future['close'].iloc[-1]
                pnl = (row['entry'] - last) / row['entry'] * 100
                result = 'open'
        
        results.append({
            '时间': row['timestamp'],
            '方向': '多' if row['signal']==1 else '空',
            '入场价': round(row['entry'], 2),
            '止损': round(row['stop'], 2),
            '止盈': round(row['target'], 2),
            '盈亏%': round(pnl, 2),
            '状态': result
        })
    
    stats_df = pd.DataFrame(results)
    # 统计已平仓信号
    closed = stats_df[stats_df['状态'] != 'open']
    if len(closed) > 0:
        wins = closed[closed['状态']=='win']
        loss = closed[closed['状态']=='loss']
        win_rate = len(wins) / len(closed) * 100
        total_return = closed['盈亏%'].sum()
        avg_win = wins['盈亏%'].mean() if len(wins)>0 else 0
        avg_loss = loss['盈亏%'].mean() if len(loss)>0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    else:
        win_rate = total_return = profit_factor = 0
    
    return stats_df, win_rate, total_return, profit_factor

# ---------- 主界面 ----------
# 自动刷新
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

# 获取数据
df_raw = fetch_klines(symbol, interval, limit, proxy)
if df_raw is None:
    st.error("无法获取数据，请检查代理设置或网络连接。")
    st.stop()

df = generate_signals(df_raw)
last = df.iloc[-1]

# ---------- 顶部状态看板 ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("最新价格", f"{last['close']:.2f}")
vol_ratio = last['volume'] / last['vol_ma']
col2.metric("量能倍数", f"{vol_ratio:.2f}x", delta="放量" if vol_ratio>1 else "缩量")

with col3:
    if last['obs_long']:
        st.warning("🧐 缩量回踩低点 (观察做多)")
    elif last['obs_short']:
        st.warning("🧐 缩量反弹高点 (观察做空)")
    elif last['trap_long']:
        st.info("🔥 放量急涨不破前高 (可能诱多)")
    elif last['trap_short']:
        st.info("🔥 放量急跌不破前低 (可能诱空)")
    else:
        st.info("💤 无明确信号")

col4.metric("最新信号", "无" if df[df['signal']!=0].empty else "有信号", 
            delta_color="off")

# ---------- 图表绘制 ----------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    row_heights=[0.7, 0.3], vertical_spacing=0.05)

# K线
fig.add_trace(go.Candlestick(
    x=df['timestamp'], open=df['open'], high=df['high'],
    low=df['low'], close=df['close'], name='K线'
), row=1, col=1)

# 做多信号标记
buys = df[df['signal'] == 1]
fig.add_trace(go.Scatter(
    x=buys['timestamp'], y=buys['low'] * 0.997,
    mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'),
    name='做多信号', text=buys['entry']
), row=1, col=1)

# 做空信号标记
sells = df[df['signal'] == -1]
fig.add_trace(go.Scatter(
    x=sells['timestamp'], y=sells['high'] * 1.003,
    mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'),
    name='做空信号', text=sells['entry']
), row=1, col=1)

# 特殊陷阱标记 (可选，用不同符号)
traps_long = df[df['trap_long']]
traps_short = df[df['trap_short']]
fig.add_trace(go.Scatter(
    x=traps_long['timestamp'], y=traps_long['high'] * 1.005,
    mode='markers', marker=dict(symbol='diamond', size=10, color='orange'),
    name='诱多陷阱', showlegend=True
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=traps_short['timestamp'], y=traps_short['low'] * 0.995,
    mode='markers', marker=dict(symbol='diamond', size=10, color='purple'),
    name='诱空陷阱', showlegend=True
), row=1, col=1)

# 止损/止盈线 (只画最近5个信号，避免杂乱)
recent_signals = df[df['signal'] != 0].tail(5)
for _, row in recent_signals.iterrows():
    # 止损线
    fig.add_hline(y=row['stop'], line_dash="dash", line_color="red", 
                  opacity=0.5, row=1, col=1)
    # 止盈线
    fig.add_hline(y=row['target'], line_dash="dash", line_color="green", 
                  opacity=0.5, row=1, col=1)

# 成交量
colors = ['#EF5350' if row['close'] < row['open'] else '#26A69A' for _, row in df.iterrows()]
fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
# 成交量均线
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vol_ma'], line=dict(color='yellow', width=1), 
                         name='成交量均线'), row=2, col=1)

fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False,
                  hovermode='x unified', showlegend=True)
fig.update_xaxes(title_text="时间", row=2, col=1)
fig.update_yaxes(title_text="价格", row=1, col=1)
fig.update_yaxes(title_text="成交量", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------- 信号与统计 ----------
stats_df, win_rate, total_return, profit_factor = backtest_stats(df)

if stats_df is not None:
    st.subheader("📋 最近信号与回测表现")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("胜率 (已平仓)", f"{win_rate:.1f}%")
    col_stat2.metric("累计盈亏%", f"{total_return:.2f}%")
    col_stat3.metric("盈亏比", f"{profit_factor:.2f}")
    
    st.dataframe(stats_df.tail(10), use_container_width=True)
else:
    st.info("暂无历史信号")

# ---------- 策略说明 ----------
with st.expander("📖 查看完整交易计划"):
    st.markdown(f"""
    ### 以太坊 5分钟 多空精准口诀策略

    **入场规则**  
    - **做多**：先出现缩量回踩前低（接近前低且成交量 < 均线×{shrink_thresh}），随后 {confirm_bars} 根K线内出现放量突破前高的实体阳线（成交量 > 均线×{expand_thresh}，实体占比 ≥ {body_min_ratio}），则在突破收盘价入场。  
    - **做空**：先出现缩量反弹前高（接近前高且成交量 < 均线×{shrink_thresh}），随后 {confirm_bars} 根K线内出现放量跌破前低的实体阴线，则在跌破收盘价入场。  

    **特殊陷阱信号**（仅提示，不自动交易）  
    - 放量急涨但收盘未能站上前高 → 诱多  
    - 放量急跌但收盘未能跌破前低 → 诱空  

    **出场规则**  
    - 止损：入场价 ± {stop_loss_pct*100}%（以前低/前高为基准）  
    - 止盈：盈亏比 {risk_reward}:1  

    **当前参数**  
    - 关键点回溯周期：{lookback}  
    - 成交量均线周期：{vol_window}  
    - 缩量阈值：{shrink_thresh}  
    - 放量阈值：{expand_thresh}  
    - 实体占比要求：≥ {body_min_ratio}  
    - 关键点接近容忍度：{touch_tolerance*100}%  
    - 确认窗口：{confirm_bars} 根K线  
    """)
