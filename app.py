import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import itertools

BASE_URL = "https://www.okx.com"

st.set_page_config(layout="wide", page_title="终极突破系统", page_icon="📈")
st.title("📊 实体+成交量+前高前低突破 · 终极优化版")

# ---------------------- 内置90天历史回测数据 ----------------------
csv_data = """body_threshold,vol_ma_period,break_threshold,交易数,多头交易数,空头交易数,多头胜率,空头胜率,多头盈利,空头盈利,胜率,总盈利,最大回撤,夏普比率,盈亏比
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
backtest_df = pd.read_csv(io.StringIO(csv_data))

# ---------------------- 获取K线数据 ----------------------
@st.cache_data(ttl=30)
def get_candles(limit=1000):
    url = f"{BASE_URL}/api/v5/market/candles"
    try:
        r = requests.get(url, params={
            "instId": "ETH-USDT",
            "bar": "5m",
            "limit": limit
        }, timeout=10).json()
        if r.get("code") != "0":
            st.error(f"API错误: {r.get('msg', '未知错误')}")
            return pd.DataFrame()
        df = pd.DataFrame(r["data"], columns=[
            "ts", "o", "h", "l", "c", "v", "volCcy", "volCcyQuote", "confirm"
        ])[::-1]
        df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
        df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float)
        return df
    except Exception as e:
        st.error(f"获取K线失败: {e}")
        return pd.DataFrame()

df_raw = get_candles(limit=1000)
if df_raw.empty:
    st.stop()

# ---------------------- 策略信号生成 ----------------------
def generate_signals(df, body_threshold, vol_ma_period, volume_multiplier):
    df = df.copy()
    df['prev_high'] = df['h'].shift(1)
    df['prev_low'] = df['l'].shift(1)
    df['vol_ma'] = df['v'].rolling(window=vol_ma_period).mean()
    df['body'] = abs(df['c'] - df['o'])
    
    buy = (
        (df['c'] > df['prev_high']) &
        (df['c'] > df['o']) &
        (df['body'] > body_threshold) &
        (df['v'] > df['vol_ma'] * volume_multiplier)
    )
    sell = (
        (df['c'] < df['prev_low']) &
        (df['c'] < df['o']) &
        (df['body'] > body_threshold) &
        (df['v'] > df['vol_ma'] * volume_multiplier)
    )
    df['signal'] = 0
    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = -1
    return df

# ---------------------- 回测函数（修正夏普）---------------------
def run_backtest(df, fee_rate):
    initial_capital = 10000
    balance = initial_capital
    position = 0
    entry_price = 0.0
    trades = []          # 每笔收益率（%）
    equity_curve = [balance]
    timestamps = [df['time'].iloc[0]]
    
    # 确保有足够数据
    start_idx = max(vol_ma_period, 2) + 1
    if start_idx >= len(df) - 1:
        return {
            'total_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'rr': np.nan,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'equity_curve': equity_curve,
            'trades': []
        }
    
    for i in range(start_idx, len(df)-1):
        sig = df['signal'].iloc[i]
        if sig != 0 and sig != position:
            next_open = df['o'].iloc[i+1]
            next_time = df['time'].iloc[i+1]
            # 平仓
            if position != 0:
                if position == 1:
                    ret = (next_open / entry_price - 1)
                else:
                    ret = (entry_price / next_open - 1)
                ret -= fee_rate
                balance *= (1 + ret)
                trades.append(ret * 100)
                equity_curve.append(balance)
                timestamps.append(next_time)
            # 开新仓
            if sig == 1:
                entry_price = next_open
                position = 1
                balance *= (1 - fee_rate)
                equity_curve.append(balance)
                timestamps.append(next_time)
            elif sig == -1:
                entry_price = next_open
                position = -1
                balance *= (1 - fee_rate)
                equity_curve.append(balance)
                timestamps.append(next_time)
    
    # 最后平仓
    if position != 0:
        last_close = df['c'].iloc[-1]
        last_time = df['time'].iloc[-1]
        if position == 1:
            ret = (last_close / entry_price - 1)
        else:
            ret = (entry_price / last_close - 1)
        ret -= fee_rate
        balance *= (1 + ret)
        trades.append(ret * 100)
        equity_curve.append(balance)
        timestamps.append(last_time)
    
    total_return = (balance / initial_capital - 1) * 100
    num_trades = len(trades)
    
    if num_trades > 0:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        win_rate = len(wins) / num_trades * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else np.inf
        rr = avg_win / avg_loss if avg_loss > 0 else np.nan
        
        # 计算日收益率用于夏普
        equity = np.array(equity_curve)
        times = pd.Series(timestamps)
        daily_returns = []
        current_day = None
        day_start_equity = equity[0]
        for i, t in enumerate(times):
            day = t.date()
            if current_day is None:
                current_day = day
            if day != current_day:
                daily_ret = (equity[i-1] / day_start_equity) - 1
                daily_returns.append(daily_ret)
                current_day = day
                day_start_equity = equity[i-1]
        daily_ret = (equity[-1] / day_start_equity) - 1
        daily_returns.append(daily_ret)
        
        daily_returns = np.array(daily_returns)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)
        else:
            sharpe = 0.0
        
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        rr = np.nan
        sharpe = 0.0
        max_drawdown = 0.0
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'rr': rr,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'equity_curve': equity_curve,
        'trades': trades
    }

# ---------------------- 侧边栏 ----------------------
with st.sidebar:
    st.header("⚙️ 策略参数")
    
    # 从历史回测选择
    st.subheader("📋 从历史回测选择")
    # 安全地创建选项列表：使用索引，并确保格式函数不会出错
    # 先备份索引
    index_list = backtest_df.index.tolist()
    # 格式化显示文本
    option_labels = []
    for idx in index_list:
        row = backtest_df.loc[idx]
        # 使用 get 或 try 防止缺失值
        try:
            label = f"body={row['body_threshold']:.2f}, vol_ma={row['vol_ma_period']}, break={row['break_threshold']:.4f}, 胜率={row['胜率']:.1f}%, 总盈利={row['总盈利']:.0f}"
        except:
            label = f"索引 {idx}"
        option_labels.append(label)
    
    selected_label = st.selectbox("选择参数行", option_labels)
    # 根据选中的文本找到对应的索引
    selected_index = index_list[option_labels.index(selected_label)]
    if st.button("应用所选参数"):
        selected_row = backtest_df.loc[selected_index]
        st.session_state['body_threshold'] = float(selected_row['body_threshold'])
        st.session_state['vol_ma_period'] = int(selected_row['vol_ma_period'])
        # 注意：break_threshold 是成交量倍数，但历史值很小，我们不自动应用它，仅应用 body 和周期
        st.rerun()
    
    st.markdown("---")
    st.subheader("🖐️ 手动调整")
    
    # 初始化 session_state
    if 'body_threshold' not in st.session_state:
        st.session_state['body_threshold'] = 0.15
    if 'vol_ma_period' not in st.session_state:
        st.session_state['vol_ma_period'] = 15
    if 'volume_multiplier' not in st.session_state:
        st.session_state['volume_multiplier'] = 1.5

    body_threshold = st.number_input("实体阈值", min_value=0.0, max_value=1.0, value=st.session_state['body_threshold'], step=0.01, format="%.2f")
    vol_ma_period = st.number_input("成交量均线周期", min_value=5, max_value=50, value=st.session_state['vol_ma_period'], step=1)
    
    # 确保成交量倍数不小于1.0
    current_mult = st.session_state['volume_multiplier']
    if current_mult < 1.0:
        current_mult = 1.0
        st.session_state['volume_multiplier'] = current_mult
    volume_multiplier = st.number_input("成交量倍数", min_value=1.0, max_value=5.0, value=current_mult, step=0.1, format="%.1f")
    
    fee_rate = st.number_input("手续费率 (单边)", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001, format="%.4f")
    
    st.markdown("---")
    st.subheader("🔍 参数扫描")
    if st.button("开始扫描（耗时约15秒）"):
        body_range = [0.1, 0.15, 0.2, 0.25]
        vol_ma_range = [10, 15, 20]
        vol_mult_range = [1.2, 1.5, 1.8, 2.0]
        
        results = []
        total_combos = len(body_range) * len(vol_ma_range) * len(vol_mult_range)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (b, vp, vm) in enumerate(itertools.product(body_range, vol_ma_range, vol_mult_range)):
            status_text.text(f"扫描中: body={b}, vol_ma={vp}, mult={vm} ({i+1}/{total_combos})")
            df_signal = generate_signals(df_raw, b, vp, vm)
            res = run_backtest(df_signal, fee_rate)
            results.append({
                'body': b,
                'vol_ma': vp,
                'mult': vm,
                '总收益率%': round(res['total_return'], 2),
                '交易次数': res['num_trades'],
                '胜率%': round(res['win_rate'], 1),
                '盈亏比': round(res['rr'], 2) if not np.isnan(res['rr']) else np.nan,
                '夏普(年化)': round(res['sharpe'], 2),
                '最大回撤%': round(res['max_drawdown'], 2)
            })
            progress_bar.progress((i+1)/total_combos)
        
        progress_bar.empty()
        status_text.empty()
        
        result_df = pd.DataFrame(results).sort_values('夏普(年化)', ascending=False)
        st.session_state['scan_results'] = result_df
        st.success("扫描完成！请查看下方表格。")
    
    st.markdown("---")
    st.caption("数据源: OKX ETH-USDT 5分钟")

# ---------------------- 当前策略运行 ----------------------
df = generate_signals(df_raw, body_threshold, vol_ma_period, volume_multiplier)
backtest_res = run_backtest(df, fee_rate)

# ---------------------- 显示实时回测表现 ----------------------
st.subheader("📈 实时回测表现（基于最近数据）")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总收益率", f"{backtest_res['total_return']:.2f}%")
    st.metric("交易次数", backtest_res['num_trades'])
with col2:
    st.metric("胜率", f"{backtest_res['win_rate']:.1f}%")
    st.metric("盈亏比", f"{backtest_res['rr']:.2f}" if not np.isnan(backtest_res['rr']) else "N/A")
with col3:
    st.metric("最大回撤", f"{backtest_res['max_drawdown']:.2f}%")
    st.metric("夏普(年化)", f"{backtest_res['sharpe']:.2f}")
with col4:
    st.metric("获利因子", f"{backtest_res['profit_factor']:.2f}")
    st.metric("手续费率", f"{fee_rate*100:.3f}%")

# 与历史数据对比（仅相同 body 和 vol_ma）
match = backtest_df[(backtest_df['body_threshold'] == body_threshold) & (backtest_df['vol_ma_period'] == vol_ma_period)]
if not match.empty:
    hist = match.iloc[0]
    st.info(f"📊 历史90天回测（相同body/vol_ma）：胜率 {hist['胜率']:.1f}%，总盈利 {hist['总盈利']:.0f}，夏普 {hist['夏普比率']:.2f}，交易数 {hist['交易数']}（注意：历史成交量倍数不同）")

# ---------------------- 资金曲线 ----------------------
if len(backtest_res['equity_curve']) > 1:
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(y=backtest_res['equity_curve'], mode='lines', name='资金曲线', line=dict(color='blue')))
    fig_equity.update_layout(title="资金曲线 (起始10000)", height=300, xaxis_title="交易步数", yaxis_title="账户余额")
    st.plotly_chart(fig_equity, use_container_width=True)

# ---------------------- 实时信号状态 ----------------------
st.subheader("🔔 实时信号状态")
latest_signal = df['signal'].iloc[-1] if not df.empty else 0
if latest_signal == 1:
    st.success("📈 最新信号：买入")
elif latest_signal == -1:
    st.error("📉 最新信号：卖出")
else:
    st.info("⏸️ 最新信号：无")

buy_times = df[df['signal'] == 1]['time']
sell_times = df[df['signal'] == -1]['time']
last_buy = buy_times.max() if not buy_times.empty else None
last_sell = sell_times.max() if not sell_times.empty else None
col1, col2 = st.columns(2)
with col1:
    if last_buy:
        st.success(f"最近买入: {last_buy.strftime('%Y-%m-%d %H:%M')}")
with col2:
    if last_sell:
        st.error(f"最近卖出: {last_sell.strftime('%Y-%m-%d %H:%M')}")

# ---------------------- K线图 ----------------------
st.subheader("📊 最新K线图")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df['time'],
    open=df['o'],
    high=df['h'],
    low=df['l'],
    close=df['c'],
    name='K线'
))
buy_points = df[df['signal'] == 1]
sell_points = df[df['signal'] == -1]
fig.add_trace(go.Scatter(
    x=buy_points['time'],
    y=buy_points['c'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=12, color='green'),
    name='买入'
))
fig.add_trace(go.Scatter(
    x=sell_points['time'],
    y=sell_points['c'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=12, color='red'),
    name='卖出'
))
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- 参数扫描结果 ----------------------
if 'scan_results' in st.session_state:
    st.subheader("🔍 参数扫描结果（按夏普降序）")
    st.dataframe(st.session_state['scan_results'].head(20), use_container_width=True)
    csv = st.session_state['scan_results'].to_csv(index=False)
    st.download_button("📥 下载扫描结果", csv, "scan_results.csv", "text/csv")

# ---------------------- 历史回测数据表格 ----------------------
with st.expander("📋 查看历史90天回测数据（27组参数）"):
    st.dataframe(backtest_df.sort_values('总盈利', ascending=False), use_container_width=True)

st.caption("⚠️ 注意：实时回测基于最近约3.5天数据（1000根5分钟K线），夏普已按日收益率年化。历史表现不代表未来。")
