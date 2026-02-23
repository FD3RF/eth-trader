"""
OKX 实时监控系统 (WebSocket 版)
- 多周期 (5m/15m/1h/4h) 实时K线
- 检测条件: VWAP突破 + EMA9上穿EMA21 + 成交量放大
- 基于收盘K线，避免未来数据
- 信号冷却，避免重复触发
- 指数退避重连，断网自动恢复
- 增强重连日志 + 信号统计
- Streamlit 仪表盘，每10秒自动刷新
- 只监控，不交易
"""

import streamlit as st
import pandas as pd
import pandas_ta as ta
import threading
import queue
import time
import json
import websocket
from datetime import datetime, timedelta
from collections import defaultdict
import random

# ==================== 配置 ====================
SYMBOL = 'BTC-USDT'                     # OKX 格式
TIMEFRAMES = ['5m', '15m', '1h', '4h']   # 监控周期
LIMIT = 200                              # 每个周期保留K线数量
REFRESH_INTERVAL = 10                     # UI刷新秒数
SIGNAL_COOLDOWN = 60                      # 同一周期信号冷却秒数

# WebSocket 地址（公共频道）
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# 全局变量（线程安全）
data_store = {}          # 格式: data_store[tf] = pd.DataFrame
data_lock = threading.RLock()  # 可重入读写锁
kline_queue = queue.Queue(maxsize=1000)  # 消息队列
ws_running = True
ws_instance = None       # 当前WebSocket对象
ws_lock = threading.Lock()
reconnect_attempt = 0     # 当前重连尝试次数
reconnect_log = []        # 重连历史记录 [(timestamp, attempt), ...]
reconnect_log_lock = threading.Lock()

# ==================== WebSocket 客户端（带指数退避重连）====================
def start_websocket():
    """启动WebSocket连接（非阻塞），返回ws对象"""
    def on_message(ws, message):
        try:
            msg = json.loads(message)
            # 只处理candle频道数据
            if 'data' in msg and 'arg' in msg and msg['arg']['channel'].startswith('candle'):
                tf = msg['arg']['channel'].replace('candle', '')  # 如 '5m'
                for item in msg['data']:
                    # OKX candle: [ts, o, h, l, c, vol, ...]
                    if len(item) < 6:
                        continue
                    ts = int(item[0])
                    o = float(item[1])
                    h = float(item[2])
                    l = float(item[3])
                    c = float(item[4])
                    vol = float(item[5])
                    # 放入队列（非阻塞）
                    try:
                        kline_queue.put_nowait((tf, ts, o, h, l, c, vol))
                    except queue.Full:
                        pass  # 队列满则丢弃，避免阻塞
        except Exception as e:
            print(f"[WS] 消息解析错误: {e}")

    def on_error(ws, error):
        print(f"[WS] 错误: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"[WS] 连接关闭 (code: {close_status_code})")
        global reconnect_attempt
        # 记录重连日志
        with reconnect_log_lock:
            reconnect_log.append((datetime.now(), reconnect_attempt + 1))
            if len(reconnect_log) > 5:
                reconnect_log.pop(0)
        # 触发重连（指数退避）
        with ws_lock:
            if ws_running:
                delay = min(30, (2 ** reconnect_attempt) + random.uniform(0, 1))
                print(f"[WS] 将在 {delay:.1f} 秒后尝试重连...")
                time.sleep(delay)
                reconnect_attempt += 1
                # 关闭旧连接（安全）
                if ws_instance:
                    try:
                        ws_instance.close()
                    except:
                        pass
                # 重连
                new_ws = start_websocket()
                # 更新全局引用
                globals()['ws_instance'] = new_ws
            else:
                print("[WS] 停止重连")

    def on_open(ws):
        """订阅K线频道"""
        global reconnect_attempt
        reconnect_attempt = 0  # 重置重连计数
        args = [{"channel": f"candle{tf}", "instId": SYMBOL} for tf in TIMEFRAMES]
        subscribe_msg = {"op": "subscribe", "args": args}
        ws.send(json.dumps(subscribe_msg))
        print(f"[WS] 已订阅 {SYMBOL} 的 {TIMEFRAMES}")

    # 创建WebSocketApp
    ws = websocket.WebSocketApp(WS_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    # 运行在独立线程中
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()
    return ws

# ==================== 数据处理线程 ====================
def data_processor():
    """从队列消费K线，更新data_store"""
    # 初始化每个周期的空DataFrame
    with data_lock:
        for tf in TIMEFRAMES:
            data_store[tf] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    while ws_running:
        try:
            tf, ts, o, h, l, c, vol = kline_queue.get(timeout=1)
        except queue.Empty:
            continue

        dt = pd.to_datetime(ts, unit='ms')

        with data_lock:
            df = data_store[tf]
            # 检查是否已存在该时间戳
            if not df.empty and df['timestamp'].iloc[-1] == dt:
                # 更新最后一根K线（最新推送可能是更新）
                df.iloc[-1] = [dt, o, h, l, c, vol]
            else:
                # 新增K线
                new_row = pd.DataFrame([[dt, o, h, l, c, vol]], columns=df.columns)
                df = pd.concat([df, new_row], ignore_index=True)
                # 限制长度
                if len(df) > LIMIT:
                    df = df.iloc[-LIMIT:]
                data_store[tf] = df
    print("[Processor] 停止")

# 启动后台线程
processor_thread = threading.Thread(target=data_processor, daemon=True)
processor_thread.start()
ws_instance = start_websocket()   # 启动WebSocket

# ==================== 指标计算（基于已收盘K线）====================
def calculate_indicators(df):
    """对DataFrame计算技术指标，返回新的DataFrame（避免修改原数据）"""
    if df.empty or len(df) < 30:
        return df.copy()

    df = df.copy()  # 避免影响原数据
    # VWAP
    df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    # EMA
    df['EMA9'] = ta.ema(df['close'], length=9)
    df['EMA21'] = ta.ema(df['close'], length=21)
    # 前5根成交量均值（排除当前K线）
    df['VOL_MA5'] = df['volume'].rolling(5).mean().shift(1)
    return df

def check_signal(df, last_signal_time):
    """
    检测最新一根已收盘K线是否满足条件
    参数:
        df: 包含指标的DataFrame（至少有两根K线）
        last_signal_time: 上次该周期信号时间戳（用于冷却）
    返回: (满足, 入场价, 止损价, 盈亏比, 目标价, 当前K线时间戳)
    """
    if df.empty or len(df) < 3:   # 至少需要两根已收盘 + 一根最新
        return False, None, None, None, None, None

    # 取倒数第二根作为已收盘K线
    closed = df.iloc[-2]
    prev = df.iloc[-3] if len(df) > 2 else None

    # 检查冷却
    kline_time = closed.name  # 假设索引是timestamp
    if last_signal_time is not None and (kline_time - last_signal_time).total_seconds() < SIGNAL_COOLDOWN:
        return False, None, None, None, None, kline_time

    # 条件1: 收盘价 > VWAP (VWAP需存在)
    if pd.isna(closed['VWAP']):
        cond1 = False
    else:
        cond1 = closed['close'] > closed['VWAP']

    # 条件2: EMA9上穿EMA21
    cond2 = False
    if prev is not None and not pd.isna(closed['EMA9']) and not pd.isna(closed['EMA21']):
        cond2 = (closed['EMA9'] > closed['EMA21']) and (prev['EMA9'] <= prev['EMA21'])

    # 条件3: 成交量 > 前5根均值
    if pd.isna(closed['VOL_MA5']):
        cond3 = False
    else:
        cond3 = closed['volume'] > closed['VOL_MA5']

    if cond1 and cond2 and cond3:
        entry = closed['close']
        stop_loss = closed['EMA21']
        if stop_loss >= entry:   # 止损必须低于入场
            return False, None, None, None, None, kline_time
        risk = entry - stop_loss
        target = entry + 2 * risk
        rr = 2.0
        return True, entry, stop_loss, rr, target, kline_time
    else:
        return False, None, None, None, None, kline_time

# ==================== Streamlit UI ====================
def init_session_state():
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []          # 存储历史信号（列表）
    if 'last_signal_times' not in st.session_state:
        st.session_state.last_signal_times = {}       # 每个周期上次信号时间戳

def add_signal(tf, entry, stop, rr, target, signal_time):
    """添加信号到历史记录"""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    signal_time_str = signal_time.strftime('%Y-%m-%d %H:%M:%S') if signal_time else now_str
    st.session_state.signal_history.append({
        '触发时间': now_str,
        '周期': tf,
        'K线时间': signal_time_str,
        '入场价': f"{entry:.2f}",
        '止损价': f"{stop:.2f}",
        '目标价': f"{target:.2f}",
        '盈亏比': f"{rr:.2f}"
    })
    # 只保留最近20条
    if len(st.session_state.signal_history) > 20:
        st.session_state.signal_history = st.session_state.signal_history[-20:]

def render_sidebar():
    """渲染侧边栏：信号统计 + 重连日志"""
    st.sidebar.header("📊 信号统计")
    # 计算每个周期的信号次数
    if st.session_state.signal_history:
        df_signals = pd.DataFrame(st.session_state.signal_history)
        signal_counts = df_signals['周期'].value_counts().to_dict()
        for tf in TIMEFRAMES:
            count = signal_counts.get(tf, 0)
            st.sidebar.metric(f"{tf} 信号次数", count)
    else:
        for tf in TIMEFRAMES:
            st.sidebar.metric(f"{tf} 信号次数", 0)

    st.sidebar.header("🔄 重连日志")
    with reconnect_log_lock:
        if reconnect_log:
            for dt, attempt in reversed(reconnect_log):
                st.sidebar.text(f"{dt.strftime('%H:%M:%S')} 尝试 #{attempt}")
        else:
            st.sidebar.text("暂无重连记录")

    # 当前连接状态
    with ws_lock:
        ws_status = "🟢 已连接" if ws_instance and ws_instance.sock and ws_instance.sock.connected else "🔴 断开"
    st.sidebar.caption(f"WebSocket: {ws_status}")
    st.sidebar.caption(f"重连尝试: {reconnect_attempt}")

def render_dashboard():
    """渲染仪表盘主要内容"""
    # 获取最新数据快照（浅拷贝，只拷贝必要的列）
    with data_lock:
        snapshot = {}
        for tf, df in data_store.items():
            if not df.empty:
                # 只复制需要的列，减少内存
                snapshot[tf] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            else:
                snapshot[tf] = df.copy()

    # 布局
    cols = st.columns(len(TIMEFRAMES))

    # 终端打印占位
    terminal = st.empty()

    # 处理每个周期
    for idx, tf in enumerate(TIMEFRAMES):
        df = snapshot.get(tf, pd.DataFrame())
        with cols[idx]:
            st.subheader(f"周期: {tf}")

            if df.empty or len(df) < 30:
                st.warning("数据收集中...")
                continue

            # 计算指标
            df = calculate_indicators(df)

            if len(df) < 2:
                st.warning("等待更多数据")
                continue

            # 已收盘K线（倒数第二根）
            last_closed = df.iloc[-2]
            # 当前最新K线（可能未完成）
            current_kline = df.iloc[-1]

            # 当前价格（最新收盘价）
            current_price = current_kline['close']

            st.metric("当前价格", f"{current_price:.2f}")

            # 趋势状态（基于已收盘）
            trend = "多头" if (last_closed['close'] > last_closed.get('VWAP', 0) and 
                               last_closed.get('EMA9', 0) > last_closed.get('EMA21', 0)) else "震荡/空头"
            st.write(f"趋势状态: **{trend}**")

            # 指标状态灯（基于已收盘）
            cond1 = last_closed['close'] > last_closed.get('VWAP', float('inf')) if not pd.isna(last_closed.get('VWAP')) else False
            cond2 = (last_closed.get('EMA9', 0) > last_closed.get('EMA21', 0)) if not pd.isna(last_closed.get('EMA9')) else False
            cond3 = last_closed['volume'] > last_closed.get('VOL_MA5', float('inf')) if not pd.isna(last_closed.get('VOL_MA5')) else False

            st.markdown("**指标状态**")
            cola, colb, colc = st.columns(3)
            with cola:
                color = "🟢" if cond1 else "🔴"
                st.write(f"{color} VWAP突破")
            with colb:
                color = "🟢" if cond2 else "🔴"
                st.write(f"{color} EMA金叉")
            with colc:
                color = "🟢" if cond3 else "🔴"
                st.write(f"{color} 量能确认")

            # 检测信号
            last_signal_time = st.session_state.last_signal_times.get(tf)
            signal, entry, stop, rr, target, kline_time = check_signal(df, last_signal_time)

            if signal:
                # 更新上次信号时间
                st.session_state.last_signal_times[tf] = kline_time

                # 显示盈亏比
                st.write("**盈亏比**")
                rr_bar = min(rr, 5) / 5
                st.progress(rr_bar, text=f"{rr:.2f} : 1")
                st.caption(f"目标: {target:.2f} | 止损: {stop:.2f}")

                # 添加到历史
                add_signal(tf, entry, stop, rr, target, kline_time)

                # 终端打印（同时显示在页面）
                with terminal.container():
                    st.code(f"""
[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 周期 {tf}
当前信号: 买入计划
入场价: {entry:.2f}
止损价: {stop:.2f}
目标价: {target:.2f}
预期盈亏比: {rr:.2f}
                    """)
                # 同时打印到系统终端
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {tf} 买入信号 入场:{entry:.2f} 止损:{stop:.2f} 目标:{target:.2f}")
            else:
                st.write("**盈亏比**: 无信号")
                st.progress(0, text="无信号")

    # 信号历史表
    st.subheader("📋 最近信号记录")
    if st.session_state.signal_history:
        hist_df = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("暂无信号记录")

    st.caption(f"最后刷新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    st.set_page_config(page_title="OKX WebSocket 监控", layout="wide")
    st.title("📡 OKX 实时监控 (WebSocket) - 高爆发策略")
    st.caption(f"只监控不交易 | 基于收盘K线 | 数据刷新 {REFRESH_INTERVAL}秒 | 信号冷却 {SIGNAL_COOLDOWN}秒")

    init_session_state()
    render_sidebar()

    # 使用 while True 循环实现自动刷新（只读仪表盘可接受）
    placeholder = st.empty()
    while True:
        with placeholder.container():
            render_dashboard()
        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("退出中...")
        global ws_running
        ws_running = False
        if ws_instance:
            ws_instance.close()
