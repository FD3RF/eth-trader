import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import vectorbt as vbt

# -------------------- 配置 --------------------
SYMBOL = "ETH-USDT"
INTERVAL = "5m"          # K线周期
LIMIT = 300              # 每次请求最大K线数
START_DATE = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")
FEES = 0.001             # 手续费 0.1%
SL_ATR = 1.5             # 止损倍数
TP_ATR = 2.5             # 止盈倍数
# ---------------------------------------------

def fetch_okx_candles(symbol, bar, limit, start_date, end_date):
    """
    获取 OKX 历史K线数据（支持翻页）
    返回 DataFrame，包含字段：时间、开、高、低、收、成交量
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    # 第一次请求不带 before，从最新开始往前取
    while True:
        resp = requests.get(url)
        if resp.status_code != 200:
            break
        data = resp.json()
        if data.get("code") != "0" or not data["data"]:
            break
        candles = data["data"]
        # 转换为 DataFrame
        df_chunk = pd.DataFrame(candles, columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"])
        df_chunk["ts"] = pd.to_datetime(df_chunk["ts"].astype(float), unit="ms")
        all_data.append(df_chunk)
        # 获取最早的时间戳，用于下一次翻页
        earliest_ts = int(candles[-1][0])
        if earliest_ts <= start_ts:
            break
        # 更新 URL，加上 before 参数
        url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}&before={earliest_ts}"
        time.sleep(0.2)  # 避免请求过快
    
    if not all_data:
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # 转换为数值
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col])
    # 过滤时间范围
    df = df[(df["ts"] >= start_date) & (df["ts"] <= end_date)]
    return df

def calculate_indicators(df):
    """计算所有技术指标（修正版）"""
    df = df.copy()
    # EMA
    df["ema_fast"] = df["c"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=26, adjust=False).mean()
    
    # RSI (Wilder平滑)
    delta = df["c"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df["c"].ewm(span=12, adjust=False).mean()
    exp26 = df["c"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp12 - exp26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # 布林带
    df["bb_mid"] = df["c"].rolling(20).mean()
    df["bb_std"] = df["c"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    
    # 成交量均线
    df["vol_ma"] = df["v"].rolling(10).mean()
    
    # ATR (真实波幅)
    high_low = df["h"] - df["l"]
    high_close = (df["h"] - df["c"].shift()).abs()
    low_close = (df["l"] - df["c"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    
    return df

def generate_signals(df, ls_ratio=1.0):
    """
    生成交易信号（基于当前行的收盘价数据，假设在下一根K线开盘执行）
    返回信号序列：1=多头，-1=空头，0=无持仓
    """
    signals = []
    for i in range(len(df)):
        if i < 50:  # 需要足够数据计算指标
            signals.append(0)
            continue
        row = df.iloc[i]
        score = 50
        
        # EMA
        if row["ema_fast"] > row["ema_slow"]:
            score += 20
        else:
            score -= 18
        
        # RSI
        if 30 < row["rsi"] < 70:
            score += 10
        elif row["rsi"] > 75:
            score -= 10
        elif row["rsi"] < 25:
            score += 5
        
        # MACD hist
        if row["macd_hist"] > 0:
            score += 12
        else:
            score -= 12
        
        # 极点突破（放量倍数1.5）
        extreme = False
        if row["c"] > row["bb_upper"] and row["v"] > row["vol_ma"] * 1.5:
            extreme = True
            score += 20
        elif row["c"] < row["bb_lower"] and row["v"] > row["vol_ma"] * 1.5:
            extreme = True
            score += 20
        
        # 多空比（固定值，因为历史数据难获取）
        # 这里暂时忽略，保留以后扩展
        
        prob = max(min(score, 95), 5)
        trend = 1 if row["ema_fast"] > row["ema_slow"] else -1
        if trend == 1 and extreme and prob > 60:
            signals.append(1)
        elif trend == -1 and extreme and prob < 40:
            signals.append(-1)
        else:
            signals.append(0)
    return signals

def run_backtest():
    print("正在获取历史数据...")
    df = fetch_okx_candles(SYMBOL, INTERVAL, LIMIT, START_DATE, END_DATE)
    if df is None or len(df) < 100:
        print("数据获取失败或数据量不足")
        return
    
    print(f"共获取 {len(df)} 根K线")
    print("计算指标...")
    df = calculate_indicators(df)
    print("生成信号...")
    df['signal'] = generate_signals(df)
    
    # 只保留有信号的区域
    df = df.dropna().reset_index(drop=True)
    
    # 准备回测数据
    price = df['c'].values
    signal = df['signal'].values
    
    # 使用 vectorbt 的 from_signals 方法
    # 我们需要定义入场和平仓信号：
    # 多头入场：signal 从 0 变为 1 的位置
    # 多头平仓：signal 从 1 变为 0 或 -1 的位置
    # 空头入场：signal 从 0 变为 -1 的位置
    # 空头平仓：signal 从 -1 变为 0 或 1 的位置
    
    # 将信号转换为布尔序列
    long_entries = (signal == 1) & (np.roll(signal, 1) != 1)
    long_entries[0] = False
    long_exits = (signal != 1) & (np.roll(signal, 1) == 1)
    
    short_entries = (signal == -1) & (np.roll(signal, 1) != -1)
    short_entries[0] = False
    short_exits = (signal != -1) & (np.roll(signal, 1) == -1)
    
    # 构建 Portfolio
    pf = vbt.Portfolio.from_signals(
        price,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10000,
        fees=FEES,
        freq='5min'  # 指定频率，用于计算年化等
    )
    
    # 输出绩效统计
    print("\n========== 回测结果 ==========")
    stats = pf.stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 绘制权益曲线
    fig = pf.plot().show()
    return pf, df

if __name__ == "__main__":
    pf, df = run_backtest()
