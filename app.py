import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==============================
# 配置交易所与参数
# ==============================

exchange = ccxt.binance()  # 使用 Binance 公共数据
symbol = 'ETH/USDT'
timeframe = '5m'
limit = 100  # 获取最近100根K线

# ==============================
# 获取K线数据函数
# ==============================

def fetch_klines():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ==============================
# 绘图函数（K线 + 成交量）
# ==============================

def plot_data(df):
    plt.figure()
    plt.plot(df['timestamp'], df['close'])
    plt.title('ETH 5-Min Close Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(df['timestamp'], df['volume'])
    plt.title('Volume')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==============================
# 主盯盘循环（实时刷新）
# ==============================

def monitor(interval=60):
    while True:
        df = fetch_klines()
        print(df.tail())  # 打印最新几行数据
        plot_data(df)
        time.sleep(interval)  # 每分钟刷新一次

# ==============================
# 启动盯盘
# ==============================

if __name__ == "__main__":
    monitor(interval=60)
