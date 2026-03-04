#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
以太坊5分钟突破策略自动交易脚本
参数：body=0.15, vol_period=15, break=0.001
交易所：币安合约（U本位） - 测试网
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import sys

# ==================== 配置参数 ====================
SYMBOL = 'ETH/USDT'           # 交易对
TIMEFRAME = '5m'               # 周期
LOOKBACK = 20                  # 突破周期
VOL_MA_PERIOD = 15             # 成交量均线周期
BODY_THRESHOLD = 0.15          # 实体强度阈值
BREAK_THRESHOLD = 0.001        # 突破阈值（0.1%）
RR_RATIO = 2.5                  # 盈亏比
MIN_HOLD = 3                    # 最小持仓K线数
STOP_MULTIPLIER = 0.5           # 止损系数（入场波幅的0.5倍）

# 交易参数
POSITION_SIZE = 0.01            # 每次开仓数量（根据资金调整）
MAX_POSITIONS = 1               # 同时最大持仓数（这里只允许单向持仓）

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eth_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== 交易所初始化（测试网）====================
exchange = ccxt.binance({
    'apiKey': 'YOUR_TESTNET_API_KEY',
    'secret': 'YOUR_TESTNET_SECRET',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',          # 合约交易
        'testnet': True,                   # 使用测试网（正式实盘请改为False）
    }
})

# 如果使用现货，改为 'spot'，并设置 testnet=False
# exchange.options['defaultType'] = 'spot'

# 可选：设置杠杆（合约需要）
# exchange.set_leverage(1, SYMBOL)   # 逐仓1倍杠杆

# ==================== 数据获取与指标计算 ====================
def fetch_ohlcv(limit=100):
    """获取最近limit根K线"""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df
    except Exception as e:
        logger.error(f"获取K线失败: {e}")
        return None

def calculate_indicators(df):
    """计算指标"""
    df = df.copy()
    df['high_max'] = df['high'].rolling(LOOKBACK).max().shift(1)
    df['low_min'] = df['low'].rolling(LOOKBACK).min().shift(1)
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
    df['vol_ma'] = df['volume'].rolling(VOL_MA_PERIOD).mean()
    return df

def check_signal(row):
    """检查最新K线信号"""
    if pd.isna(row['high_max']) or pd.isna(row['low_min']):
        return 0

    in_middle = (row['close'] > row['low_min'] * 1.05) and (row['close'] < row['high_max'] * 0.95)
    strong_body = row['body_ratio'] > BODY_THRESHOLD
    valid_volume = row['volume'] > row['vol_ma']

    # 多头
    if (row['close'] > row['high_max']) and strong_body and valid_volume:
        break_up = (row['close'] - row['high_max']) / row['high_max']
        if break_up > BREAK_THRESHOLD and not in_middle:
            return 1
    # 空头
    if (row['close'] < row['low_min']) and strong_body and valid_volume:
        break_down = (row['low_min'] - row['close']) / row['low_min']
        if break_down > BREAK_THRESHOLD and not in_middle:
            return -1
    return 0

# ==================== 交易执行 ====================
def get_position():
    """获取当前持仓方向（1多头，-1空头，0无持仓）"""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if pos['symbol'] == SYMBOL:
                if float(pos['contracts']) > 0:
                    return 1, float(pos['entryPrice'])
                elif float(pos['contracts']) < 0:
                    return -1, float(pos['entryPrice'])
        return 0, None
    except Exception as e:
        logger.error(f"获取持仓失败: {e}")
        return 0, None

def place_order(signal, price, range_val):
    """开仓"""
    try:
        if signal == 1:
            # 做多
            order = exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
            logger.info(f"开多 @ {price}, 数量 {POSITION_SIZE}, 订单 {order['id']}")
            # 设置止损止盈（限价单）
            stop_loss = price - range_val * STOP_MULTIPLIER
            take_profit = price + (price - stop_loss) * RR_RATIO
            # 止损单（市价）
            exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell', POSITION_SIZE, None, {'stopPrice': stop_loss})
            # 止盈单（限价）
            exchange.create_limit_sell_order(SYMBOL, POSITION_SIZE, take_profit)
            logger.info(f"设置止损 {stop_loss:.2f}, 止盈 {take_profit:.2f}")
        elif signal == -1:
            # 做空
            order = exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
            logger.info(f"开空 @ {price}, 数量 {POSITION_SIZE}, 订单 {order['id']}")
            stop_loss = price + range_val * STOP_MULTIPLIER
            take_profit = price - (stop_loss - price) * RR_RATIO
            exchange.create_order(SYMBOL, 'STOP_MARKET', 'buy', POSITION_SIZE, None, {'stopPrice': stop_loss})
            exchange.create_limit_buy_order(SYMBOL, POSITION_SIZE, take_profit)
            logger.info(f"设置止损 {stop_loss:.2f}, 止盈 {take_profit:.2f}")
    except Exception as e:
        logger.error(f"开仓失败: {e}")

def close_position(direction):
    """手动平仓（用于反向信号平仓）"""
    try:
        if direction == 1:
            exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
            logger.info("反向信号平多")
        elif direction == -1:
            exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
            logger.info("反向信号平空")
    except Exception as e:
        logger.error(f"平仓失败: {e}")

# ==================== 主循环 ====================
def main():
    logger.info("启动自动交易监控...")
    hold = 0          # 持仓K线计数
    last_signal_time = None  # 上次信号时间（防止重复开仓）

    while True:
        try:
            # 获取最新K线
            df = fetch_ohlcv(limit=LOOKBACK+20)
            if df is None:
                time.sleep(5)
                continue

            df = calculate_indicators(df)
            latest = df.iloc[-1]                     # 最新收盘的K线
            current_price = exchange.fetch_ticker(SYMBOL)['last']  # 实时价格

            # 获取当前持仓
            position, entry_price = get_position()

            # 更新持仓计数
            if position != 0:
                hold += 1
            else:
                hold = 0

            # 检查反向信号平仓
            if position != 0:
                signal = check_signal(latest)
                if hold >= MIN_HOLD and signal == -position:
                    close_position(position)
                    position = 0
                    hold = 0

            # 开仓（无持仓时）
            if position == 0:
                signal = check_signal(latest)
                if signal != 0:
                    # 避免同一根K线重复开仓
                    now = latest.name
                    if last_signal_time != now:
                        place_order(signal, latest['close'], latest['range'])
                        last_signal_time = now

            # 每隔5秒检查一次
            time.sleep(5)

        except KeyboardInterrupt:
            logger.info("用户中断，退出程序")
            break
        except Exception as e:
            logger.error(f"主循环异常: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
