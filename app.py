# -*- coding: utf-8 -*-
"""
实盘框架（模板）
纯K信号 + 风控 + 下单接口（需你接API）
不含任何交易所凭据
"""

import time
import numpy as np
import pandas as pd

# ================================
# 风控参数
# ================================
RISK_PER_TRADE = 0.01   # 单笔风险 1%
RR = 2.5                # 盈亏比
MIN_HOLD = 3             # 最小持仓K线

# ================================
# 信号生成（纯K）
# ================================
def generate_signal(df):
    row = df.iloc[-1]
    strong_body = row['body_ratio'] > 0.35

    if (row['close'] > row['high_max']) and strong_body and (row['close'] > row['open']):
        return 1  # 多
    if (row['close'] < row['low_min']) and strong_body and (row['close'] < row['open']):
        return -1 # 空
    return 0

# ================================
# 仓位计算（动态）
# ================================
def calc_position_size(capital, atr):
    return (capital * RISK_PER_TRADE) / (atr * 2 + 1e-9)

# ================================
# 模拟下单接口（需替换为实盘API）
# ================================
def place_order(side, size):
    """
    side: 'buy' 或 'sell'
    size: 下单数量
    """
    print(f"[ORDER] side={side}, size={size}")
    # 这里接你的交易所API
    return True

def close_position():
    print("[CLOSE] 平仓")
    return True

# ================================
# 实盘循环（伪框架）
# ================================
def trading_loop(api, capital=10000):
    """
    api: 你自己的行情/数据接口
    """
    position = 0
    entry = 0
    hold = 0

    while True:
        df = api.get_latest()  # 你需实现：获取最新K线DataFrame
        df = build_features(df)

        signal = generate_signal(df)
        row = df.iloc[-1]
        atr = row['atr']

        if position != 0:
            hold += 1
        else:
            hold = 0

        # ===== 平仓逻辑 =====
        if position == 1:
            stop = entry - atr * 1.5
            take = entry + (entry - stop) * RR

            if row['low'] <= stop or (hold >= MIN_HOLD and signal == -1):
                close_position()
                position = 0

        if position == -1:
            stop = entry + atr * 1.5
            take = entry - (stop - entry) * RR

            if row['high'] >= stop or (hold >= MIN_HOLD and signal == 1):
                close_position()
                position = 0

        # ===== 开仓 =====
        if position == 0 and signal != 0:
            size = calc_position_size(capital, atr)

            if signal == 1:
                place_order('buy', size)
                entry = row['close']
                position = 1

            elif signal == -1:
                place_order('sell', size)
                entry = row['close']
                position = -1

        time.sleep(1)  # 控制频率

# ================================
# 纯K特征构建
# ================================
def build_features(df):
    df = df.copy()
    lookback = 20

    df['high_max'] = df['high'].rolling(lookback).max().shift(1)
    df['low_min'] = df['low'].rolling(lookback).min().shift(1)

    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-9)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)

    df['atr'] = tr.rolling(14).mean()
    df.dropna(inplace=True)
    return df

# ================================
# 使用说明
# ================================
"""
使用步骤：

1. 实现行情接口：
   api.get_latest() -> 返回 DataFrame（含 open/high/low/close/vol）

2. 实现下单接口：
   place_order / close_position 接入交易所 API

3. 小资金测试：
   先用模拟账户

4. 风控：
   单笔 1%
"""

print("实盘框架加载完成")
