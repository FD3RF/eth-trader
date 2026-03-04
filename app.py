import pandas as pd

# ================== 参数（可改）==================
RR = 2.0          # 盈亏比 1:2
SL_BUFFER = 5     # 止损缓冲USDT
# ================================================

df = pd.read_csv('ETHUSDT_5m_last_90days.csv')
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df = df[['ts', 'open', 'high', 'low', 'close']].copy()

# 上一根K线
df['prev_open'] = df['open'].shift(1)
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)
df['prev_close'] = df['close'].shift(1)

# 多头吞没突破信号
df['long'] = (
    (df['prev_close'] < df['prev_open']) &      # 前阴
    (df['close'] > df['open']) &                # 当前阳
    (df['open'] <= df['prev_close']) &          # 实体吞没
    (df['close'] >= df['prev_open']) &
    (df['close'] > df['prev_high'])             # 突破前高
)

# 空头吞没突破信号
df['short'] = (
    (df['prev_close'] > df['prev_open']) &
    (df['close'] < df['open']) &
    (df['open'] >= df['prev_close']) &
    (df['close'] <= df['prev_open']) &
    (df['close'] < df['prev_low'])
)

# ================== 回测 ==================
trades = []
position = 0
entry = 0
sl = 0
tp = 0
max_dd = 0
equity = 0
peak = 0

for i in range(1, len(df)):
    row = df.iloc[i]
    
    if position == 0:
        if row['long']:
            entry = row['close'] + 0.1      # 下一根近似开盘（实盘可改open）
            sl = row['low'] - SL_BUFFER
            tp = entry + RR * (entry - sl)
            position = 1
        elif row['short']:
            entry = row['close'] - 0.1
            sl = row['high'] + SL_BUFFER
            tp = entry - RR * (sl - entry)
            position = -1
    
    else:
        high = row['high']
        low = row['low']
        close = row['close']
        
        if position == 1:  # 多单
            if low <= sl:
                pnl = sl - entry
                trades.append(pnl)
                position = 0
            elif high >= tp:
                pnl = tp - entry
                trades.append(pnl)
                position = 0
        elif position == -1:  # 空单
            if high >= sl:
                pnl = entry - sl
                trades.append(pnl)
                position = 0
            elif low <= tp:
                pnl = entry - tp
                trades.append(pnl)
                position = 0
        
        # 更新回撤
        equity += pnl if 'pnl' in locals() else 0
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

# ================== 结果 ==================
trades_df = pd.DataFrame({'pnl': trades})
total_trades = len(trades_df)
win_rate = (trades_df['pnl'] > 0).mean() * 100 if total_trades > 0 else 0
profit_factor = trades_df[trades_df['pnl']>0]['pnl'].sum() / abs(trades_df[trades_df['pnl']<0]['pnl'].sum()) if len(trades_df[trades_df['pnl']<0]) > 0 else float('inf')
total_pnl = trades_df['pnl'].sum()

print("✅ 纯K吞没突破战法【真实回测结果】- 你的90天数据")
print(f"交易次数: {total_trades}")
print(f"胜率: {win_rate:.2f}%")
print(f"获利因子: {profit_factor:.2f}")
print(f"总收益（点）: {total_pnl:.0f}")
print(f"最大回撤（点）: {max_dd:.0f}")
print(f"平均盈利: {trades_df[trades_df['pnl']>0]['pnl'].mean():.2f}")
print(f"平均亏损: {trades_df[trades_df['pnl']<0]['pnl'].mean():.2f}")
print("\n运行成功！把上面完整输出贴给我，我立刻帮你优化（加4H过滤、锤子线、1:3版本等）")
