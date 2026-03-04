import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================== 参数设置（可改）==================
RISK_REWARD = 2.0          # 1:2 盈亏比
SL_BUFFER = 5              # 止损额外缓冲（USDT），防滑点
INITIAL_CAPITAL = 10000    # 初始资金（USDT），仅用于画权益曲线
RISK_PER_TRADE = 0.01      # 每笔风险1%（实际用点数计算盈亏）
# ===================================================

# 读取数据
df = pd.read_csv('ETHUSDT_5m_last_90days.csv')
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df = df[['ts', 'open', 'high', 'low', 'close']].copy()
df.columns = ['datetime', 'open', 'high', 'low', 'close']

# 计算吞没突破信号
df['prev_open'] = df['open'].shift(1)
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)
df['prev_close'] = df['close'].shift(1)

# 多头信号
df['long_signal'] = (
    (df['prev_close'] < df['prev_open']) &                    # 前K阴线
    (df['close'] > df['open']) &                             # 当前K阳线
    (df['open'] <= df['prev_close']) &                       # 实体吞没
    (df['close'] >= df['prev_open']) &
    (df['close'] > df['prev_high'])                          # 突破前高
)

# 空头信号
df['short_signal'] = (
    (df['prev_close'] > df['prev_open']) &                   # 前K阳线
    (df['close'] < df['open']) &                             # 当前K阴线
    (df['open'] >= df['prev_close']) &                       # 实体吞没
    (df['close'] <= df['prev_open']) &
    (df['close'] < df['prev_low'])                           # 突破前低
)

# ================== 回测逻辑 ==================
trades = []
equity = [INITIAL_CAPITAL]
position = 0  # 1=多  -1=空  0=空仓
entry_price = 0
sl_price = 0
tp_price = 0

for i in range(1, len(df)):
    if position == 0:
        # 开多
        if df.iloc[i]['long_signal']:
            entry_price = df.iloc[i]['close']          # 下一根开盘用收盘价近似（实盘可改成下一根open）
            sl_price = df.iloc[i]['low'] - SL_BUFFER
            tp_price = entry_price + RISK_REWARD * (entry_price - sl_price)
            position = 1
            entry_idx = i
        
        # 开空
        elif df.iloc[i]['short_signal']:
            entry_price = df.iloc[i]['close']
            sl_price = df.iloc[i]['high'] + SL_BUFFER
            tp_price = entry_price - RISK_REWARD * (sl_price - entry_price)
            position = -1
            entry_idx = i
    
    else:
        # 检查平仓（实盘用实时价格，这里用收盘价判断）
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        current_close = df.iloc[i]['close']
        
        if position == 1:   # 多单
            if current_low <= sl_price:
                pnl = sl_price - entry_price
                trades.append({'type':'long', 'pnl':pnl, 'exit':'SL'})
                position = 0
            elif current_high >= tp_price:
                pnl = tp_price - entry_price
                trades.append({'type':'long', 'pnl':pnl, 'exit':'TP'})
                position = 0
            elif i - entry_idx > 48:  # 最多持仓4小时（48根5分钟），防止过夜
                pnl = current_close - entry_price
                trades.append({'type':'long', 'pnl':pnl, 'exit':'Timeout'})
                position = 0
        
        elif position == -1:  # 空单
            if current_high >= sl_price:
                pnl = entry_price - sl_price
                trades.append({'type':'short', 'pnl':pnl, 'exit':'SL'})
                position = 0
            elif current_low <= tp_price:
                pnl = entry_price - tp_price
                trades.append({'type':'short', 'pnl':pnl, 'exit':'TP'})
                position = 0
            elif i - entry_idx > 48:
                pnl = entry_price - current_close
                trades.append({'type':'short', 'pnl':pnl, 'exit':'Timeout'})
                position = 0
        
        equity.append(equity[-1] + pnl * 1)  # 1张合约，每点1 USDT（实际按仓位大小调整）

# ================== 统计结果 ==================
trades_df = pd.DataFrame(trades)
total_trades = len(trades_df)
win_trades = len(trades_df[trades_df['pnl'] > 0])
win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
profit_factor = trades_df[trades_df['pnl']>0]['pnl'].sum() / abs(trades_df[trades_df['pnl']<0]['pnl'].sum()) if len(trades_df[trades_df['pnl']<0]) > 0 else float('inf')
total_pnl = trades_df['pnl'].sum()
max_dd = (pd.Series(equity).cummax() - pd.Series(equity)).max()

print(f"📊 纯K吞没突破战法回测结果（90天5分钟数据）")
print(f"交易次数: {total_trades}")
print(f"胜率: {win_rate:.2f}%")
print(f"获利因子: {profit_factor:.2f}")
print(f"总收益（点）: {total_pnl:.0f}")
print(f"最大回撤（点）: {max_dd:.0f}")
print(f"平均盈利: {trades_df[trades_df['pnl']>0]['pnl'].mean():.2f}")
print(f"平均亏损: {trades_df[trades_df['pnl']<0]['pnl'].mean():.2f}")

# 画权益曲线（和Streamlit一模一样风格）
plt.figure(figsize=(12,6))
plt.plot(equity, label='权益曲线', color='cyan', linewidth=2)
plt.title('纯K吞没突破战法权益曲线（ETHUSDT 5m）')
plt.ylabel('账户权益 (USDT)')
plt.xlabel('交易笔数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
