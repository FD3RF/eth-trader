# -*- coding: utf-8 -*-
"""
终极量化交易策略 - 多空阈值分离优化版
作者：AI Assistant
版本：4.0
说明：本代码仅供学习研究，不构成投资建议。请自行承担风险。
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. 数据加载 ======================
def load_data(filepath):
    """加载CSV数据，转换为标准格式"""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={'vol':'volume'}, inplace=True)
    return df[['open','high','low','close','volume']]

# ====================== 2. 特征工程 ======================
def create_features(df):
    """生成全部特征集"""
    df = df.copy()
    
    # 基础价格特征
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # 成交量特征
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # 技术指标
    # EMA
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_cross'] = (df['ema10'] - df['ema20']) / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - 100 / (1 + rs)
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # 布林带
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = 2 * df['bb_std'] / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_mid']) / (2 * df['bb_std'])
    
    # 滞后特征
    for lag in [1,2,3]:
        df[f'close_lag{lag}'] = df['close'].shift(lag)
        df[f'volume_lag{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
        df[f'atr_lag{lag}'] = df['atr'].shift(lag)
    
    # 滚动统计
    for window in [5,10,20]:
        df[f'close_max_{window}'] = df['close'].rolling(window).max()
        df[f'close_min_{window}'] = df['close'].rolling(window).min()
        df[f'close_std_{window}'] = df['close'].rolling(window).std()
    
    # 目标变量：预测未来5根K线是否上涨
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # 删除NaN
    df.dropna(inplace=True)
    return df

# ====================== 3. 数据分割 ======================
def split_data(df, train_ratio=0.6, val_ratio=0.2):
    """按时间顺序分割"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test

# ====================== 4. 训练XGBoost模型 ======================
import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_xgboost(train, val, features):
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )
    
    # 验证集准确率
    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    print(f"✅ 验证集准确率: {acc:.3f}")
    return model

# ====================== 5. 回测函数（支持多空不同阈值） ======================
def backtest(df, model, features, threshold_long, threshold_short, fee_rate, slippage, atr_mult, rr):
    """
    回测函数：根据模型信号交易，采用ATR止损止盈
    - threshold_long: 做多阈值（概率 >= threshold_long）
    - threshold_short: 做空阈值（概率 <= threshold_short）
    """
    df = df.copy()
    df['prob'] = model.predict_proba(df[features])[:, 1]
    df['signal'] = 0
    df.loc[df['prob'] >= threshold_long, 'signal'] = 1      # 做多
    df.loc[df['prob'] <= threshold_short, 'signal'] = -1    # 做空
    
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    equity = [0.0]
    trades = []
    wins = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        open_price = row['open']
        high = row['high']
        low = row['low']
        atr = row['atr']
        
        # 处理现有持仓
        if position == 1:
            stop = entry_price - atr_mult * entry_atr
            take = entry_price + rr * atr_mult * entry_atr
            exit_price = None
            if low <= stop:
                exit_price = stop
            elif high >= take:
                exit_price = take
            if exit_price is not None:
                pnl = (exit_price - entry_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
                position = 0
        
        elif position == -1:
            stop = entry_price + atr_mult * entry_atr
            take = entry_price - rr * atr_mult * entry_atr
            exit_price = None
            if high >= stop:
                exit_price = stop
            elif low <= take:
                exit_price = take
            if exit_price is not None:
                pnl = (entry_price - exit_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
                position = 0
        
        # 新信号（使用上一根K线的信号）
        if prev['signal'] == 1 and position != 1:
            if position == -1:
                pnl = (entry_price - open_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
            position = 1
            entry_price = open_price
            entry_atr = atr
        
        elif prev['signal'] == -1 and position != -1:
            if position == 1:
                pnl = (open_price - entry_price) * (1 - fee_rate * 2) - slippage
                equity.append(equity[-1] + pnl)
                trades.append(pnl)
                if pnl > 0: wins += 1
            position = -1
            entry_price = open_price
            entry_atr = atr
    
    # 最后平仓
    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            pnl = (last_price - entry_price) * (1 - fee_rate * 2) - slippage
        else:
            pnl = (entry_price - last_price) * (1 - fee_rate * 2) - slippage
        equity.append(equity[-1] + pnl)
        trades.append(pnl)
        if pnl > 0: wins += 1
    
    # 计算指标
    total_pnl = sum(trades)
    win_rate = wins / len(trades) if trades else 0
    max_equity = np.maximum.accumulate(equity)
    drawdown = max_equity - equity
    max_dd = np.max(drawdown)
    sharpe = np.mean(trades) / np.std(trades) * np.sqrt(365*24*60/5) if len(trades)>1 and np.std(trades)!=0 else 0
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': len(trades),
        'equity': equity
    }

# ====================== 6. 多空阈值联合优化 ======================
def optimize_thresholds_and_params(val_df, model, features, fee_rate, slippage):
    """
    在验证集上联合优化：多空阈值 + 止损倍数 + 盈亏比
    """
    best_sharpe = -999
    best_params = None  # (th_long, th_short, atr_mult, rr)
    best_result = None
    
    # 定义搜索范围
    long_thresholds = np.arange(0.5, 0.8, 0.05)   # 做多阈值 0.5~0.75
    short_thresholds = np.arange(0.2, 0.5, 0.05)  # 做空阈值 0.2~0.45
    atr_mult_list = [1.0, 1.5, 2.0]
    rr_list = [1.5, 2.0, 2.5, 3.0]
    
    total_combos = len(long_thresholds) * len(short_thresholds) * len(atr_mult_list) * len(rr_list)
    print(f"   总组合数: {total_combos}，正在优化...")
    
    count = 0
    for th_long in long_thresholds:
        for th_short in short_thresholds:
            for atr_mult in atr_mult_list:
                for rr in rr_list:
                    count += 1
                    if count % 100 == 0:
                        print(f"   进度: {count}/{total_combos}")
                    res = backtest(val_df, model, features, th_long, th_short, fee_rate, slippage, atr_mult, rr)
                    if res['sharpe'] > best_sharpe and res['trades'] >= 20:
                        best_sharpe = res['sharpe']
                        best_params = (th_long, th_short, atr_mult, rr)
                        best_result = res
    
    print(f"✅ 最优组合: 做多阈值={best_params[0]:.2f}, 做空阈值={best_params[1]:.2f}, ATR倍数={best_params[2]}, 盈亏比={best_params[3]}, 夏普={best_sharpe:.2f}")
    return best_params, best_result

# ====================== 7. 主程序 ======================
if __name__ == "__main__":
    # ====== 用户设置 ======
    FILE_PATH = "ETHUSDT_5m_last_90days.csv"  # 请修改为您的文件路径
    FEE_RATE = 0.0004      # 0.04% 双向手续费
    SLIPPAGE = 0.5         # 滑点 0.5 USDT
    # =====================
    
    print("=" * 70)
    print("🚀 终极量化交易策略 - 多空阈值分离优化版")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    df_raw = load_data(FILE_PATH)
    print(f"    ✅ 数据加载成功！总K线数: {len(df_raw)}")
    
    # 2. 生成特征
    print("\n[2] 生成特征集...")
    df = create_features(df_raw)
    print(f"    ✅ 特征生成完成！有效K线数: {len(df)}")
    
    # 3. 分割数据集
    print("\n[3] 分割数据集 (训练/验证/测试)...")
    train, val, test = split_data(df, train_ratio=0.6, val_ratio=0.2)
    print(f"    ✅ 训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")
    
    # 特征列
    features = [col for col in df.columns if col not in ['open','high','low','close','volume','target']]
    print(f"    ✅ 特征数量: {len(features)}")
    
    # 4. 训练XGBoost
    print("\n[4] 训练XGBoost模型...")
    model = train_xgboost(train, val, features)
    
    # 5. 联合优化多空阈值和交易参数
    print("\n[5] 在验证集上联合优化多空阈值、止损倍数、盈亏比...")
    best_params, val_res = optimize_thresholds_and_params(val, model, features, FEE_RATE, SLIPPAGE)
    
    th_long, th_short, atr_mult, rr = best_params
    
    print("\n    ✅ 验证集最优结果:")
    print(f"       做多阈值: {th_long:.2f}")
    print(f"       做空阈值: {th_short:.2f}")
    print(f"       ATR倍数: {atr_mult}")
    print(f"       盈亏比: {rr}")
    print(f"       交易次数: {val_res['trades']}")
    print(f"       胜率: {val_res['win_rate']*100:.1f}%")
    print(f"       总盈利: {val_res['total_pnl']:.2f} USDT")
    print(f"       最大回撤: {val_res['max_dd']:.2f} USDT")
    print(f"       夏普比率: {val_res['sharpe']:.2f}")
    
    # 6. 测试集验证
    print("\n[6] 在测试集上验证最终策略...")
    test_res = backtest(test, model, features, th_long, th_short, FEE_RATE, SLIPPAGE, atr_mult, rr)
    
    print("\n" + "=" * 70)
    print("🎯 测试集最终结果")
    print("=" * 70)
    print(f"   做多阈值: {th_long:.2f}")
    print(f"   做空阈值: {th_short:.2f}")
    print(f"   ATR倍数: {atr_mult}")
    print(f"   盈亏比: {rr}")
    print(f"   交易次数: {test_res['trades']}")
    print(f"   胜率: {test_res['win_rate']*100:.1f}%")
    print(f"   总盈利: {test_res['total_pnl']:.2f} USDT")
    print(f"   最大回撤: {test_res['max_dd']:.2f} USDT")
    print(f"   夏普比率: {test_res['sharpe']:.2f}")
    
    if test_res['sharpe'] > 1.0:
        print("\n✨ 测试结果优秀！可以考虑进一步优化或实盘模拟。")
    elif test_res['sharpe'] > 0:
        print("\n📊 测试结果为正收益，但稳定性一般。可尝试调整特征或参数范围。")
    else:
        print("\n⚠️ 测试结果为负，策略可能无效。请检查特征或逻辑。")
