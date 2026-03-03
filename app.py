# 假设 df 已包含原始OHLCV数据
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. 特征工程
def create_features(df):
    df = df.copy()
    # 基础特征
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['high_low_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 技术指标（复用您已有的）
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_slow'] = df['close'].ewm(span=20).mean()
    df['rsi'] = ...  # 计算RSI
    df['atr'] = ...  # 计算ATR
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    
    # 滞后特征
    for lag in [1,2,3]:
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
        df[f'macd_lag{lag}'] = df['macd'].shift(lag)
    
    # 目标变量：未来5根收益为正
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # 删除NaN行
    df.dropna(inplace=True)
    return df

df = create_features(df)

# 2. 划分数据集
train_val = df.iloc[:int(0.8*len(df))]
test = df.iloc[int(0.8*len(df)):]

# 再从train_val中划分训练和验证（按时间）
train = train_val.iloc[:int(0.75*len(train_val))]
val = train_val.iloc[int(0.75*len(train_val)):]

# 3. 定义特征
features = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]

# 4. 训练
model = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, early_stopping_rounds=20)
model.fit(train[features], train['target'],
          eval_set=[(val[features], val['target'])],
          verbose=False)

# 5. 在测试集上生成信号
test['prob'] = model.predict_proba(test[features])[:,1]
test['signal'] = 0
test.loc[test['prob'] > 0.6, 'signal'] = 1
test.loc[test['prob'] < 0.4, 'signal'] = -1

# 6. 回测（使用您之前的逻辑）
# ...
