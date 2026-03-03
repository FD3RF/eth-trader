import pandas as pd
import numpy as np
import xgboost as xgb
import ta
import matplotlib.pyplot as plt

# =========================
# 参数
# =========================
HOLD_BARS = 5
TRAIN_WINDOW = 8000
TEST_WINDOW = 2000
SLIPPAGE = 0.0005
FEE = 0.0004
THRESHOLD = 0.001  # 预测收益阈值

# =========================
# 读取数据
# =========================
df = pd.read_csv("your_data.csv")
df['ts'] = pd.to_datetime(df['ts'])
df = df.sort_values('ts').reset_index(drop=True)

# =========================
# 特征工程
# =========================
df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)
df['rsi'] = ta.momentum.rsi(df['close'], 14)
df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)

df['ema_dist'] = (df['close'] - df['ema20']) / df['ema20']
df['volatility'] = df['atr'] / df['close']
df['return_1'] = df['close'].pct_change(1)
df['return_3'] = df['close'].pct_change(3)

# =========================
# 真实未来收益标签
# =========================
entry_price = df['open'].shift(-1)
exit_price = df['close'].shift(-HOLD_BARS)

df['future_return'] = (exit_price / entry_price - 1)

df = df.dropna().reset_index(drop=True)

features = [
    'ema_dist',
    'adx',
    'rsi',
    'volatility',
    'return_1',
    'return_3'
]

# =========================
# Walk Forward
# =========================
equity_curve = []
equity = 1000
index_pointer = TRAIN_WINDOW

while index_pointer + TEST_WINDOW < len(df):

    train = df.iloc[index_pointer - TRAIN_WINDOW:index_pointer]
    test = df.iloc[index_pointer:index_pointer + TEST_WINDOW]

    X_train = train[features]
    y_train = train['future_return']

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    X_test = test[features]
    preds = model.predict(X_test)

    for i in range(len(test) - HOLD_BARS - 1):

        predicted = preds[i]

        if predicted > THRESHOLD:

            entry = test.iloc[i]['open'] * (1 + SLIPPAGE)
            exit_ = test.iloc[i + HOLD_BARS]['close'] * (1 - SLIPPAGE)

            ret = (exit_ / entry - 1)
            ret -= FEE * 2

            equity *= (1 + ret)

        equity_curve.append(equity)

    index_pointer += TEST_WINDOW

# =========================
# 结果
# =========================
equity_curve = pd.Series(equity_curve)
returns = equity_curve.pct_change().dropna()

total_return = equity_curve.iloc[-1] - 1000
max_dd = (equity_curve.cummax() - equity_curve).max()
sharpe = returns.mean() / returns.std() * np.sqrt(365*24*12)

print("总收益:", round(total_return,2))
print("最大回撤:", round(max_dd,2))
print("Sharpe:", round(sharpe,2))

plt.figure(figsize=(10,5))
plt.plot(equity_curve)
plt.title("Walk Forward Equity Curve")
plt.show()
