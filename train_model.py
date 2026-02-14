import ccxt
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. 获取历史数据（过去3年，约26000根1小时K线）
exchange = ccxt.binance()
symbol = 'ETH/USDT'
timeframe = '1h'
limit = 20000  # 获取足够数据，可调整

print("正在下载数据，请稍候...")
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 2. 计算技术指标
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['ma20'] = df['close'].rolling(20).mean()
df['ma60'] = df['close'].rolling(60).mean()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()               # MACD线
df['macd_signal'] = macd.macd_signal() # 信号线
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
df['atr_pct'] = df['atr'] / df['close'] * 100
adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
df['adx'] = adx.adx()

# 3. 定义特征
features = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
df = df.dropna(subset=features)

# 4. 标签：未来6小时涨幅 > 2% 为1，否则0
df['target'] = (df['close'].shift(-6) > df['close'] * 1.02).astype(int)
df = df.dropna()

X = df[features]
y = df['target']

# 注意：时间序列数据不要随机打乱，保持顺序
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. 训练模型
model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# 6. 保存模型
joblib.dump(model, 'eth_ai_model.pkl')
print(f"模型已保存为 eth_ai_model.pkl")
print(f"测试集准确率: {model.score(X_test, y_test):.3f}")
