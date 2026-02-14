# train_model.py
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# 获取历史数据（过去3年）
exchange = ccxt.binance()
symbol = 'ETH/USDT'
timeframe = '1h'
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=20000)  # 约3年
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 添加技术指标
import ta
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['ma20'] = df['close'].rolling(20).mean()
df['ma60'] = df['close'].rolling(60).mean()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
df['atr_pct'] = df['atr'] / df['close'] * 100
adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
df['adx'] = adx.adx()

# 特征
features = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
df = df.dropna()

# 标签：未来6小时涨幅 > 2% 为1
df['target'] = (df['close'].shift(-6) > df['close'] * 1.02).astype(int)
df = df.dropna()

X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

joblib.dump(model, 'eth_ai_model.pkl')
print(f"模型保存，测试准确率: {model.score(X_test, y_test):.2f}")