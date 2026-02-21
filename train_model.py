import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import GradientBoostingClassifier
import joblib

SYMBOL = "ETH/USDT:USDT"
TIMEFRAME = "5m"
LIMIT = 5000
FEATURES = ["rsi", "ma20", "ma60", "macd", "macd_signal", "atr", "adx"]

# 初始化 OKX 交易所
exchange = ccxt.okx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# 获取 K 线数据
ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])

# 计算技术指标
df['rsi'] = ta.rsi(df['c'], length=14)
df['ma20'] = ta.sma(df['c'], length=20)
df['ma60'] = ta.sma(df['c'], length=60)
macd = ta.macd(df['c'])
df['macd'] = macd['MACD_12_26_9']
df['macd_signal'] = macd['MACDs_12_26_9']          # 修正：使用信号线
df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)
df['adx'] = ta.adx(df['h'], df['l'], df['c'], length=14)['ADX_14']  # 修正：提取 ADX 列

# 标签：未来3根K线收盘价是否比当前高 0.2%以上（过滤小波动）
df['target'] = (df['c'].shift(-3) > df['c'] * 1.002).astype(int)

# 删除 NaN 行
df.dropna(inplace=True)
print(f"有效数据量：{len(df)}")

# 准备特征和标签
X = df[FEATURES]
y = df['target']

# 训练模型
model = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)
model.fit(X, y)

# 保存模型
joblib.dump(model, "eth_ai_model.pkl")
print("✅ 模型已保存为 eth_ai_model.pkl")
