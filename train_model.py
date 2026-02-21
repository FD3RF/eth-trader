import ccxt
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# =============================
# 配置参数
# =============================
SYMBOL = 'ETH/USDT:USDT'          # Bybit 永续合约
TIMEFRAME = '1h'                   # 1小时K线
LIMIT = 20000                      # 获取20000根K线（约2.28年）
TEST_SIZE = 0.2                    # 20% 作为测试集
RANDOM_STATE = 42

# =============================
# 初始化交易所（Bybit 永续合约）
# =============================
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'linear'}
})

print(f"正在从 Bybit 下载 {SYMBOL} 的 {TIMEFRAME} 数据，数量：{LIMIT}，请稍等...")
try:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
except Exception as e:
    print(f"数据下载失败：{e}")
    exit(1)

# 转换为 DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
print(f"数据下载完成，共 {len(df)} 条记录")

# =============================
# 计算技术指标（使用 ta 库）
# =============================
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['ma20'] = ta.trend.sma_indicator(df['close'], window=20)
df['ma60'] = ta.trend.sma_indicator(df['close'], window=60)
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
df['atr_pct'] = df['atr'] / df['close'] * 100
adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
df['adx'] = adx.adx()

# =============================
# 生成标签：未来6小时涨幅是否 > 2%
# =============================
df['target'] = (df['close'].shift(-6) > df['close'] * 1.02).astype(int)

# 删除 NaN 值（指标计算和 shift 会产生 NaN）
df = df.dropna()

print(f"处理后数据量：{len(df)} 条")
print(f"正样本比例：{df['target'].mean():.2%}")

# =============================
# 准备特征和标签
# =============================
features = ['rsi', 'ma20', 'ma60', 'macd', 'macd_signal', 'atr_pct', 'adx']
X = df[features]
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"训练集大小：{len(X_train)}，测试集大小：{len(X_test)}")

# =============================
# 训练模型
# =============================
model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

print("开始训练模型...")
model.fit(X_train, y_train)

# =============================
# 评估模型（可选）
# =============================
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"训练集准确率：{train_score:.4f}")
print(f"测试集准确率：{test_score:.4f}")

# =============================
# 保存模型
# =============================
MODEL_FILE = 'eth_ai_model.pkl'
joblib.dump(model, MODEL_FILE)
print(f"模型已保存为 {MODEL_FILE}")
