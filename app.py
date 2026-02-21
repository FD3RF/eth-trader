import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# =============================
# 配置参数
# =============================
SYMBOL = 'ETH/USDT'                # OKX 永续合约，与 app.py 一致
TIMEFRAME = '1h'                   # 1小时K线（可根据需要调整）
LIMIT = 20000                      # 获取 20000 根 K 线
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================
# 初始化交易所（OKX 永续合约）
# =============================
exchange = ccxt.okx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

print(f"正在从 OKX 下载 {SYMBOL} 的 {TIMEFRAME} 数据，数量：{LIMIT}，请稍等...")
try:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
except Exception as e:
    print(f"数据下载失败：{e}")
    exit(1)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
print(f"数据下载完成，共 {len(df)} 条记录")

# =============================
# 计算技术指标（与 app.py 严格一致）
# =============================
df['rsi'] = ta.rsi(df['close'], length=14)
df['ma20'] = ta.sma(df['close'], length=20)
df['ma60'] = ta.sma(df['close'], length=60)

macd = ta.macd(df['close'])
df['macd'] = macd['MACD_12_26_9']
df['macd_signal'] = macd['MACDs_12_26_9']

df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['atr_pct'] = df['atr'] / df['close'] * 100   # 百分比形式
df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']

# =============================
# 生成标签：未来6小时涨幅是否 > 2%
# =============================
df['target'] = (df['close'].shift(-6) > df['close'] * 1.02).astype(int)

# 删除 NaN 值
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
# 评估模型
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
print(f"✅ 模型已保存为 {MODEL_FILE}")
print(f"模型使用的特征：{model.feature_names_in_.tolist()}")
