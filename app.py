# -*- coding: utf-8 -*-
"""
ETH 100x AI 模型训练脚本（优化版）
功能：
- 从 OKX 获取历史K线数据
- 计算多种技术指标
- 定义合理标签（未来N根K线涨幅）
- 使用时间序列交叉验证
- 超参数网格搜索优化
- 保存最佳模型及特征列表
"""

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================
# 配置参数
# =============================
SYMBOL = "ETH/USDT-SWAP"          # OKX 永续合约
TIMEFRAME = "5m"                  # K线周期
LIMIT = 10000                      # 获取的 K 线数量（建议至少10000，约1个月数据）
TEST_SIZE = 0.2                    # 测试集比例（用于最终评估）
RANDOM_STATE = 42
MODEL_FILE = "eth_ai_model.pkl"
FEATURE_FILE = "eth_ai_features.pkl"

# 标签定义：预测未来 N 根 K 线涨幅是否超过 THRESHOLD
N_FUTURE = 6                       # 未来6根K线
THRESHOLD = 0.01                    # 涨幅阈值 1%

# 特征列表
FEATURES = [
    'rsi', 'ma20', 'ma60', 'macd', 'macd_signal',
    'atr_pct', 'adx', 'bb_upper', 'bb_lower', 'bb_width',
    'volume_ratio', 'ema20', 'ema50'
]

# =============================
# 初始化交易所（OKX）
# =============================
print("初始化 OKX 交易所...")
exchange = ccxt.okx({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})
# 如果使用代理，取消注释
# exchange.proxies = {'http': 'http://your-proxy', 'https': 'https://your-proxy'}

# =============================
# 获取历史数据
# =============================
print(f"获取 {SYMBOL} 最近 {LIMIT} 根 {TIMEFRAME} K线...")
ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
print(f"成功获取 {len(df)} 条数据")

# =============================
# 计算技术指标
# =============================
print("计算技术指标...")

# 基础指标
df['rsi'] = ta.rsi(df['close'], length=14)
df['ma20'] = ta.sma(df['close'], length=20)
df['ma60'] = ta.sma(df['close'], length=60)
df['ema20'] = ta.ema(df['close'], length=20)
df['ema50'] = ta.ema(df['close'], length=50)

# MACD
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
df['macd'] = macd['MACD_12_26_9']
df['macd_signal'] = macd['MACDs_12_26_9']
df['macd_hist'] = macd['MACDh_12_26_9']

# ATR
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['atr_pct'] = df['atr'] / df['close'] * 100

# ADX
adx = ta.adx(df['high'], df['low'], df['close'], length=14)
df['adx'] = adx['ADX_14']

# 布林带
bb = ta.bbands(df['close'], length=20, std=2)
df['bb_upper'] = bb['BBU_20_2.0']
df['bb_lower'] = bb['BBL_20_2.0']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

# 成交量比率
df['volume_ma20'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma20']

# =============================
# 生成标签：未来 N 根 K 线涨幅是否超过阈值
# =============================
df['future_return'] = (df['close'].shift(-N_FUTURE) - df['close']) / df['close']
df['target'] = (df['future_return'] > THRESHOLD).astype(int)

# 删除 NaN 行
df.dropna(inplace=True)
print(f"清洗后数据量：{len(df)} 条")
print(f"正样本比例: {df['target'].mean():.2%}")

# =============================
# 准备特征和标签
# =============================
X = df[FEATURES]
y = df['target']
print(f"特征矩阵形状: {X.shape}")

# =============================
# 时间序列交叉验证
# =============================
tscv = TimeSeriesSplit(n_splits=5)
print("使用时间序列交叉验证 (5折)")

# =============================
# 超参数网格搜索
# =============================
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("开始网格搜索...")
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证 AUC: {grid_search.best_score_:.4f}")

# =============================
# 在最终测试集上评估（按时间顺序的最后20%）
# =============================
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 用最佳参数重新训练（或者直接用grid_search.best_estimator_已经在整个数据集上训练过）
# 这里重新训练以便准确评估测试集
final_model = GradientBoostingClassifier(**grid_search.best_params_, random_state=RANDOM_STATE)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

print("\n========== 测试集评估 ==========")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=['下跌', '上涨']))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# =============================
# 特征重要性
# =============================
importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n特征重要性:")
print(importance)

# =============================
# 保存模型和特征列表
# =============================
joblib.dump(final_model, MODEL_FILE)
joblib.dump(FEATURES, FEATURE_FILE)
print(f"模型已保存至 {MODEL_FILE}")
print(f"特征列表已保存至 {FEATURE_FILE}")

# 可选：保存预处理参数（如均值、标准差）以便在预测时使用
# 但本模型不需要标准化，因为树模型对尺度不敏感
print("训练完成！")
