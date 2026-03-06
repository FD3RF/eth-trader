import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)

rows = 2000

data = pd.DataFrame({
    "price_change": np.random.randn(rows),
    "volume_ratio": np.random.rand(rows)*2,
    "trend_strength": np.random.rand(rows)*100
})

data["target"] = (data["price_change"] > 0).astype(int)

X = data[["price_change","volume_ratio","trend_strength"]]
y = data["target"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X,y)

joblib.dump(model,"eth_ai_model.pkl")

print("AI模型训练完成")
