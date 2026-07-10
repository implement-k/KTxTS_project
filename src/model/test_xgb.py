import numpy as np
from xgboost import XGBClassifier, XGBRegressor
X = np.random.rand(1274641, 33)
y = (np.random.rand(1274641) > 0.5).astype(np.int64)
c = XGBClassifier(n_estimators=10, max_depth=6, random_state=42)
print("fitting classifier")
c.fit(X, y)
print("classifier fit done")
