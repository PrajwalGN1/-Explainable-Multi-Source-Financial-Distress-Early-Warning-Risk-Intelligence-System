import numpy as np
import xgboost as xgb

X = np.random.randn(20, 5)
y = np.random.randint(0, 2, 20)
clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
clf.fit(X, y)

# TreeSHAP via Booster
booster = clf.get_booster()
dmat = xgb.DMatrix(X)
shap_values = booster.predict(dmat, pred_contribs=True)
print("TreeSHAP calculation successful! Shape:", shap_values.shape)
print("Base value (bias):", shap_values[0, -1])
print("Sample feature contributions:", shap_values[0, :-1])
