import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("pipeline.py").read().split('# 3. Base learners')[0])

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# --- 1. Full nested tuning: Hybrid features + single XGBoost ---
pipe_xgb = Pipeline([
    ("features", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True)),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)),
])
grid_xgb = {
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [3, 4, 5],
    "clf__learning_rate": [0.03, 0.05, 0.1],
}
search_xgb = GridSearchCV(pipe_xgb, grid_xgb, scoring="f1", cv=inner_cv, n_jobs=-1)
nested_scores_xgb = cross_val_score(search_xgb, X_raw, y, cv=outer_cv, scoring="f1", n_jobs=1)
print("Nested CV F1 - Hybrid XGBoost (full grid search each outer fold):")
print(" folds:", np.round(nested_scores_xgb, 4).tolist())
print(f" mean={nested_scores_xgb.mean():.4f} std={nested_scores_xgb.std():.4f}")

print("XGB_NESTED_DONE")
