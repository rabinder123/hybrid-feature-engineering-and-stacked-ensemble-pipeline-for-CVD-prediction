import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("pipeline.py").read().split('# 3. Base learners')[0])

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

def make_base_learners():
    return [
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)),
        ("et", ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0)),
        ("svm", SVC(kernel="rbf", C=1.5, probability=True, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ]

models = {
    "Baseline XGBoost (raw)": Pipeline([
        ("features", HybridFeaturePipeline(use_engineered=False, use_cluster_pca=False)),
        ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0)),
    ]),
    "Hybrid XGBoost (single)": Pipeline([
        ("features", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True)),
        ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0)),
    ]),
    "Hybrid Stacking (proposed)": Pipeline([
        ("features", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True)),
        ("clf", StackingClassifier(estimators=make_base_learners(),
                                    final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                    cv=5, stack_method="predict_proba", n_jobs=-1)),
    ]),
}

oof_probs = {}
for name, pipe in models.items():
    probs = cross_val_predict(pipe, X_raw, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_probs[name] = probs
    print(f"Done: {name}")

np.savez("oof_probs.npz", **{k.replace(" ", "_"): v for k, v in oof_probs.items()}, y=y,
         age=df["age"].values, sex=df["sex"].values)
print("Saved oof_probs.npz")
