import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("pipeline.py").read().split('# 3. Base learners')[0])

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

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

configs = {
    "Raw+LogReg": Pipeline([("features", HybridFeaturePipeline(False, False)),
                             ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
    "Raw+RF": Pipeline([("features", HybridFeaturePipeline(False, False)),
                         ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE))]),
    "Raw+XGB": Pipeline([("features", HybridFeaturePipeline(False, False)),
                          ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                 eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0))]),
    "Hybrid+XGB": Pipeline([("features", HybridFeaturePipeline(True, True)),
                             ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                    eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0))]),
    "Hybrid+Voting": Pipeline([("features", HybridFeaturePipeline(True, True)),
                                ("clf", VotingClassifier(estimators=make_base_learners(), voting="soft", n_jobs=-1))]),
    "Hybrid+Stacking": Pipeline([("features", HybridFeaturePipeline(True, True)),
                                  ("clf", StackingClassifier(estimators=make_base_learners(),
                                          final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                          cv=5, stack_method="predict_proba", n_jobs=-1))]),
}

per_fold = {}
for name, pipe in configs.items():
    res = cross_validate(pipe, X_raw, y, cv=cv, scoring=SCORING, n_jobs=-1)
    per_fold[name] = {m: res[f"test_{m}"] for m in SCORING}
    print(f"{name:18s} F1 folds: {np.round(res['test_f1'],4).tolist()}")

np.savez("per_fold_scores.npz", **{
    f"{name}__{m}": vals for name, d in per_fold.items() for m, vals in d.items()
})
print("SAVED per_fold_scores.npz")
