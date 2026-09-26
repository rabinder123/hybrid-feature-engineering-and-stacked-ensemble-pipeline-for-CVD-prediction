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

# refit on full data once to report the selected hyperparameters (informational only)
search_xgb.fit(X_raw, y)
print(" best params (fit on full data, for reference only):", search_xgb.best_params_)

# --- 2. Scoped nested tuning: Stacking meta-learner regularization only ---
# Full grid search over all 6 base learners' hyperparameters inside nested CV is
# computationally prohibitive (6 learners x grids x 3 inner x 5 outer folds). We scope
# the honest search to the piece unique to the "hybrid" claim: how hard the meta-learner
# regularizes over the base learners' out-of-fold predictions. Base learners keep the
# same settings used throughout.
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

pipe_stack = Pipeline([
    ("features", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True)),
    ("clf", StackingClassifier(estimators=make_base_learners(),
                                final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                cv=5, stack_method="predict_proba", n_jobs=-1)),
])
grid_stack = {"clf__final_estimator__C": [0.1, 1.0, 10.0]}
search_stack = GridSearchCV(pipe_stack, grid_stack, scoring="f1", cv=inner_cv, n_jobs=-1)
nested_scores_stack = cross_val_score(search_stack, X_raw, y, cv=outer_cv, scoring="f1", n_jobs=1)
print("\nNested CV F1 - Hybrid Stacking (meta-learner C tuned each outer fold):")
print(" folds:", np.round(nested_scores_stack, 4).tolist())
print(f" mean={nested_scores_stack.mean():.4f} std={nested_scores_stack.std():.4f}")

from scipy import stats
t_stat, p_val = stats.ttest_rel(nested_scores_stack, nested_scores_xgb)
print(f"\nNote: nested scores use 5 outer folds (vs 10 in the main comparison), so not directly)")
print(f"comparable to the earlier table row-for-row. Paired t-test on these 5 folds: t={t_stat:.3f}, p={p_val:.4f}")
