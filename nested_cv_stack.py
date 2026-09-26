import warnings; warnings.filterwarnings("ignore")
import time
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
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

def make_base_learners():
    return [
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=8, random_state=RANDOM_STATE)),
        ("et", ExtraTreesClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0)),
        ("svm", SVC(kernel="rbf", C=1.5, probability=True, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ]

# Scoped down from the main 10-fold/6-learner setup for tractability: 3 outer / 2 inner
# folds, internal stacking cv=3 instead of 5, smaller forests. This is a reduced-fidelity
# diagnostic to check whether the meta-learner's regularization strength matters -
# not a re-run of the headline numbers.
pipe_stack = Pipeline([
    ("features", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True)),
    ("clf", StackingClassifier(estimators=make_base_learners(),
                                final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                cv=3, stack_method="predict_proba", n_jobs=-1)),
])
grid_stack = {"clf__final_estimator__C": [0.1, 1.0, 10.0]}

t0 = time.time()
search_stack = GridSearchCV(pipe_stack, grid_stack, scoring="f1", cv=inner_cv, n_jobs=-1)
nested_scores_stack = cross_val_score(search_stack, X_raw, y, cv=outer_cv, scoring="f1", n_jobs=1)
print(f"Elapsed: {time.time()-t0:.1f}s")
print("Nested CV F1 - Hybrid Stacking (meta-learner C tuned each outer fold, reduced-fidelity):")
print(" folds:", np.round(nested_scores_stack, 4).tolist())
print(f" mean={nested_scores_stack.mean():.4f} std={nested_scores_stack.std():.4f}")

search_stack.fit(X_raw, y)
print(" best C (fit on full data, for reference only):", search_stack.best_params_)
print("STACK_NESTED_DONE")
