import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("pipeline.py").read().split('# 3. Base learners')[0])

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def xgb():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)

ablation_configs = {
    "Raw only (baseline)":                 dict(include_domain=False, include_missingness=False, include_clusterpca=False),
    "+ Domain features only":              dict(include_domain=True,  include_missingness=False, include_clusterpca=False),
    "+ Missingness indicators only":       dict(include_domain=False, include_missingness=True,  include_clusterpca=False),
    "+ Cluster/PCA meta-features only":    dict(include_domain=False, include_missingness=False, include_clusterpca=True),
    "+ Domain + Missingness":              dict(include_domain=True,  include_missingness=True,  include_clusterpca=False),
    "+ All combined (proposed hybrid)":    dict(include_domain=True,  include_missingness=True,  include_clusterpca=True),
}

ablation_results = []
ablation_fold_f1 = {}
for name, kwargs in ablation_configs.items():
    pipe = Pipeline([("features", AblationFeaturePipeline(**kwargs)), ("clf", xgb())])
    res = cross_validate(pipe, X_raw, y, cv=cv, scoring=SCORING, n_jobs=-1)
    ablation_fold_f1[name] = res["test_f1"]
    row = {"config": name}
    for m in SCORING:
        row[m + "_mean"] = round(res[f"test_{m}"].mean(), 4)
        row[m + "_std"] = round(res[f"test_{m}"].std(), 4)
    ablation_results.append(row)
    print(f"{name:38s} | Acc {row['accuracy_mean']:.4f} | F1 {row['f1_mean']:.4f} | AUC {row['roc_auc_mean']:.4f}")

pd.DataFrame(ablation_results).to_csv("ablation_results.csv", index=False)
np.savez("ablation_fold_f1.npz", **{k.replace(" ", "_").replace("/","_").replace("(","").replace(")","").replace("+",""): v
                                      for k, v in ablation_fold_f1.items()})
print("\nABLATION_DONE")
