import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("pipeline.py").read().split('# 3. Base learners')[0])

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42

# ============================================================
# PART A: Real SHAP analysis on the hybrid XGBoost model
# ============================================================
fp = HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True)
fp.fit(X_raw, y)
Xt = fp.transform(X_raw)
feat_names = (list(fp.num_cols_) + BINARY_COLS
              + list(fp.ohe_.get_feature_names_out(NOMINAL_COLS))
              + ['cluster_id', 'cluster_dist', 'pca_1', 'pca_2', 'pca_3'])

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                       eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
model.fit(Xt, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xt)  # (n_samples, n_features)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feat_names).sort_values(ascending=False)
print("=== Top 15 features by mean |SHAP value| ===")
print(shap_importance.head(15))
shap_importance.head(15).to_csv("shap_importance.csv")

# Direction of effect: mean signed SHAP for top features (positive = pushes toward disease)
mean_signed_shap = pd.Series(shap_values.mean(axis=0), index=feat_names)
print("\n=== Mean signed SHAP (top 15 by |value|) ===")
print(mean_signed_shap.loc[shap_importance.head(15).index])

# Correlation between SHAP-based ranking and XGBoost's built-in gain-based importance
gain_importance = pd.Series(model.feature_importances_, index=feat_names)
rank_corr = shap_importance.rank().corr(gain_importance.rank(), method='spearman')
print(f"\nSpearman rank correlation, SHAP importance vs XGBoost gain importance: {rank_corr:.3f}")

np.save("shap_values.npy", shap_values)
print("\nSHAP_DONE")

# ============================================================
# PART B: Real RFE comparison — does trimming the hybrid feature
# set with Recursive Feature Elimination help or hurt, under the
# SAME 10-fold protocol used for the headline results?
# ============================================================
print("\n" + "="*60)

class HybridFeaturePipelineRFE(HybridFeaturePipeline):
    """Same as HybridFeaturePipeline, but after building the full hybrid
    matrix, applies RFE (fit on the training fold only) to select a
    subset of columns before returning them."""
    def __init__(self, use_engineered=True, use_cluster_pca=True, n_clusters=4, n_pca=3,
                 n_features_to_select=15, rfe_estimator=None):
        super().__init__(use_engineered, use_cluster_pca, n_clusters, n_pca)
        self.n_features_to_select = n_features_to_select
        self.rfe_estimator = rfe_estimator

    def fit(self, X, y=None):
        super().fit(X, y)
        Xt_full = super().transform(X)
        est = self.rfe_estimator if self.rfe_estimator is not None else \
              RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
        self.rfe_ = RFE(est, n_features_to_select=self.n_features_to_select)
        self.rfe_.fit(Xt_full, y)
        return self

    def transform(self, X):
        Xt_full = super().transform(X)
        return self.rfe_.transform(Xt_full)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

configs_rfe = {
    "Hybrid (all ~30 features) + XGBoost":        (HybridFeaturePipeline(True, True), None),
    "Hybrid + RFE-15 + XGBoost":                  (HybridFeaturePipelineRFE(n_features_to_select=15), None),
    "Hybrid + RFE-10 + XGBoost":                  (HybridFeaturePipelineRFE(n_features_to_select=10), None),
}

rfe_results = []
selected_features_by_config = {}
for name, (feat_pipe, _) in configs_rfe.items():
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
    pipe = Pipeline([("features", feat_pipe), ("clf", clf)])
    res = cross_validate(pipe, X_raw, y, cv=cv, scoring=SCORING, n_jobs=-1)
    row = {"model": name}
    for m in SCORING:
        row[m + "_mean"] = round(res[f"test_{m}"].mean(), 4)
        row[m + "_std"] = round(res[f"test_{m}"].std(), 4)
    rfe_results.append(row)
    print(f"{name:40s} | Acc {row['accuracy_mean']:.4f} | F1 {row['f1_mean']:.4f} | AUC {row['roc_auc_mean']:.4f}")

# Also report which features RFE-15 kept, fit on the FULL dataset (informational only,
# not used for the CV numbers above) so we can describe them in the paper.
full_rfe = HybridFeaturePipelineRFE(n_features_to_select=15)
full_rfe.fit(X_raw, y)
kept_mask = full_rfe.rfe_.support_
kept_features = [f for f, k in zip(feat_names, kept_mask) if k]
print("\nFeatures kept by RFE-15 (fit on full data, for description only):")
print(kept_features)

pd.DataFrame(rfe_results).to_csv("rfe_results.csv", index=False)
print("\nRFE_DONE")
