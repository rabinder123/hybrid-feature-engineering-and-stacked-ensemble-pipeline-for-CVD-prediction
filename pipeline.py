import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from scipy import stats

RANDOM_STATE = 42

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
df.columns = [c.strip() for c in df.columns]

# The merged Statlog+Cleveland+Hungary dataset encodes missing cholesterol / resting bp
# as literal 0, not NaN. Left untreated, this silently corrupts every model that touches
# these columns (0 mg/dl cholesterol is not physiologically real). We surface it explicitly.
df["cholesterol"] = df["cholesterol"].replace(0, np.nan)
df["resting bp s"] = df["resting bp s"].replace(0, np.nan)

y = df["target"].values
X_raw = df.drop(columns=["target"])

print("Rows:", len(df), "| Positive rate:", round(y.mean(), 3))
print("Missing cholesterol:", X_raw["cholesterol"].isna().sum(),
      "| Missing resting bp s:", X_raw["resting bp s"].isna().sum())

# -----------------------------
# 2. Custom hybrid feature engineering transformer
#    (domain-clinical features; all deterministic / fixed-threshold -> no CV leakage)
# -----------------------------
class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-driven hybrid features layered on top of the raw clinical variables."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Missingness indicators BEFORE imputation happens downstream (source-of-cohort signal:
        # Hungarian/Statlog sub-cohorts systematically omit cholesterol far more than Cleveland).
        X["chol_was_missing"] = X["cholesterol"].isna().astype(int)
        X["bp_was_missing"] = X["resting bp s"].isna().astype(int)

        age = X["age"]
        hr = X["max heart rate"]
        bp = X["resting bp s"]
        chol = X["cholesterol"]

        # Age-predicted max heart rate reserve / achievement ratio
        age_pred_max_hr = 220 - age
        X["hr_reserve"] = age_pred_max_hr - hr
        X["hr_achievement_ratio"] = hr / age_pred_max_hr.replace(0, np.nan)

        # AHA-style resting BP category (ordinal, fixed clinical thresholds)
        X["bp_category"] = pd.cut(
            bp, bins=[-1, 120, 129, 139, 300],
            labels=[0, 1, 2, 3]
        ).astype(float)

        # Cholesterol risk category (NCEP ATP III thresholds), NaN kept as its own state
        X["chol_category"] = pd.cut(
            chol, bins=[-1, 200, 239, 1000],
            labels=[0, 1, 2]
        ).astype(float)

        # Ischemia-severity interactions
        X["oldpeak_x_slope"] = X["oldpeak"] * X["ST slope"]
        X["angina_x_oldpeak"] = X["exercise angina"] * X["oldpeak"]
        X["oldpeak_x_hrreserve"] = X["oldpeak"] * X["hr_reserve"]

        # Ratio / normalized features
        X["chol_to_age"] = chol / age
        X["bp_to_age"] = bp / age

        # Composite categorical risk-factor count (cheap, interpretable, fixed rule)
        X["risk_factor_count"] = (
            (X["sex"] == 1).astype(int)
            + (X["fasting blood sugar"] == 1).astype(int)
            + (X["exercise angina"] == 1).astype(int)
            + (X["oldpeak"] > 1.0).astype(int)
            + (X["resting ecg"] != 0).astype(int)
        )

        return X


class ClinicalFeatureEngineerAblation(BaseEstimator, TransformerMixin):
    """Like ClinicalFeatureEngineer, but domain-derived features and
    missingness-indicator features can be toggled independently, for
    ablation purposes."""
    def __init__(self, include_domain=True, include_missingness=True):
        self.include_domain = include_domain
        self.include_missingness = include_missingness

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.include_missingness:
            X["chol_was_missing"] = X["cholesterol"].isna().astype(int)
            X["bp_was_missing"] = X["resting bp s"].isna().astype(int)
        if self.include_domain:
            age = X["age"]; hr = X["max heart rate"]; bp = X["resting bp s"]; chol = X["cholesterol"]
            age_pred_max_hr = 220 - age
            X["hr_reserve"] = age_pred_max_hr - hr
            X["hr_achievement_ratio"] = hr / age_pred_max_hr.replace(0, np.nan)
            X["bp_category"] = pd.cut(bp, bins=[-1, 120, 129, 139, 300], labels=[0, 1, 2, 3]).astype(float)
            X["chol_category"] = pd.cut(chol, bins=[-1, 200, 239, 1000], labels=[0, 1, 2]).astype(float)
            X["oldpeak_x_slope"] = X["oldpeak"] * X["ST slope"]
            X["angina_x_oldpeak"] = X["exercise angina"] * X["oldpeak"]
            X["oldpeak_x_hrreserve"] = X["oldpeak"] * X["hr_reserve"]
            X["chol_to_age"] = chol / age
            X["bp_to_age"] = bp / age
            X["risk_factor_count"] = (
                (X["sex"] == 1).astype(int) + (X["fasting blood sugar"] == 1).astype(int)
                + (X["exercise angina"] == 1).astype(int) + (X["oldpeak"] > 1.0).astype(int)
                + (X["resting ecg"] != 0).astype(int)
            )
        return X


class ClusterPCAAugmenter(BaseEstimator, TransformerMixin):
    """Unsupervised meta-features: KMeans cluster id/distances + PCA components,
    fit strictly on the training fold to avoid leakage."""
    def __init__(self, n_clusters=4, n_pca=3, cont_cols=None):
        self.n_clusters = n_clusters
        self.n_pca = n_pca
        self.cont_cols = cont_cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.cont_cols) if not isinstance(X, pd.DataFrame) else X
        self.scaler_ = StandardScaler().fit(X[self.cont_cols])
        Xs = self.scaler_.transform(X[self.cont_cols])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=RANDOM_STATE, n_init=10).fit(Xs)
        self.pca_ = PCA(n_components=self.n_pca, random_state=RANDOM_STATE).fit(Xs)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cont_cols) if not isinstance(X, pd.DataFrame) else X
        Xs = self.scaler_.transform(X[self.cont_cols])
        cluster_id = self.kmeans_.predict(Xs).reshape(-1, 1)
        cluster_dist = self.kmeans_.transform(Xs).min(axis=1).reshape(-1, 1)
        pcs = self.pca_.transform(Xs)
        out = np.hstack([cluster_id, cluster_dist, pcs])
        return out


NOMINAL_COLS = ["chest pain type", "resting ecg", "ST slope"]
BASE_NUMERIC = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
BINARY_COLS = ["sex", "fasting blood sugar", "exercise angina"]
ENGINEERED_NUMERIC = [
    "hr_reserve", "hr_achievement_ratio", "bp_category", "chol_category",
    "oldpeak_x_slope", "angina_x_oldpeak", "oldpeak_x_hrreserve",
    "chol_to_age", "bp_to_age", "risk_factor_count",
    "chol_was_missing", "bp_was_missing",
]
CLUSTER_SRC_COLS = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak", "hr_reserve"]

print("\nEngineered feature columns added:", len(ENGINEERED_NUMERIC) + 5, "(incl. 4 cluster/PCA meta-features)")


class HybridFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end leakage-safe preprocessor:
      raw df -> clinical feature engineering -> median imputation -> scaling
             -> one-hot nominal encoding -> KMeans/PCA meta-features -> single matrix
    Every stateful step (imputer, scaler, encoder, KMeans, PCA) is fit ONLY on the
    training fold it's called with, then reused for transform on held-out data.
    """
    def __init__(self, use_engineered=True, use_cluster_pca=True, n_clusters=4, n_pca=3):
        self.use_engineered = use_engineered
        self.use_cluster_pca = use_cluster_pca
        self.n_clusters = n_clusters
        self.n_pca = n_pca

    def fit(self, X, y=None):
        Xe = self._engineer(X)
        num_cols = BASE_NUMERIC + (ENGINEERED_NUMERIC if self.use_engineered else [])
        self.num_cols_ = num_cols
        self.imputer_ = SimpleImputer(strategy="median").fit(Xe[num_cols])
        Xe_imp = Xe.copy()
        Xe_imp[num_cols] = self.imputer_.transform(Xe[num_cols])
        self.scaler_ = StandardScaler().fit(Xe_imp[num_cols])
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xe_imp[NOMINAL_COLS])
        if self.use_cluster_pca:
            self.cluster_aug_ = ClusterPCAAugmenter(
                n_clusters=self.n_clusters, n_pca=self.n_pca, cont_cols=CLUSTER_SRC_COLS
            ).fit(Xe_imp)
        return self

    def transform(self, X):
        Xe = self._engineer(X)
        Xe[self.num_cols_] = self.imputer_.transform(Xe[self.num_cols_])
        num_scaled = self.scaler_.transform(Xe[self.num_cols_])
        onehot = self.ohe_.transform(Xe[NOMINAL_COLS])
        binary = Xe[BINARY_COLS].values
        parts = [num_scaled, binary, onehot]
        if self.use_cluster_pca:
            parts.append(self.cluster_aug_.transform(Xe))
        return np.hstack(parts)

    def _engineer(self, X):
        X = X.copy()
        if self.use_engineered:
            X = ClinicalFeatureEngineer().transform(X)
        else:
            # still need chol/bp missing flags absent, but base cols must exist unmodified
            pass
        return X


DOMAIN_NUMERIC = ["hr_reserve", "hr_achievement_ratio", "bp_category", "chol_category",
                   "oldpeak_x_slope", "angina_x_oldpeak", "oldpeak_x_hrreserve",
                   "chol_to_age", "bp_to_age", "risk_factor_count"]
MISSING_NUMERIC = ["chol_was_missing", "bp_was_missing"]
CLUSTER_SRC_COLS_NODOMAIN = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]


class AblationFeaturePipeline(BaseEstimator, TransformerMixin):
    """Same leakage-safe design as HybridFeaturePipeline, but the three feature
    families (domain, missingness, cluster/PCA) can be toggled independently
    for ablation."""
    def __init__(self, include_domain=False, include_missingness=False,
                 include_clusterpca=False, n_clusters=4, n_pca=3):
        self.include_domain = include_domain
        self.include_missingness = include_missingness
        self.include_clusterpca = include_clusterpca
        self.n_clusters = n_clusters
        self.n_pca = n_pca

    def fit(self, X, y=None):
        Xe = self._engineer(X)
        num_cols = BASE_NUMERIC.copy()
        if self.include_domain:
            num_cols += DOMAIN_NUMERIC
        if self.include_missingness:
            num_cols += MISSING_NUMERIC
        self.num_cols_ = num_cols
        self.imputer_ = SimpleImputer(strategy="median").fit(Xe[num_cols])
        Xe_imp = Xe.copy()
        Xe_imp[num_cols] = self.imputer_.transform(Xe[num_cols])
        self.scaler_ = StandardScaler().fit(Xe_imp[num_cols])
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xe_imp[NOMINAL_COLS])
        if self.include_clusterpca:
            src = CLUSTER_SRC_COLS if self.include_domain else CLUSTER_SRC_COLS_NODOMAIN
            self.cluster_src_ = src
            self.cluster_aug_ = ClusterPCAAugmenter(
                n_clusters=self.n_clusters, n_pca=self.n_pca, cont_cols=src
            ).fit(Xe_imp)
        return self

    def transform(self, X):
        Xe = self._engineer(X)
        Xe[self.num_cols_] = self.imputer_.transform(Xe[self.num_cols_])
        num_scaled = self.scaler_.transform(Xe[self.num_cols_])
        onehot = self.ohe_.transform(Xe[NOMINAL_COLS])
        binary = Xe[BINARY_COLS].values
        parts = [num_scaled, binary, onehot]
        if self.include_clusterpca:
            parts.append(self.cluster_aug_.transform(Xe))
        return np.hstack(parts)

    def _engineer(self, X):
        X = X.copy()
        if self.include_domain or self.include_missingness:
            X = ClinicalFeatureEngineerAblation(
                include_domain=self.include_domain,
                include_missingness=self.include_missingness
            ).transform(X)
        return X


# -----------------------------
# 3. Base learners + stacking ensemble ("novel hybrid" = heterogeneous learner
#    families + meta-learner trained on out-of-fold predictions)
# -----------------------------
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


def make_stacking_model():
    base = make_base_learners()
    meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=base, final_estimator=meta, cv=5, stack_method="predict_proba", n_jobs=-1
    )


def make_voting_model():
    base = make_base_learners()
    return VotingClassifier(estimators=base, voting="soft", n_jobs=-1)


# -----------------------------
# 4. Cross-validated comparison: baseline vs hybrid-feature single models vs hybrid ensemble
# -----------------------------
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

def evaluate(name, feature_pipeline, model):
    pipe = Pipeline([("features", feature_pipeline), ("clf", model)])
    res = cross_validate(pipe, X_raw, y, cv=cv, scoring=SCORING, n_jobs=-1, return_train_score=False)
    row = {"model": name}
    for m in SCORING:
        row[m] = res[f"test_{m}"]
    return row

results = []
fold_f1 = {}

configs = [
    ("Baseline: raw features + LogReg", HybridFeaturePipeline(use_engineered=False, use_cluster_pca=False),
     LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ("Baseline: raw features + RandomForest", HybridFeaturePipeline(use_engineered=False, use_cluster_pca=False),
     RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)),
    ("Baseline: raw features + XGBoost", HybridFeaturePipeline(use_engineered=False, use_cluster_pca=False),
     XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                    random_state=RANDOM_STATE, verbosity=0)),
    ("Hybrid features + XGBoost (single)", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True),
     XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                    random_state=RANDOM_STATE, verbosity=0)),
    ("Hybrid features + Soft Voting Ensemble", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True),
     make_voting_model()),
    ("Hybrid features + Stacking Ensemble (proposed)", HybridFeaturePipeline(use_engineered=True, use_cluster_pca=True),
     make_stacking_model()),
]

for name, fp, model in configs:
    row = evaluate(name, fp, model)
    fold_f1[name] = row["f1"]
    summary = {"model": name}
    for m in SCORING:
        summary[m + "_mean"] = round(row[m].mean(), 4)
        summary[m + "_std"] = round(row[m].std(), 4)
    results.append(summary)
    print(f"{name:55s} | Acc {summary['accuracy_mean']:.4f} | Prec {summary['precision_mean']:.4f} | "
          f"Rec {summary['recall_mean']:.4f} | F1 {summary['f1_mean']:.4f} | AUC {summary['roc_auc_mean']:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("cv_results.csv", index=False)

# Paired t-test: proposed hybrid stacking vs strongest baseline, on per-fold F1
baseline_key = "Baseline: raw features + XGBoost"
proposed_key = "Hybrid features + Stacking Ensemble (proposed)"
t_stat, p_val = stats.ttest_rel(fold_f1[proposed_key], fold_f1[baseline_key])
print(f"\nPaired t-test (F1, 10-fold), proposed vs best baseline: t={t_stat:.3f}, p={p_val:.4f}")

np.save("fold_f1.npy", fold_f1, allow_pickle=True)

