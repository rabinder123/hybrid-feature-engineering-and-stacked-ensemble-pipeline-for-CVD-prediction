import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import scikit_posthocs as sp

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ============================================================
# Load per-fold scores (6 configs x 10 folds) and OOF probabilities
# ============================================================
pf = np.load("per_fold_scores.npz")
oof = np.load("oof_probs.npz")
y = oof["y"]
p_base = oof["Baseline_XGBoost_(raw)"]
p_prop = oof["Hybrid_Stacking_(proposed)"]

f1_prop = pf["Hybrid+Stacking__f1"]
f1_base = pf["Raw+XGB__f1"]
acc_prop = pf["Hybrid+Stacking__accuracy"]
acc_base = pf["Raw+XGB__accuracy"]
auc_prop = pf["Hybrid+Stacking__roc_auc"]
auc_base = pf["Raw+XGB__roc_auc"]

print("="*70)
print("1. PAIRED COMPARISON: proposed vs. raw-feature XGBoost baseline")
print("="*70)

# --- Paired t-test (already reported) ---
t_stat, t_p = stats.ttest_rel(f1_prop, f1_base)
print(f"Paired t-test on F1:      t={t_stat:.3f}, p={t_p:.4f}")

# --- Wilcoxon signed-rank test (nonparametric alternative) ---
w_stat, w_p = stats.wilcoxon(f1_prop, f1_base)
print(f"Wilcoxon signed-rank F1:  W={w_stat:.3f}, p={w_p:.4f}")

# --- Effect size: Cohen's d for paired samples ---
diff = f1_prop - f1_base
cohens_d = diff.mean() / diff.std(ddof=1)
print(f"Cohen's d (paired, F1):   d={cohens_d:.3f}  (mean diff={diff.mean():.4f}, SD diff={diff.std(ddof=1):.4f})")

# ============================================================
# 2. Bootstrap 95% CIs (resampling patients, using OOF predictions)
# ============================================================
print("\n" + "="*70)
print("2. BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 resamples)")
print("="*70)

n = len(y)
n_boot = 2000

def metrics_at(idx, probs):
    yy = y[idx]; pp = probs[idx]
    preds = (pp >= 0.5).astype(int)
    if len(set(yy)) < 2:
        return np.nan, np.nan, np.nan
    acc = (preds == yy).mean()
    tp = ((preds == 1) & (yy == 1)).sum(); fp = ((preds == 1) & (yy == 0)).sum()
    fn = ((preds == 0) & (yy == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    auc = stats.rankdata(pp)[yy == 1].mean()  # placeholder, replaced below with sklearn
    return acc, f1, auc

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

boot_acc_prop, boot_f1_prop, boot_auc_prop = [], [], []
boot_acc_base, boot_f1_base, boot_auc_base = [], [], []
boot_diff_f1, boot_diff_auc = [], []

for b in range(n_boot):
    idx = rng.integers(0, n, n)
    yy = y[idx]
    if len(set(yy)) < 2:
        continue
    pp_prop = p_prop[idx]; pp_base = p_base[idx]
    pred_prop = (pp_prop >= 0.5).astype(int); pred_base = (pp_base >= 0.5).astype(int)

    a_p = accuracy_score(yy, pred_prop); f_p = f1_score(yy, pred_prop, zero_division=0); au_p = roc_auc_score(yy, pp_prop)
    a_b = accuracy_score(yy, pred_base); f_b = f1_score(yy, pred_base, zero_division=0); au_b = roc_auc_score(yy, pp_base)

    boot_acc_prop.append(a_p); boot_f1_prop.append(f_p); boot_auc_prop.append(au_p)
    boot_acc_base.append(a_b); boot_f1_base.append(f_b); boot_auc_base.append(au_b)
    boot_diff_f1.append(f_p - f_b); boot_diff_auc.append(au_p - au_b)

def ci(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

print(f"Proposed  Accuracy: {np.mean(boot_acc_prop):.4f}  95% CI {ci(boot_acc_prop)}")
print(f"Proposed  F1:       {np.mean(boot_f1_prop):.4f}  95% CI {ci(boot_f1_prop)}")
print(f"Proposed  AUC:      {np.mean(boot_auc_prop):.4f}  95% CI {ci(boot_auc_prop)}")
print(f"Baseline  Accuracy: {np.mean(boot_acc_base):.4f}  95% CI {ci(boot_acc_base)}")
print(f"Baseline  F1:       {np.mean(boot_f1_base):.4f}  95% CI {ci(boot_f1_base)}")
print(f"Baseline  AUC:      {np.mean(boot_auc_base):.4f}  95% CI {ci(boot_auc_base)}")
print(f"Diff (proposed-baseline) F1:  mean={np.mean(boot_diff_f1):.4f}  95% CI {ci(boot_diff_f1)}")
print(f"Diff (proposed-baseline) AUC: mean={np.mean(boot_diff_auc):.4f}  95% CI {ci(boot_diff_auc)}")
frac_f1_positive = np.mean(np.array(boot_diff_f1) > 0)
frac_auc_positive = np.mean(np.array(boot_diff_auc) > 0)
print(f"Fraction of bootstrap resamples where proposed > baseline: F1 {frac_f1_positive:.3f}, AUC {frac_auc_positive:.3f}")

# ============================================================
# 3. DeLong's test for correlated ROC AUCs
# ============================================================
print("\n" + "="*70)
print("3. DELONG'S TEST (proposed vs. baseline AUC, correlated)")
print("="*70)

def delong_variance(y_true, scores):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    pos = np.where(y_sorted == 1)[0]
    neg = np.where(y_sorted == 0)[0]
    m, n = len(pos), len(neg)
    scores_sorted = scores[order]
    pos_scores = scores_sorted[pos]
    neg_scores = scores_sorted[neg]
    tx = np.array([ (pos_scores > s).sum() + 0.5*(pos_scores == s).sum() for s in neg_scores]) / m
    ty = np.array([ (neg_scores < s).sum() + 0.5*(neg_scores == s).sum() for s in pos_scores]) / n
    auc = ty.mean()
    v10 = ty
    v01 = tx
    return auc, v10, v01, pos, neg

def delong_test(y_true, scores_a, scores_b):
    auc_a, v10_a, v01_a, pos, neg = delong_variance(y_true, scores_a)
    auc_b, v10_b, v01_b, _, _ = delong_variance(y_true, scores_b)
    m, n = len(pos), len(neg)
    s10 = np.cov(np.vstack([v10_a, v10_b])) / m
    s01 = np.cov(np.vstack([v01_a, v01_b])) / n
    s = s10 + s01
    var = s[0,0] + s[1,1] - 2*s[0,1]
    z = (auc_a - auc_b) / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p

auc_a, auc_b, z, p_delong = delong_test(y, p_prop, p_base)
print(f"AUC proposed={auc_a:.4f}, AUC baseline={auc_b:.4f}")
print(f"DeLong z={z:.3f}, p={p_delong:.4f}")

# ============================================================
# 4. Friedman test + Nemenyi post-hoc across ALL SIX configurations
# ============================================================
print("\n" + "="*70)
print("4. FRIEDMAN TEST + NEMENYI POST-HOC (6 configs x 10 folds, F1)")
print("="*70)

config_names = ["Raw+LogReg", "Raw+RF", "Raw+XGB", "Hybrid+XGB", "Hybrid+Voting", "Hybrid+Stacking"]
f1_matrix = np.column_stack([pf[f"{c}__f1"] for c in config_names])  # 10 folds x 6 configs
mean_ranks = pd.Series(
    stats.rankdata(-f1_matrix, axis=1).mean(axis=0), index=config_names
).sort_values()
print("Mean rank per model (lower = better):")
print(mean_ranks)

fried_stat, fried_p = stats.friedmanchisquare(*[f1_matrix[:, i] for i in range(f1_matrix.shape[1])])
print(f"\nFriedman chi-square={fried_stat:.3f}, p={fried_p:.5f}")

nemenyi = sp.posthoc_nemenyi_friedman(f1_matrix)
nemenyi.columns = config_names
nemenyi.index = config_names
print("\nNemenyi post-hoc pairwise p-values:")
print(nemenyi.round(4))
nemenyi.to_csv("nemenyi_results.csv")

print("\nALL_STATS_DONE")
