import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score

d = np.load("oof_probs.npz")
y = d["y"]
p_base = d["Baseline_XGBoost_(raw)"]
p_prop = d["Hybrid_Stacking_(proposed)"]
pred_base = (p_base >= 0.5).astype(int)
pred_prop = (p_prop >= 0.5).astype(int)

print("="*70)
print("1. McNEMAR'S TEST (paired predictions, proposed vs baseline)")
print("="*70)
correct_base = (pred_base == y)
correct_prop = (pred_prop == y)

# Discordant pairs
n01 = np.sum(~correct_base & correct_prop)   # base wrong, proposed right
n10 = np.sum(correct_base & ~correct_prop)   # base right, proposed wrong
n00 = np.sum(~correct_base & ~correct_prop)  # both wrong
n11 = np.sum(correct_base & correct_prop)    # both right

print(f"Contingency table: both correct={n11}, both wrong={n00}, "
      f"baseline-only-correct={n10}, proposed-only-correct={n01}")

# Exact McNemar (binomial test on discordant pairs) - appropriate given n10+n01 < 25 possibly, check
n_discordant = n10 + n01
print(f"Discordant pairs: {n_discordant} (baseline-only={n10}, proposed-only={n01})")

# Exact binomial test
mcnemar_exact_p = stats.binomtest(n01, n_discordant, p=0.5, alternative='two-sided').pvalue
print(f"McNemar exact (binomial) test: p={mcnemar_exact_p:.4f}")

# Chi-square version with continuity correction (standard McNemar)
if n_discordant > 0:
    chi2_stat = (abs(n10 - n01) - 1)**2 / n_discordant
    chi2_p = 1 - stats.chi2.cdf(chi2_stat, df=1)
    print(f"McNemar chi-square (continuity-corrected): chi2={chi2_stat:.3f}, p={chi2_p:.4f}")

print("\n" + "="*70)
print("2. PER-MODEL BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 resamples)")
print("="*70)
rng = np.random.default_rng(42)
n = len(y)
n_boot = 2000

results = {"baseline": {"acc": [], "f1": [], "auc": [], "ap": []},
           "proposed": {"acc": [], "f1": [], "auc": [], "ap": []}}

for b in range(n_boot):
    idx = rng.integers(0, n, n)
    yy = y[idx]
    if len(set(yy)) < 2:
        continue
    for name, probs in [("baseline", p_base), ("proposed", p_prop)]:
        pp = probs[idx]
        preds = (pp >= 0.5).astype(int)
        results[name]["acc"].append(accuracy_score(yy, preds))
        results[name]["f1"].append(f1_score(yy, preds, zero_division=0))
        results[name]["auc"].append(roc_auc_score(yy, pp))
        results[name]["ap"].append(average_precision_score(yy, pp))

for name in ["baseline", "proposed"]:
    print(f"\n{name.upper()}:")
    for metric in ["acc", "f1", "auc", "ap"]:
        arr = np.array(results[name][metric])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {metric:4s}: mean={arr.mean():.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

print("\n" + "="*70)
print("3. PRECISION-RECALL AUC (AVERAGE PRECISION)")
print("="*70)
ap_base = average_precision_score(y, p_base)
ap_prop = average_precision_score(y, p_prop)
print(f"Baseline  AP (PR-AUC): {ap_base:.4f}")
print(f"Proposed  AP (PR-AUC): {ap_prop:.4f}")
print(f"Difference: {ap_prop - ap_base:+.4f}")

print("\n" + "="*70)
print("4. HOSMER-LEMESHOW CALIBRATION GOODNESS-OF-FIT TEST")
print("="*70)

def hosmer_lemeshow(y_true, y_prob, g=10):
    df = np.column_stack([y_true, y_prob])
    order = np.argsort(df[:, 1])
    df = df[order]
    groups = np.array_split(np.arange(len(df)), g)
    chi2 = 0.0
    table = []
    for grp in groups:
        obs_events = df[grp, 0].sum()
        n_grp = len(grp)
        exp_events = df[grp, 1].sum()
        obs_nonevents = n_grp - obs_events
        exp_nonevents = n_grp - exp_events
        if exp_events > 0:
            chi2 += (obs_events - exp_events) ** 2 / exp_events
        if exp_nonevents > 0:
            chi2 += (obs_nonevents - exp_nonevents) ** 2 / exp_nonevents
        table.append((n_grp, obs_events, exp_events))
    dof = g - 2
    p_val = 1 - stats.chi2.cdf(chi2, dof)
    return chi2, p_val, table

for name, probs in [("Baseline XGBoost", p_base), ("Proposed Stacking", p_prop)]:
    chi2, p_val, table = hosmer_lemeshow(y, probs, g=10)
    print(f"{name}: HL chi2={chi2:.3f}, df=8, p={p_val:.4f}  "
          f"({'no evidence of miscalibration' if p_val > 0.05 else 'evidence of miscalibration'})")

print("\nDONE")
