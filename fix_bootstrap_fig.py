import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

rng = np.random.default_rng(42)
oof = np.load('oof_probs.npz')
y = oof['y']; p_prop = oof['Hybrid_Stacking_(proposed)']; p_base = oof['Baseline_XGBoost_(raw)']
n = len(y); n_boot = 2000
diff_f1 = []
for b in range(n_boot):
    idx = rng.integers(0, n, n)
    yy = y[idx]
    if len(set(yy)) < 2:
        continue
    pred_p = (p_prop[idx] >= 0.5).astype(int)
    pred_b = (p_base[idx] >= 0.5).astype(int)
    diff_f1.append(f1_score(yy, pred_p, zero_division=0) - f1_score(yy, pred_b, zero_division=0))
diff_f1 = np.array(diff_f1)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(diff_f1, bins=50, color='tab:blue', alpha=0.75, edgecolor='white', linewidth=0.3)
ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
ci_lo, ci_hi = np.percentile(diff_f1, 2.5), np.percentile(diff_f1, 97.5)
ax.axvline(ci_lo, color='red', linewidth=1, linestyle='--')
ax.axvline(ci_hi, color='red', linewidth=1, linestyle='--', label='95% CI')
ax.set_xlabel(r'F1 (proposed) $-$ F1 (baseline)')
ax.set_ylabel('Count (of 2,000 bootstrap resamples)')
ax.set_title('Bootstrap distribution of paired F1 difference')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig_bootstrap.pdf')
plt.close()
print('Fixed bootstrap figure saved. CI:', ci_lo, ci_hi)
