# Source Code — CVD Ensemble Manuscript

Analysis pipeline and statistical scripts underlying the manuscript
"Feature Engineering and Stacked Ensembles for Heart Disease
Classification: An Ablation and Validation Study" (target: *The
Cardiothoracic Surgeon*, Springer Nature Collection "Artificial
Intelligence and Predictive Analytics in Cardiothoracic Care").

Dataset (not included here — see manuscript Declarations): the merged
Statlog + Cleveland + Hungary heart disease dataset
(`heart_statlog_cleveland_hungary_final.csv`, 1,190 rows / 918 after
deduplication), available at
https://www.kaggle.com/datasets/mexwell/heart-disease-dataset,
originally sourced from the UCI Machine Learning Repository
(Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease;
Statlog (Heart): https://archive.ics.uci.edu/dataset/145/statlog+heart).

## Scripts

| File | Purpose |
|---|---|
| `pipeline.py` | Core hybrid feature-engineering + stacked-ensemble pipeline. Defines `ClinicalFeatureEngineer` (domain-derived clinical features: heart-rate deficit, ischemia interaction terms, risk-factor count, guideline-threshold categories), `ClusterPCAAugmenter` (K-means cluster ID/distance + PCA components), and `HybridFeaturePipeline`. Runs the multi-configuration comparison (raw vs. hybrid features x single-model vs. voting vs. stacking) with 10-fold CV, including the leakage-corrected internal nesting (each base learner wrapped as a full `Pipeline(features + classifier)` refit inside each internal fold of `StackingClassifier`). Produces the corrected per-fold performance scores used in Table 1/2.
| `collect_per_fold.py` | Aggregates per-fold metrics across configurations/seeds into the summary arrays used for the paired statistical tests. |
| `oof_predictions.py` | Generates pooled out-of-fold predicted probabilities for the key model configurations (raw XGBoost, hybrid XGBoost, raw stacking, hybrid stacking) on the deduplicated cohort — the basis for DeLong's test, McNemar's test, calibration curves, and the patient-level bootstrap CIs. |
| `nested_cv.py` | Outer/inner nested cross-validation driver used to test whether the headline hybrid-stacking advantage survives when hyperparameter selection is confined to training folds only. |
| `nested_cv_xgb.py` | Nested CV specialization for the raw/hybrid XGBoost comparison. |
| `nested_cv_stack.py` | Nested CV specialization for the stacking ensemble. |
| `ablation.py` | Feature-ablation study isolating the contribution of clustering vs. PCA vs. clinical-derived features to model performance. |
| `advanced_stats.py` | Statistical battery: paired t-test, Wilcoxon signed-rank, Cohen's d, DeLong's test, McNemar's test, Friedman test with Nemenyi post-hoc, Hosmer-Lemeshow calibration test. |
| `more_stats.py` | Supplementary statistical computations (patient-level bootstrap confidence intervals, additional effect-size/robustness checks) supporting the Statistical Validation subsection. |
| `shap_and_rfe.py` | Interpretability analysis: SHAP (TreeExplainer) feature-importance values for the hybrid XGBoost model, Recursive Feature Elimination, and Variance Inflation Factor multicollinearity diagnostics. |
| `fix_bootstrap_fig.py` | Utility script that regenerates the bootstrap-distribution figure (`fig_bootstrap.pdf`) from saved results, used to correct a figure/table numerical mismatch caught during review. |

## Reproducibility notes

- All scripts assume a fixed random seed (documented in `pipeline.py`) for
  reproducibility, per the manuscript's Software Stack and Reproducibility
  subsection (Section 4.2).
- The corrected pipeline (leakage fix: full preprocessing pipeline refit
  inside each internal CV fold, not fit once per outer fold) is what
  produced the final numbers reported in the manuscript. Results computed
  from a naively-refit (leaky) version are NOT reported as primary
  findings anywhere in the manuscript — they were used only to identify
  and document the leakage issue itself (see Methods, Validation Protocol).
- Duplicate-removal step (272 cross-source duplicate patient records,
  544/1,190 rows, reducing the cohort to n=918) is performed at the start
  of `pipeline.py` before any modeling.

For the compiled manuscript and its own source (`.tex`, `.bib`, figures),
see the companion package `manuscript_cts_source_final.zip`.
# hybrid-feature-engineering-and-stacked-ensemble-pipeline-for-CVD-prediction1
# hybrid-feature-engineering-and-stacked-ensemble-pipeline-for-CVD-prediction
# hybrid-feature-engineering-and-stacked-ensemble-pipeline-for-CVD-prediction
