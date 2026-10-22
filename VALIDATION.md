# Validation

Checked on 2026-10-22 with Python 3.12.14, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2 and seaborn 0.13.2.

```text
python smoke_test.py
PASS: 9 notebook code cells; 9 models; identifiers/targets excluded; unseen category handled.
Synthetic fixtures only. No real-data or clinical performance claim.
```

The smoke test executed every notebook code cell on deterministic synthetic fixtures: 150 training rows and 45 holdout rows. It covered the cleaning and plotting workflow, five-fold cross-validation and holdout evaluation for three binary classifiers, three multiclass classifiers and three regressors. It asserted finite evaluation metrics, exclusion of both identifier spellings and all three target columns from predictors, and use of preprocessing/PCA/model pipelines. A new categorical value appeared only in the holdout data; the expected encoder warning was emitted and prediction completed.

The original CSV files were unavailable during these checks. Saved historical notebook outputs were cleared, and no synthetic scores are presented as original empirical results. This validates execution and selected implementation properties; it does not validate the study's original results, statistical design or clinical applicability. The IQR row filtering limitation is described in the README.
