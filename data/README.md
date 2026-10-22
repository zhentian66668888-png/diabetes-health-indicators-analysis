# Local data schema

Supply `Diabetes.csv` for training and `Diabetes_test.csv` for holdout evaluation. Data is loaded from this directory by default, or from `DIABETES_DATA_DIR`.

Required fields in both files:

| Column | Use |
|---|---|
| `diagnosed_diabetes` | Recorded binary target |
| `diabetes_stage` | Recorded categorical target |
| `diabetes_risk_score` | Recorded numeric target |
| `bmi` | Numeric predictor used in the exploratory plots |

Additional numeric and categorical predictor columns are supported. The test data must contain every training predictor. Optional `patient_id` and `Patient ID` columns are excluded from model predictors; omit identifiers when preparing analytical data where possible. Missing rows are dropped under the notebook's existing complete-case policy.

The CSV source and permissions must be supplied by the data owner. No patient-level files are included in this repository. `smoke_test.py` generates its own clearly synthetic temporary inputs.
