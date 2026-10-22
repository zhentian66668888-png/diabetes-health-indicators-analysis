# Diabetes Health Indicators Analysis

An independently completed Python project exploring health indicators through data cleaning, visualization and predictive modeling.

## Analysis

- Inspect missing values and filter extreme training observations using an IQR rule.
- Visualize target distributions, feature correlations and BMI by recorded diabetes stage.
- Compare Logistic Regression, Random Forest and SVM for binary classification.
- Compare Decision Tree, Random Forest and KNN for multiclass classification.
- Compare Linear Regression, Ridge and Random Forest for risk-score regression.
- Use encoding, scaling and PCA inside model pipelines so that these transformations are fitted within each cross-validation fold.

The full notebook is included, with all nine original code cells and their analysis sections. Limited publication corrections are documented in [CHANGELOG.md](CHANGELOG.md).

## Run with the original data

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab main.ipynb
```

Place `Diabetes.csv` and `Diabetes_test.csv` in `data/`, or set `DIABETES_DATA_DIR` to a local directory containing both files. Both splits need the three recorded target columns because this notebook evaluates predictions. See [the data schema](data/README.md).

The original data source and redistribution terms have not been documented here, and the files were unavailable during publication checks. Real records are not distributed with the code. Use only data you are authorized to process.

## Verify the code without real records

```bash
python smoke_test.py
```

This generates synthetic fixtures in a temporary directory and executes all nine code cells, including cross-validation and holdout prediction for all nine models. It also checks identifier/target exclusion and a previously unseen categorical value. Synthetic metrics are software checks, not findings about real people. See [VALIDATION.md](VALIDATION.md).

## Interpretation and limitations

Original study notes are retained in the final notebook section and clearly marked as historical, unreproduced interpretation. There are no newly claimed real-data scores in this public edition.

The IQR row filter is computed on the complete training partition before cross-validation, so those cross-validation estimates can still be optimistic. An improved evaluation would nest the row-filtering policy and model selection within the evaluation design. Outlier removal and complete-case analysis can also change the population represented by the results. Feature timing and overlap with recorded target definitions must be checked before drawing predictive conclusions.

This is an educational analysis workflow, not a clinically validated diagnostic tool.
