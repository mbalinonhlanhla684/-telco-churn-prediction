Telco Customer Churn Prediction — End-to-End ML Pipeline

**Author:** Nonhlanhla Mabli Ntuli

Problem Statement
Customer churn (subscribers cancelling service) directly impacts telecom revenue. The cost of retaining an existing customer is significantly lower than acquiring a new one, so the business goal is to **identify at-risk customers before they leave**, enabling targeted retention offers. This project builds a classification pipeline that predicts churn probability from customer account and usage data.

Dataset
**Source:** [Telco Customer Churn dataset, Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
**Size:** 7,043 customers, 21 features
**Feature categories:** demographics (gender, senior citizen status, dependents), account details (tenure, contract type, payment method, billing), service usage (internet type, streaming, security/backup add-ons), and billing amounts (monthly and total charges)
**Target:** `Churn` (Yes/No) — class distribution is imbalanced at 73.5% No / 26.5% Yes, which directly shaped the modeling approach below

Data Cleaning & Validation

Rather than cleaning ad hoc, all logic was consolidated into a single, tested, idempotent function — `clean_telco_data()` — designed to be safely re-run on the same data without side effects (a common real-world requirement when pipelines get re-triggered).

**Issues identified and resolved:**

| Issue | Root Cause | Resolution |
|---|---|---|
| `TotalCharges` stored as `object` (text) dtype instead of numeric | 11 rows contained blank-string values (`" "`) instead of numbers | Converted via `pd.to_numeric(errors='coerce')`, then diagnosed root cause before fixing |
| Diagnosis of the 11 missing values | Investigated by cross-referencing against `tenure` — all 11 had `tenure == 0` | Confirmed these were new customers with no completed billing cycle (not data corruption); filled with `0` rather than dropping rows, preserving 100% of records |
| `SeniorCitizen` encoded as `0`/`1` while all other binary fields used `'Yes'`/`'No'` text | Inconsistent source encoding | Standardized to `'Yes'`/`'No'`, with a dtype check added so the function doesn't double-convert already-clean data (idempotency safeguard) |
| Potential exact-duplicate customer records | N/A — verified, not found | Ran `.duplicated().sum()` as a standard validation check (returned 0) |
| Categorical columns with unexpected cardinality (e.g. 3 values instead of 2) | Columns like `OnlineSecurity` include a `"No internet service"` category for customers without internet | Investigated with `.unique()` per column; confirmed as valid business logic, not an error, before proceeding |

Engineering practices applied:**
- Cleaning function takes raw, untransformed data as input and is safe to call multiple times (verified by testing against a freshly reloaded CSV, which caught a real bug: re-running the original inline `SeniorCitizen` mapping on already-converted data silently corrupted the entire column to `NaN`)
- Every fix was diagnosed *before* being applied — no blind imputation. E.g., the `TotalCharges` nulls were traced to a specific customer segment (zero-tenure) before deciding a fill value, rather than defaulting to mean/median imputation without justification

Preprocessing & Feature Engineering
- Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) standardized via `StandardScaler` (zero mean, unit variance) — necessary for logistic regression's gradient-based optimization to converge efficiently (confirmed by resolving a `ConvergenceWarning` after scaling was added)
- Categorical features one-hot encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')` — `drop='first'` avoids multicollinearity from redundant dummy columns; `handle_unknown='ignore'` ensures the pipeline won't crash in production if it encounters an unseen category
- All preprocessing and modeling steps combined into a single `sklearn.pipeline.Pipeline` object using `ColumnTransformer` to route numeric vs. categorical columns to their correct transforms — this makes the entire preprocessing + inference logic a single serializable, deployable artifact (`telco_churn_pipeline.pkl`), eliminating the risk of preprocessing/training mismatch at inference time

Modeling & Evaluation Methodology

**Models compared:** Logistic Regression, Random Forest, XGBoost

**Class imbalance handling:** Used `class_weight='balanced'` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost, computed as the ratio of negative to positive class counts) rather than naive resampling, to avoid distorting the dataset's real-world distribution.

**Threshold tuning:** Rather than accepting the default 0.5 classification threshold, tested thresholds from 0.3–0.7 for each model and selected the value maximizing F1-score on the minority (churn) class — since accuracy alone is misleading on imbalanced data (a model predicting "No churn" for every customer would score ~73.5% accuracy while being useless).

**Validation:** Ran 5-fold cross-validation on the final model to confirm the reported F1-score wasn't an artifact of a single lucky train/test split (mean F1 = 0.627, std dev = 0.011 — low variance confirms stability).

Results

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | Accuracy |
|---|---|---|---|---|
| **Logistic Regression (final, threshold=0.6)** | 0.57 | 0.74 | **0.65** | 0.79 |
| Random Forest (tuned threshold=0.3) | 0.55 | 0.76 | 0.63 | 0.77 |
| XGBoost (tuned threshold=0.6) | 0.59 | 0.62 | 0.61 | 0.79 |

Logistic Regression was selected as the final model — despite being the simplest of the three, it consistently outperformed both ensemble methods on F1 across multiple thresholds, likely due to the dataset's largely linear, categorical structure. This result was deliberately not assumed going in; all three models were tuned and fairly compared before selecting a winner, rather than defaulting to the most sophisticated option.

Key Business Insights (from model coefficients)

| Factor | Effect | Interpretation |
|---|---|---|
| Tenure | Strong ↓ churn | Long-standing customers are highly loyal |
| Two-year contract | Strong ↓ churn | Contract lock-in is the single strongest retention lever |
| Fiber optic internet | Strong ↑ churn | Possible service quality or price sensitivity issue worth business investigation |
| Total charges | ↑ churn | High cumulative spend combined with shorter tenure signals risk |
| Electronic check payment | ↑ churn | Less "sticky" payment method vs. automatic billing |

Reproducibility
- All cleaning and preprocessing logic is encapsulated in functions/pipeline objects, not inline notebook cells — runnable end-to-end on fresh data
- Final artifacts saved via `joblib`: `telco_churn_pipeline.pkl` (full preprocessing + model in one object)
- Model performance independently validated via cross-validation, not a single train/test split

 Tech Stack
Python · pandas · scikit-learn · XGBoost · matplotlib · Google Colab

Project Structure
```
├── telco_churn_prediction.ipynb   # Full notebook: cleaning, EDA, modeling, evaluation
├── telco_churn_pipeline.pkl       # Saved end-to-end pipeline (preprocessing + model)
├── feature_importance.png         # Bar chart of top churn-driving features
└── README.md                      # This file
```
