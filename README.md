# 🛡️ IEEE-CIS Fraud Detection — End-to-End Machine Learning Pipeline
    
**Dataset:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

---

## 1. Business Problem / Motivation

Credit card fraud is one of the most costly problems in the financial industry. According to industry reports, billions of dollars are lost each year to fraudulent transactions. For financial institutions, failing to catch fraud means direct monetary loss and damage to customer trust. At the same time, incorrectly flagging a legitimate transaction as fraud frustrates customers and can lead to lost business.

This project addresses a real-world binary classification problem: **given a financial transaction and its associated identity/device information, predict whether it is fraudulent.**

The challenge is not just building an accurate model — it is building one that is sensitive enough to catch fraud (high recall) while remaining interpretable enough for a business to trust and act on. That is the goal of this capstone.

---

## 2. Project Overview

| | |
|---|---|
| **Goal** | Predict whether a financial transaction is fraudulent (binary classification) |
| **Approach** | End-to-end ML pipeline: data cleaning → EDA → multiple models → imbalance handling → XAI |
| **Best Model** | XGBoost with threshold tuning and feature engineering |
| **Best ROC-AUC** | 0.9360 (baseline XGBoost), improved further with threshold tuning to target recall ≥ 0.80 |
| **Key Techniques** | SMOTE, ADASYN, Random Oversampling, SHAP, Feature Engineering, 5-Fold Stratified CV |

---

## 3. Data

- **Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **Type:** Tabular, anonymized financial transaction data
- **Size:** ~590,000 transactions across two files
  - `train_transaction.csv` — transaction-level features
  - `train_identity.csv` — identity/device features linked by `TransactionID`
- **Target Variable:** `isFraud` (0 = legitimate, 1 = fraud)
- **Class Distribution:** ~96.5% non-fraud, ~3.5% fraud — heavily imbalanced

**Key Features:**

| Feature | Description |
|--------|-------------|
| `TransactionAmt` | Dollar amount of the transaction |
| `TransactionDT` | Time offset in seconds from a reference point |
| `card1`–`card6` | Anonymized payment card identifiers and attributes |
| `addr1`, `addr2` | Billing address codes |
| `P_emaildomain` | Purchaser email domain |
| `R_emaildomain` | Recipient email domain |
| `C1`–`C14` | Count-based behavioral features (transaction frequency) |
| `D1`–`D15` | Time-delta features (days since related events) |
| `V1`–`V339` | Anonymized Vesta-engineered features capturing risk signals |

> **Note:** The data cannot be redistributed per Kaggle's terms. Download it directly from the Kaggle link above and place the CSVs in the `data/` folder before running notebooks.

---

## 4. Data Preprocessing

### 4.1 Merging Datasets
The two source files were joined on `TransactionID` using a left join, preserving all transactions while attaching available identity information.

```python
data = pd.merge(transactions, identity, on="TransactionID", how="left")
# Result: ~590,000 rows, 433 columns
```

### 4.2 Handling Missing Values

**Step 1 — Drop sparse columns:** Any column with more than 70% missing values was removed. This eliminated highly sparse features that would add noise without meaningful signal.

```python
cols_to_drop = missing_percent[missing_percent > 70].index
data = data.drop(columns=cols_to_drop)
# Reduced from 433 → ~200 columns
```

**Step 2 — Fill remaining missing values (after train/test split in final implementation):**
- Numeric columns → filled with the **median** (robust to outliers)
- Categorical columns → filled with the **mode** (most common value)

> In the final implementation, missing values are filled **after** the train/test split to prevent data leakage from the test set influencing training statistics.

### 4.3 Feature Engineering (Final Implementation)

| New Feature | Method | Why |
|---|---|---|
| `log_TransactionAmt` | `np.log1p(TransactionAmt)` | Transaction amounts are right-skewed; log transform compresses extremes and improves model learning |
| `email_domain_match` | `P_emaildomain == R_emaildomain` | Mismatched purchaser and recipient email domains are a known fraud signal |
| `missing_count` | Count of nulls per row | Fraudsters often submit incomplete identity/device information |
| Frequency Encoding | Replace high-cardinality cols with value frequency | Avoids dummy column explosion while preserving cardinality signal |

### 4.4 Encoding and Splitting

- Categorical variables encoded with `pd.get_dummies()` in the baseline pipeline
- Frequency encoding used in the final implementation for high-cardinality columns
- 80/20 stratified train/test split to preserve the 3.5% fraud ratio across both sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 4.5 Feature Scaling

`StandardScaler` applied for models requiring scaled inputs (Logistic Regression, Naive Bayes, Neural Network). Tree-based models do not require scaling and were trained on the raw feature matrix.

---

## 5. Exploratory Data Analysis (EDA)

### Key Visualizations

**1. Fraud vs. Non-Fraud Distribution**
Only 3.5% of transactions are fraudulent. This immediately signals that standard accuracy is misleading and that class imbalance techniques are required.

**2. Transaction Amount by Fraud Status (Boxplot)**
Fraudulent transactions show a different distribution — the median fraud amount is lower, but the range includes more extreme outliers — supporting the use of `log_TransactionAmt`.

**3. Fraud Rate by Product Type (ProductCD)**
Product type `C` has the highest fraud rate (~6.5%), while type `W` has the lowest (~1.3%). Product category is a meaningful predictor.

**4. Correlation Heatmap of Top Fraud-Related Features**
The top 10 features most correlated with `isFraud` were identified. Several C-series count features show moderate positive correlation with fraud.

### Additional Insights
- Discover cards had the highest fraud rate (~5.8%) among card networks
- Anonymous and protonmail email domains showed significantly higher fraud rates than mainstream providers
- Very low and very high transaction amounts had elevated fraud rates — consistent with test charges and large fraudulent purchases

---

## 6. Modeling Approach

| Model | Type | Reason for Inclusion |
|---|---|---|
| Logistic Regression | Baseline | Simple, interpretable baseline; establishes minimum performance bar |
| Naive Bayes | Probabilistic | Fast probabilistic baseline; useful for benchmarking recall behavior |
| Decision Tree | Simple | Interpretable tree-based model; shows single-tree limitations vs. ensembles |
| Random Forest | Ensemble | Strong out-of-the-box performer; handles non-linearity and missing values well |
| XGBoost | Gradient Boosting | State-of-the-art for tabular data; supports `scale_pos_weight` for imbalance |
| Neural Network (MLP) | Deep Learning | Tests whether a neural approach offers any advantage over gradient boosting |

**Final Model Choice: XGBoost**
XGBoost achieved the best ROC-AUC (0.936) with the strongest overall balance of precision and recall. It also integrates cleanly with SHAP for interpretability, which is critical for a fraud detection use case.

---

## 7. Model Training

### Tools Used
- `scikit-learn` — Logistic Regression, Naive Bayes, Decision Tree, Random Forest, preprocessing, metrics
- `xgboost` — XGBClassifier
- `imbalanced-learn` — RandomOverSampler, SMOTE, ADASYN
- `shap` — TreeExplainer, summary plots, waterfall plots
- `joblib` — model serialization

### Hyperparameters

**Random Forest:**
```python
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

**XGBoost (Baseline):**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
```

**XGBoost (Final — with class weighting):**
```python
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
XGBClassifier(..., scale_pos_weight=scale_pos_weight)
```

### Training Process
- Tree-based models trained on the full 80% training split
- Models requiring scaling (LR, NB, MLP) trained on a 100,000-sample subset for efficiency
- Final XGBoost model validated using **5-fold Stratified Cross-Validation**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_final, X_train_top, y_train, cv=cv, scoring="roc_auc")
```

---

## 8. Results

### Why These Metrics?

| Metric | Why It Matters Here |
|---|---|
| **Recall** | Most important — measures how many actual fraud cases we catch. Missing fraud = direct financial loss |
| **Precision** | Measures how often a fraud alert is correct. Low precision = too many false alarms |
| **F1 Score** | Harmonic mean of precision and recall — balances both in a single number |
| **ROC-AUC** | Measures the model's ability to rank fraud above non-fraud; threshold-independent |

### Model Comparison Table

| Model | Precision | Recall | F1 Score | ROC-AUC | Test Set |
|---|---|---|---|---|---|
| Logistic Regression | 0.82 | 0.17 | 0.28 | 0.833 | Sampled |
| Naive Bayes | 0.04 | 0.99 | 0.07 | 0.513 | Sampled |
| Decision Tree | 0.53 | 0.57 | 0.55 | 0.778 | Full |
| Neural Network (MLP) | 0.69 | 0.39 | 0.50 | 0.861 | Sampled |
| Random Forest | 0.92 | 0.46 | 0.61 | 0.930 | Full |
| **XGBoost** | **0.90** | 0.44 | 0.59 | **0.936** | Full |

### After Imbalance Handling

| Model | Technique | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | Baseline | 0.92 | 0.46 | 0.61 | 0.930 |
| Random Forest | + Random Oversampling | 0.55 | 0.70 | 0.62 | ~0.930 |
| Random Forest | + SMOTE | 0.54 | 0.72 | 0.61 | ~0.929 |
| Random Forest | + ADASYN | 0.53 | 0.73 | 0.61 | ~0.929 |
| XGBoost | Baseline | 0.90 | 0.44 | 0.59 | 0.936 |
| XGBoost | + Random Oversampling | 0.58 | 0.68 | 0.63 | ~0.934 |
| XGBoost | + SMOTE | 0.57 | 0.70 | 0.62 | ~0.933 |
| XGBoost | + ADASYN | 0.56 | 0.71 | 0.62 | ~0.933 |

### Final Model — With Threshold Tuning

The classification threshold was lowered from 0.50 to ~0.30 to target recall ≥ 0.80. In fraud detection, missing a fraud case is more costly than a false alarm.

| Model | ROC-AUC | Recall | F1 Score | Threshold |
|---|---|---|---|---|
| XGBoost (Final) | ~0.940 | 0.80+ | 0.62+ | ~0.30 |
| XGBoost + RF Ensemble | ~0.938 | ~0.79 | ~0.61 | ~0.30 |

---

## 9. Model Interpretation

### Global — Feature Importance

| Rank | Feature | Why It Matters |
|---|---|---|
| 1 | `TransactionAmt` | Unusual amounts (very high or very low) are a primary fraud signal |
| 2 | `TransactionDT` | Time of transaction encodes behavioral patterns; fraudsters often act at unusual hours |
| 3 | `card1` | Certain card identifiers are systematically associated with higher fraud rates |
| 4 | `C13` | Historical count of related transactions; high counts can signal automated fraud |
| 5 | `C1` | Transaction frequency over a recent window; activity spikes indicate suspicious behavior |
| 6 | `addr1` | Billing region; mismatches between location and card behavior are red flags |
| 7 | `D15` | Time since last related transaction; unusual gaps or rapid sequences signal fraud |

### Local — SHAP Values

**SHAP Summary Plot:** Shows the distribution of SHAP values across 200 test transactions. Features at the top have the largest average impact. Red = high feature value, Blue = low feature value. Reveals not just *which* features matter but *how* their values push predictions toward fraud or non-fraud.

**SHAP Waterfall Plot:** Explains a single transaction step-by-step. Starting from the model's base rate, each feature either increases or decreases the fraud probability — the type of explanation a fraud analyst would review before flagging a transaction.

**Key SHAP Findings:**
- High `TransactionAmt` values push predictions strongly toward fraud
- Low `C13` values (few prior transactions) are associated with fraud — new accounts are riskier
- Matching email domains reduce fraud probability; mismatches increase it
- The engineered `log_TransactionAmt` feature showed meaningful SHAP contribution

---

## 10. Key Insights

**XGBoost outperformed all other models** because gradient boosting iteratively minimizes residual errors, making it highly effective at finding subtle fraud patterns in tabular data.

**Threshold tuning was the single biggest practical improvement.** The default 0.50 threshold left recall at ~0.44. Lowering to ~0.30 raised recall to 0.80+ — a 36-point improvement — at the cost of some precision. For fraud detection, this trade-off is clearly worth making.

**Feature engineering added measurable signal.** `log_TransactionAmt`, `email_domain_match`, and `missing_count` all appeared in SHAP analysis as meaningful contributors.

**SMOTE and ADASYN improved recall by ~25–27 points** compared to baseline. ADASYN had a slight edge because it focuses synthetic generation on harder-to-classify boundary cases.

### Business Impact
A model achieving 0.80+ recall means 8 out of every 10 fraudulent transactions are flagged for review — compared to fewer than 5 out of 10 with the default threshold. At scale across hundreds of thousands of daily transactions, this difference translates to millions of dollars in prevented losses. SHAP explanations also make the model defensible to regulators and compliance teams who require explainable automated financial decisions.

---

## 11. Conclusion

This project built a complete, production-aware fraud detection pipeline on the IEEE-CIS dataset. Starting from raw, messy tabular data with severe class imbalance and over 400 features, the final XGBoost model achieved a ROC-AUC of 0.936 with recall tuned above 0.80.

Key contributions:
- Systematic comparison of 6 model families
- Evaluation of 3 imbalance handling techniques on the top 2 models
- Deliberate threshold tuning with a target recall objective
- 5-fold stratified cross-validation for reliable generalization estimates
- Full XAI implementation with global and local explanations
- Practical feature engineering grounded in fraud domain knowledge

---

## 12. Future Work

| Improvement | Description |
|---|---|
| Hyperparameter Tuning | Apply `Optuna` or `GridSearchCV` to systematically search XGBoost parameters |
| Fix Data Leakage | Use `sklearn.Pipeline` to ensure all preprocessing is fit only on training data |
| Consistent Evaluation Set | Evaluate all 6 models on the same test set for fair comparison |
| Neural Network Training | Increase `max_iter` well beyond 20 so the MLP actually converges |
| Real-Time Deployment | Package model into a REST API (FastAPI or Flask) for live transaction scoring |
| Concept Drift Monitoring | Implement model monitoring to detect performance degradation over time |
| Cost-Sensitive Learning | Optimize threshold using explicit dollar costs for false negatives vs false positives |

---

## 13. How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/Suav00/IEE-CIS-Fraud-Detection.git
cd IEE-CIS-Fraud-Detection
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Download the data**
1. Go to https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place both files in the `data/` folder

**Step 4 — Run the notebook**
```bash
jupyter notebook notebooks/fraud_detection_capstone.ipynb
```
Run all cells top to bottom. The notebook will load data, preprocess, train all models, apply imbalance techniques, run XAI, and save the final model.

**Step 5 — Load the saved model**
```python
import joblib
model = joblib.load("models/xgb_fraud_model.pkl")
features = joblib.load("models/model_features.pkl")
```
