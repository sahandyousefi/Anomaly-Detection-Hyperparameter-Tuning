# Financial Fraud Detection Using XGBoost and Hyperparameter Tuning

A machine learning project applying XGBoost classification to detect fraudulent financial transactions on a large-scale payments dataset. The pipeline covers data preprocessing, class imbalance handling via undersampling, domain-driven feature engineering, model training with threshold optimization, hyperparameter tuning, and model interpretability using SHAP values.

---

## Business Objective

Financial fraud is a growing concern in the banking and payments industry. Traditional rule-based fraud detection systems fail to scale effectively, producing high false positive rates and missing a significant share of fraudulent activity.

The goal of this project is to develop an automated, machine learning-based fraud detection system that:

- Accurately detects fraudulent transactions in real time
- Minimizes financial losses by reducing missed fraud cases (False Negatives)
- Reduces unnecessary transaction blocks by controlling false positives

A strong fraud detection system is not just about catching fraud — it is about minimizing risk while maintaining a seamless experience for legitimate users.

---

## Dataset

The dataset used is the **PaySim synthetic financial transaction dataset** (`PS_20174392719_1491204439457_log.csv`), sourced from Kaggle. PaySim simulates mobile money transactions based on real transaction logs from a financial company.

| Feature | Description |
|---|---|
| `step` | Unit of time (1 step = 1 hour); covers 30 days of transactions |
| `type` | Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount in local currency |
| `nameOrig` | Originating customer ID |
| `oldbalanceOrg` | Originator balance before transaction (dropped) |
| `newbalanceOrig` | Originator balance after transaction (dropped) |
| `nameDest` | Destination customer or merchant ID |
| `oldbalanceDest` | Destination balance before transaction (dropped) |
| `newbalanceDest` | Destination balance after transaction (dropped) |
| `isFraud` | Target variable: 1 = Fraudulent, 0 = Legitimate |
| `isFlaggedFraud` | System flag for large transfers (dropped after analysis) |

Balance columns were dropped due to high collinearity and noise. `isFlaggedFraud` was dropped after correlation analysis revealed near-zero predictive value (correlation = 0.044 with `isFraud`).

---

## Methodology

### 1. Data Preprocessing

- Verified null values and duplicates — dataset was clean with no missing entries
- Dropped `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` as they added noise without improving predictive power
- Applied Label Encoding to the `type` column (categorical to numeric)
- Optimized data types to reduce memory consumption:
  - `step`, converted to `int32`
  - `isFraud`, `isFlaggedFraud` converted to `int8`
  - `amount` converted to `float32`
- Memory usage was measurably reduced after type optimization

**Class Imbalance Handling**

The dataset is heavily imbalanced — legitimate transactions vastly outnumber fraudulent ones. Random undersampling was applied with `sampling_strategy=0.1`, resulting in a 10:1 ratio of non-fraud to fraud cases, making the training set manageable and more balanced for the classifier.

---

### 2. Feature Engineering

The following domain-driven features were engineered to improve fraud signal detection:

| Feature | Description |
|---|---|
| `high_amount` | Binary flag: 1 if transaction amount exceeds 200,000 |
| `orig_avg_amount` | Mean transaction amount per originating account |
| `suspicious_amount` | Binary flag: 1 if amount exceeds 5x the originator's historical average |
| `orig_tx_count` | Total number of transactions per originating account |
| `dest_tx_count` | Total number of transactions per destination account |
| `orig_tx_std` | Standard deviation of amounts sent by the originating account |
| `dest_tx_std` | Standard deviation of amounts received by the destination account |
| `is_transfer` | Binary flag: 1 if transaction type is TRANSFER |
| `is_cashout` | Binary flag: 1 if transaction type is CASH_OUT |
| `hour_of_day` | Hour extracted from `step` (step % 24) |
| `is_merchant` | Binary flag: 1 if destination ID starts with 'M' (merchant) |

`isFlaggedFraud` was dropped after analysis showed that the vast majority of real fraud cases (thousands of transactions) were never flagged by the legacy rule-based system, confirming its lack of predictive utility.

---

### 3. Model Development

**Algorithm: XGBoost Classifier**

XGBoost (Extreme Gradient Boosting) was selected for its strong performance on tabular, imbalanced classification problems, native handling of sparse features, and built-in feature importance ranking.

**Initial Model Configuration:**

| Parameter | Value |
|---|---|
| `n_estimators` | 750 |
| `learning_rate` | 0.3 |
| `max_depth` | 7 |
| `random_state` | 42 |

**Train-Test Split:** 80% training / 20% test, stratified on the target variable.

**Decision Threshold Adjustment**

Rather than using the default 0.5 probability threshold, the classification threshold was lowered to **0.15** to prioritize recall — ensuring more fraudulent transactions are caught at the cost of a controlled increase in false positives. This is the correct trade-off in a fraud detection context.

---

### 4. Hyperparameter Tuning

Feature importance scores were plotted using XGBoost's built-in `plot_importance` function to identify and remove low-signal features. `isFlaggedFraud` was confirmed as the lowest-importance feature and removed before the final model was retrained.

The final model was retrained after feature pruning and threshold adjustment, yielding the performance metrics documented below.

---

### 5. Final Model Performance

**Accuracy: 94.32%**

The model correctly classifies 94.32% of all transactions. However, in fraud detection, accuracy alone is insufficient due to class imbalance — recall and precision on the fraud class are the primary evaluation criteria.

**Confusion Matrix:**

| | Predicted Non-Fraud | Predicted Fraud |
|---|---|---|
| Actual Non-Fraud | 15,636 (True Negative) | 790 (False Positive) |
| Actual Fraud | 236 (False Negative) | 1,407 (True Positive) |

**Classification Report:**

| Metric | Non-Fraud (0) | Fraud (1) | Macro Average |
|---|---|---|---|
| Precision | 0.99 | 0.64 | 0.81 |
| Recall | 0.95 | 0.86 | 0.90 |
| F1-Score | 0.97 | 0.73 | 0.85 |

**Why Recall is the Priority Metric**

In fraud detection, False Negatives (missed fraud) are far more costly than False Positives (legitimate transactions incorrectly flagged):

- A falsely flagged user can verify their identity and resume transactions with minimal disruption
- A missed fraud case allows financial loss to occur undetected and may enable repeated abuse

Achieving **Recall of 86%** on the fraud class means the model successfully identifies 86 out of every 100 real fraud cases — a strong result for a financial fraud detection system operating on highly imbalanced real-world data.

---

### 6. Model Interpretability — SHAP Analysis

SHAP (SHapley Additive exPlanations) values were computed using the `shap` library to explain individual and aggregate model predictions.

**Key findings from the SHAP summary plot:**

- `amount` is the single most influential feature. High transaction amounts (red) strongly increase the predicted fraud probability, while small amounts (blue) reduce it.
- `step` (time of transaction) has a mixed but notable impact, suggesting that fraud tends to cluster around specific time windows — consistent with organized fraud behavior.
- `type` (transaction type) contributes meaningfully, with TRANSFER and CASH_OUT transaction types associated with elevated fraud risk.

SHAP analysis confirms that the model's decisions are grounded in economically interpretable signals rather than spurious correlations — an important property for deployment in a regulated financial environment.

---

## Key Findings

- The legacy `isFlaggedFraud` system flag had near-zero correlation (0.044) with actual fraud, and thousands of real fraud cases were never flagged by the rule-based system — highlighting the inadequacy of traditional approaches.
- Custom-engineered features, particularly `suspicious_amount` (5x a user's historical average) and transaction frequency counts, meaningfully improved the model's ability to detect anomalous behavior.
- Lowering the decision threshold from 0.50 to 0.15 significantly improved recall from a lower baseline to 86%, at the cost of an acceptable increase in false positives.
- SHAP analysis confirmed that transaction amount, timing, and type are the dominant drivers of fraud predictions, aligning with domain knowledge in financial crime detection.
- The final model achieves 94.32% accuracy with 86% fraud recall — suitable for deployment as a first-pass automated fraud screening system, with human review applied to flagged cases.

---

## Project Structure

    Anomaly-Detection-Hyperparameter-Tuning/
    |-- anomaly-detection-hyper-parameter-tuning.ipynb    # Full pipeline: preprocessing, feature engineering, modeling, SHAP
    |-- README.md

---

## Technologies Used

- Python 3
- Pandas, NumPy — data manipulation, type optimization, feature engineering
- Matplotlib, Seaborn — class distribution plots, boxplots, performance visualization
- Scikit-learn — train-test split, Label Encoding, RandomUnderSampler, classification metrics
- imbalanced-learn — RandomUnderSampler for class imbalance handling
- XGBoost — gradient boosted classifier, feature importance ranking
- SHAP — model interpretability and feature contribution analysis
- Jupyter Notebook — interactive development and documentation
- Kaggle — dataset source (PaySim) and execution environment

---

## Getting Started

1. Clone the repository:

       git clone https://github.com/sahandyousefi/Anomaly-Detection-Hyperparameter-Tuning.git
       cd Anomaly-Detection-Hyperparameter-Tuning

2. Install dependencies:

       pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap jupyter

3. Download the PaySim dataset from Kaggle and place `PS_20174392719_1491204439457_log.csv` in the input directory, or update the file path in the notebook to match your local setup.

4. Launch the notebook:

       jupyter notebook anomaly-detection-hyper-parameter-tuning.ipynb

---

## Author

Sahand Yousefi
[GitHub](https://github.com/sahandyousefi) | [LinkedIn](https://www.linkedin.com/in/sahand-yousefi/)
