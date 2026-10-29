# Fraudulent Transaction Detector

A machine learning (ML) project for detecting fraudulent cross-border payment transactions using the Nova Pay dataset.

## Project Overview

This project explores fraud detection on a real-world payments dataset. It includes:

- Data cleaning and exploratory data analysis (EDA)
- A logistic regression baseline model
- LightGBM model with hyperparameter tuning
- SHAP model explainability
- FastAPI endpoint for real-time predictions

## Dataset

The dataset (`nova_pay_combined.csv`) contains ~11,400 cross-border payment transactions with features such as:

- Transaction amount, fees, and exchange rates
- Payment channel (ATM, Mobile, Web)
- Device trust score, IP risk score, and location data
- Transaction velocity (1h and 24h)
- KYC tier and account age
- **Target**: `is_fraud`

### Key EDA Findings

| Feature              | Correlation with Fraud |
|----------------------|------------------------|
| txn_velocity_24h     | 0.752                  |
| txn_velocity_1h      | 0.688                  |
| risk_score_internal  | 0.602                  |
| ip_risk_score        | 0.520                  |
| device_trust_score   | -0.437                 |
| account_age_days     | -0.285                 |

- Web transactions had the highest fraud rate.
- Lower KYC tiers were associated with higher fraud.
- High transaction velocity and internal risk scores were the strongest fraud signals.

## Modeling Approach

1. **Data cleaning**: Handled inconsistent categories, missing values, invalid timestamps, and sentinel values.
2. **Time-based split**: 80/20 train/test split ordered by timestamp (`shuffle=False`) to simulate real-world deployment.
3. **Preprocessing**: StandardScaler for numerical features + OneHotEncoder for categorical features.
4. **Models**:
   - Logistic Regression with `class_weight='balanced'`
   - LightGBM with `class_weight='balanced'`
   - LightGBM with hyperparameter tuning via RandomizedSearchCV

### Best LightGBM Parameters (after tuning)
- `n_estimators`: 300
- `learning_rate`: 0.1
- `max_depth`: -1
- `num_leaves`: 20
- `min_child_samples`: 30
- `class_weight`: balanced

## Results

| Model               | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|---------------------|----------|-----------------|--------------|----------|
| Logistic Regression | 0.93     | 0.67            | 0.89         | 0.76     |
| LightGBM            | 0.98     | 0.92            | 0.88         | 0.90     |
| LightGBM (tuned)    | 0.98     | 0.96            | 0.88         | 0.92     |

LightGBM outperform the logistic regression baseline, improving fraud precision from 0.67 to 0.86 and fraud F1 from 0.76 to 0.86. It better captures non-linear relationships between features such as transaction velocity, device trust, and KYC tier. Hyperparameter tuning improved fraud precision from 0.92 to 0.96 and fraud F1 from 0.90 to 0.92 while maintaining the same recall.

## Model Explainability

SHAP was used to interpret the tuned LightGBM model and identify the strongest drivers of fraud predictions.

## API (Inference)

A FastAPI application is included for serving predictions.

### Run the API
```bash
uvicorn app:app --reload
Then open: http://127.0.0.1:8000/docs

### Example Request
{
  "home_country": "GB",
  "source_currency": "GBP",
  "dest_currency": "USD",
  "channel": "web",
  "amount_src": 150.0,
  "amount_usd": 190.0,
  "fee": 2.5,
  "exchange_rate_src_to_dest": 1.27,
  "new_device": 0,
  "ip_country": "GB",
  "ip_county": "London",
  "location_mismatch": 0,
  "ip_risk_score": 0.2,
  "kyc_tier": "tier_2",
  "account_age_days": 400,
  "device_trust_score": 0.8,
  "chargeback_history_count": 0,
  "risk_score_internal": 0.3,
  "txn_velocity_1h": 1,
  "txn_velocity_24h": 3,
  "corridor_risk": 0.1
}

## How to Run 

```bash
git clone https://github.com/PleromaRay/fraudulent_transaction_detector.git
cd fraudulent_transaction_detector
pip install -r requirements.txt
   - Notebooks are in the EDA/ folder.
   - To serve the model: uvicorn app:app --reload

## Future Improvements
   - Docker containerization