# Fraudulent Transaction Detector

A machine learning (ML) project for detecting fraudulent cross-border payment transactions using the Nova Pay dataset.

## Project Overview

This project explores fraud detection on a real-world payments dataset. It includes:

- Data cleaning and exploratory data analysis (EDA)
- A logistic regression baseline model
- A LightGBM model for improved fraud detection
- Time-based train/test splitting to avoid data leakage

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

## Results

| Model               | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|---------------------|----------|-----------------|--------------|----------|
| Logistic Regression | 0.93     | 0.67            | 0.89         | 0.76     |
| LightGBM            | 0.97     | 0.86            | 0.88         | 0.86     |

LightGBM outperform the logistic regression baseline, improving fraud precision from 0.67 to 0.86 and fraud F1 from 0.76 to 0.80. It better captures non-linear relationships between features such as transaction velocity, device trust, and KYC tier.

## How to Run

```bash
git clone https://github.com/PleromaRay/fraudulent_transaction_detector.git
cd fraudulent_transaction_detector
pip install -r requirements.txt