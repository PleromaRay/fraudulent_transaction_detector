# fraudulent_transaction_detector
# Nova Pay Transaction Analysis

Exploratory Data Analysis (EDA) and data cleaning pipeline for the Nova Pay transaction dataset.  
This project focuses on understanding the data, fixing quality issues, and preparing it for further analysis and modeling (e.g., fraud detection).

## Dataset Overview

The dataset (`nova_pay_combined.csv`) contained ~11,400 cross-border payment transactions with the following key columns:

- Transaction identifiers and timestamps
- Home country, source/destination currencies, and payment channel (ATM, mobile, web)
- Amounts (source and USD), fees, and exchange rates
- Device, IP, and location-related features
- Risk indicators (IP risk, device trust, velocity, corridor risk, KYC tier)
- Target: `is_fraud` (binary)

**Known data quality issues addressed:**
- Inconsistent categorical values (whitespace, case, typos)
- Invalid timestamps
- Negative sentinel values in `fee` and `device_trust_score`
- Missing values across multiple columns

## Key Findings from EDA

**Fraud rate by channel**
- Web has the highest fraud rate
- Followed by ATM
- Then Mobile
- Unknown has the lowest fraud rate

**Fraud rate by KYC tier**
- Accounts with the lowest KYC tier have the highest fraud rates

**Correlation with `is_fraud`**

| Feature                | Correlation |
|------------------------|-------------|
| txn_velocity_24h       | 0.752       |
| txn_velocity_1h        | 0.688       |
| risk_score_internal    | 0.602       |
| ip_risk_score          | 0.520       |
| corridor_risk          | 0.194       |
| amount_usd             | 0.130       |
| fee                    | -0.014      |
| account_age_days       | -0.285      |
| device_trust_score     | -0.437      |

Transaction velocity and internal risk score show the strongest positive relationships with fraud, while higher device trust score and older account age are associated with lower fraud rates.

