from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load teh model
model = joblib.load("fraud_detection_model.joblib")

# Instantiate FastAPI
app = FastAPI(title="Fraud Detection API")

# Validate incoming data
class Transaction(BaseModel):
    home_country: str
    source_currency: str
    dest_currency: str
    channel: str
    amount_src: float
    amount_usd: float
    fee: float
    exchange_rate_src_to_dest: float
    new_device: int
    ip_country: str
    ip_county: str
    location_mismatch: int
    ip_risk_score: float
    kyc_tier: str
    account_age_days: float
    device_trust_score: float
    chargeback_history_count: int
    risk_score_internal: float
    txn_velocity_1h: float
    txn_velocity_24h: float
    corridor_risk: float

@app.post("/predict")
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.dict()])
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }