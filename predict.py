import pickle
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_FILE = 'xgb_model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Diabetes Prediction Service")
class PatientRecord(base_model := BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hba1c_level: float
    blood_glucose_level: float

@app.post("/predict")
def predict(patient: PatientRecord):

    patient_dict = patient.dict()
    X = dv.transform([patient_dict])
    features = list(dv.get_feature_names_out())
    dpred = xgb.DMatrix(X, feature_names=features)

    pred_prob = float(model.predict(dpred)[0])
    prediction = bool(pred_prob >= 0.52)
    
    return {
        "diabetes_probability": pred_prob,
        "diabetes_prediction": prediction
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
