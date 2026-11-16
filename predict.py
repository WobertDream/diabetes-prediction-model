import pickle
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_FILE = 'model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Diabetes Prediction Service (Random Forest)")

class PatientRecord(BaseModel):
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

    # Scikit-learn outputs probabilities for [class_0, class_1]
    pred_prob = float(model.predict_proba(X)[0, 1])
    prediction = bool(pred_prob >= 0.50)
    
    return {
        "diabetes_probability": pred_prob,
        "diabetes_prediction": prediction
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
