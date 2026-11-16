# diabetes-prediction-model

# Diabetes Prediction Model & Deployment

A machine learning web service that predicts the likelihood of diabetes based on patient health features, built with **XGBoost**, **FastAPI**, and containerized using **Docker**.

## Project Overview
This project covers the full end-to-end MLOps pipeline:
1. **Training (`train.py`)**: Trains an XGBoost classifier on health data and saves the artifact.
2. **Serving (`predict.py`)**: Exposes a FastAPI prediction endpoint using Pydantic validation.
3. **Containerization (`Dockerfile`)**: Packages the application into a lightweight Docker container for reproducible deployments.

---

## How to Run Locally

1. **Install dependencies:**
bash
> pip install -r requirements.txt

2. **Run the FastAPI service:**
bash
> uvicorn predict:app --reload --port 9696

3. **Test the endpoint (in a separate terminal):**
bash
> python test.py

---

## How to Run with Docker

1. **Build the Docker image:**
bash
> docker build -t diabetes-prediction-service .

2. **Run the container:**
bash
> docker run --rm -p 9696:9696 diabetes-prediction-service

3. **Test the containerized service:**
bash
> python test.py

---

## Testing with Custom Parameters

If you want to test different patient profiles or edge cases, you can edit the sample payload dictionary directly inside `test.py`. 

Open `test.py` and modify the dictionary values under the request body:

```python
# test.py snippet
patient_data = {
    "gender": "F",                   # Options: "M", "F" (str)
    "age": 45.0,                     # Numerical age in years (float)
    "hypertension": 0,               # 0 for No, 1 for Yes (int)
    "heart_disease": 0,              # 0 for No, 1 for Yes (int)
    "smoking_history": "no",         # Options: "yes", "no", "no info" (str)
    "bmi": 27.5,                     # Body Mass Index (float)
    "hba1c_level": 6.2,              # HbA1c blood sugar level (%) (float)
    "blood_glucose_level": 140       # Blood glucose level (mg/dl) (int)
}
```
Refer to the above instructions on how to run locally or with Docker.
