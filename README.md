# Diabetes Prediction Model & Deployment

## Introduction

Diabetes is a major public health challenge in first world nations, driven largely by aging populations, sedentary lifestyle and the rise of obesity. Early detection can then enable timely intervention for pre-diabetic individuals, preventing major chronic illnesses and infections from developing. This project leverages machine learning models to predict the likelihood an individual has diabetes given patient health features, built with **XGBoost**, **FastAPI**, and containerized using **Docker**.

---

## Problem Description

Diabetes is a condition that results in blood sugar remaining at high levels in the body for prolonged periods, which can lead to complications such as damaging blood vessels and nerves throughout the body. It is also a prevalent in adults living in developed nations with ranges between 8% and 12%, whereby 90% to 95% are type 2. This project uses the Kaggle Diabetes prediction dataset created by Mohammed Mustafa, containing 2,000 patient records with 8 features and 1 response column.

---

## Dataset details & Model Tuning

The **Diabetes prediction dataset** was sourced from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data)

The data preparation, exploratory data analysis, and model tuning steps are documented in `EDA.ipynb` and `train.py`.

| Field | Description | Type |
| :--- | :--- | :---: |
| **gender** | Gender of the patient (e.g., M, F). | Categorical |
| **age** | Age of the patient in years. | Numerical |
| **hypertension** | Indicates whether the patient has hypertension ( `0`: no, `1`: yes). | Binary |
| **heart_disease** | Indicates whether the patient has heart disease ( `0`: no, `1`: yes). | Binary |
| **smoking_history** | Smoking history category of the patient (yes, no, no info). | Categorical |
| **bmi** | Body Mass Index calculated from weight and height. | Numerical |
| **hba1c_level** | HbA1c blood sugar level of the patient. (%) | Numerical |
| **blood_glucose_level** | Blood glucose level of the patient. (mg/dl) | Numerical |
| **diabetes** | Target variable indicating the presence of diabetes ( `0`: negative, `1`: positive). | Binary |

---

## Project Overview
This project covers the full end-to-end MLOps pipeline:
1. **Training (`train.py`)**: Trains an XGBoost classifier on health data and saves the artifact.
2. **Serving (`predict.py`)**: Exposes a FastAPI prediction endpoint using Pydantic validation.
3. **Containerization (`Dockerfile`)**: Packages the application into a lightweight Docker container for reproducible deployments.

---

## How to Run Locally

1. **Install dependencies:**

bash
```
pip install -r requirements.txt
```

2. **Run the FastAPI service:**

bash
```
uvicorn predict:app --reload --port 9696
```

3. **Test the endpoint (in a separate terminal):**

bash
```
python test.py
```

---

## How to Run with Docker

1. **Build the Docker image:**

bash
```
docker build -t diabetes-prediction-service . 
```

2. **Run the container:**

bash
```
docker run --rm -p 9696:9696 diabetes-prediction-service
```

3. **Test the containerized service:**

bash
```
python test.py
```

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
