import requests

# The URL of your local FastAPI endpoint
url = 'http://localhost:9696/predict'


patient = {
    "gender": "M",
    "age": 45.0,
    "hypertension": 1,
    "heart_disease": 1,
    "smoking_history": "never",
    "bmi": 28.3,
    "hba1c_level": 3.7,
    "blood_glucose_level": 150,
}

response = requests.post(url, json=patient)

print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(response.json())
