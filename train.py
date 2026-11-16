import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

DATA_FILE = 'diabetes-dataset-2k.csv'
OUTPUT_FILE = 'model.bin'
RANDOM_SEED = 1101

print("1. Loading and preparing data...")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"[Error]: The dataset file '{DATA_FILE}' was not found.")
    print("Please make sure the CSV file is placed in the same folder as this script.")
    sys.exit(1)
except Exception as e:
    print(f"[Error]: An unexpected error occurred while reading '{DATA_FILE}': {e}")
    sys.exit(1)

df['smoking_history'] = df['smoking_history'].mask((df['smoking_history'] == 'current') |
                                                   (df['smoking_history'] == 'ever') |
                                                   (df['smoking_history'] == 'former'), 'yes')
df['smoking_history'] = df['smoking_history'].mask((df['smoking_history'] == 'never') |
                                                   (df['smoking_history'] == 'not current'), 'no')
df.columns = df.columns.str.lower()
response = 'diabetes'
numerical = ['age', 'bmi', 'blood_glucose_level', 'hba1c_level']
categorical = ['hypertension', 'heart_disease', 'smoking_history', 'gender']

if response not in df.columns:
    print(f"[Error]: Target column '{response}' not found in the dataset columns: {list(df.columns)}")
    sys.exit(1)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df[response])

print(f"Full training set shape: {df_full_train.shape}")
print(f"Test set shape: {df_test.shape}")

print("2. Fitting DictVectorizer...")

dv = DictVectorizer(sparse=False)

full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dicts)
y_full_train = df_full_train[response].values

print("3. Training the final Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=3,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
model.fit(X_full_train, y_full_train)

print("4. Saving the model and vectorizer...")

try:
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model successfully saved to {OUTPUT_FILE}!")
except IOError as e:
    print(f"[Error]: Failed to save the model to '{OUTPUT_FILE}': {e}")
    sys.exit(1)
except Exception as e:
    print(f"[Error]: An unexpected error occurred while saving the model: {e}")
    sys.exit(1)
