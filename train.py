import sys
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

DATA_FILE = 'diabetes-dataset-2k.csv'
OUTPUT_FILE = 'xgb_model.bin'
RANDOM_SEED = 1101

XGB_PARAMS = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': RANDOM_SEED
}
NUM_BOOST_ROUND = 100



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



print("2. Fitting DictVectorizer and building DMatrices...")

dv = DictVectorizer(sparse=False)

full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dicts)
y_full_train = df_full_train[response].values

test_dicts = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dicts)
y_test = df_test[response].values

features = list(dv.get_feature_names_out())
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)



print("3. Training the final XGBoost model...")

model = xgb.train(
    XGB_PARAMS,
    dfulltrain,
    num_boost_round=NUM_BOOST_ROUND
)



print("4. Saving the model and vectorizer...")

try:
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model successfully saved to {OUTPUT_FILE}!")
except IOError as e:
    print(f"[Error]: Failed to save the model to '{OUTPUT_FILE}'. Check your file write permissions: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[Error]: An unexpected error occurred while saving the model: {e}")
    sys.exit(1)

print(f"Model successfully saved to {OUTPUT_FILE}!")
