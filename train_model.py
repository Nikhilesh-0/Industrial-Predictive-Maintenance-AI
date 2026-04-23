"""
Run this once before starting the app to train and save the model.
Usage: python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

CSV_PATH = "ai4i2020.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{CSV_PATH}'. "
        "Please place ai4i2020.csv in the same directory."
    )

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

X = df.drop(columns=['UDI', 'Product ID', 'Type', 'Machine failure',
                      'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
                      'Process temperature [K]'])
y = df['Machine failure']

print(f"Features used: {X.columns.tolist()}")
print(f"Class distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)

joblib.dump(rf, "rf_model.pkl")
print("\n✅ Model saved to rf_model.pkl")

from sklearn.metrics import classification_report
y_pred = rf.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))
