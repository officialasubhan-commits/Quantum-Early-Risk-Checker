import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "heart_disease")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "heart_disease")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "heart_disease")

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

def ingest_heart_disease_data():
    print("=" * 70, flush=True)
    print(" [DATA INGESTION] Ingesting UCI Cleveland Heart Disease Dataset... ", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    raw_csv_path = os.path.join(RAW_DIR, "heart_disease_cleveland.csv")

    if not os.path.exists(raw_csv_path):
        print(f" -> Downloading dataset from UCI Repository: {UCI_URL}", flush=True)
        df = pd.read_csv(UCI_URL, header=None, names=COLUMN_NAMES, na_values="?")
        df.to_csv(raw_csv_path, index=False)
    else:
        print(f" -> Loading local raw dataset from: {raw_csv_path}", flush=True)
        df = pd.read_csv(raw_csv_path, na_values="?")

    total_records = len(df)
    print(f" -> Total Records Ingested: {total_records}", flush=True)

    # Missing Value Handling
    missing_counts = df.isnull().sum().to_dict()
    print(f" -> Missing Values Detected: {missing_counts}", flush=True)

    for col in ["ca", "thal"]:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"    - Imputed missing values in '{col}' with median: {median_val}", flush=True)

    # Binary Target Mapping (0 = Absence, 1 = Presence of Heart Disease)
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns=["num"], inplace=True)

    class_dist = df["target"].value_counts().to_dict()
    print(f" -> Binary Target Class Distribution: Absence (0): {class_dist.get(0, 0)}, Presence (1): {class_dist.get(1, 0)}", flush=True)

    # Stratified Train (70%), Validation (15%), Test (15%) Splits
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["target"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["target"])

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f" -> Saved Data Splits:")
    print(f"    - Train Set:      {len(train_df)} samples ({train_path})")
    print(f"    - Validation Set: {len(val_df)} samples ({val_path})")
    print(f"    - Test Set:       {len(test_df)} samples ({test_path})")

    report = {
        "disease_id": "heart_disease",
        "disease_name": "Cardiovascular / Heart Disease",
        "dataset_name": "UCI Heart Disease Cleveland Database",
        "source_url": UCI_URL,
        "total_records": total_records,
        "features_count": 13,
        "features_list": [col for col in df.columns if col != "target"],
        "class_distribution": class_dist,
        "missing_values_imputed": missing_counts,
        "splits": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df)
        }
    }

    report_path = os.path.join(REPORTS_DIR, "dataset_ingestion_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f" -> Saved Dataset Ingestion Report to: {report_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    ingest_heart_disease_data()
