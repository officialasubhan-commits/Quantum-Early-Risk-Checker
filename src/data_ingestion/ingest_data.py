import os
import json
import requests
import pandas as pd
import numpy as np

# Dataset Metadata
DATASET_METADATA = {
    "name": "CDC Diabetes Health Indicators (BRFSS 2015)",
    "source": "Centers for Disease Control and Prevention (CDC) - Behavioral Risk Factor Surveillance System (BRFSS 2015)",
    "official_url": "https://www.cdc.gov/brfss/annual_data/annual_2015.html",
    "license": "Public Domain / Open Government Data (CC0 1.0 Universal)",
    "target_variable": "Diabetes_binary",
    "target_mapping": {0.0: "No Diabetes", 1.0: "Prediabetes or Diabetes Risk"},
    "clinical_validation_note": "Dataset originates from official CDC national health surveillance surveys. Contains self-reported survey responses and health indicators, not direct laboratory biomarker measurements.",
    "urls": [
        "https://raw.githubusercontent.com/nabilshajahan3110/CDC-Diabetic-Health-Indicators/main/diabetes_binary_health_indicators_BRFSS2015.csv"
    ]
}

# Expected column range boundaries based on BRFSS 2015 Codebook
VALID_COLUMN_RANGES = {
    "Diabetes_binary": (0, 1),
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (12, 98),
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8)
}

def fetch_raw_dataset(raw_dir: str = "data/raw") -> str:
    """
    Downloads or confirms local presence of the CDC BRFSS 2015 Diabetes Health Indicators dataset.
    """
    os.makedirs(raw_dir, exist_ok=True)
    target_path = os.path.join(raw_dir, "cdc_diabetes_health_indicators.csv")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        print(f"[DATA INGESTION] Dataset exists locally at: {target_path} ({os.path.getsize(target_path):,} bytes)")
        return target_path

    print("[DATA INGESTION] Downloading dataset from official public mirror...")
    url = DATASET_METADATA["urls"][0]
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"[DATA INGESTION] Download successful! Saved to {target_path}")
        else:
            raise RuntimeError(f"HTTP Status {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset from {url}: {e}")
        
    return target_path

def load_and_validate_data(file_path: str, report_dir: str = "reports") -> pd.DataFrame:
    """
    Loads raw CSV data and performs rigorous validation checks:
    - Row count
    - Column count
    - Column names
    - Data types
    - Missing values
    - Duplicate records
    - Target distribution
    - Invalid/out-of-range values
    - Class imbalance
    - Source & License metadata
    Saves a dataset ingestion report to `reports/dataset_ingestion_report.json`.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
    df = pd.read_csv(file_path)
    
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
    
    target_col = DATASET_METADATA["target_variable"]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns: {cols}")
        
    missing_per_col = df.isnull().sum().to_dict()
    total_missing = int(sum(missing_per_col.values()))
    
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = float(duplicate_count / n_rows)
    
    # Class distribution & imbalance
    target_counts = df[target_col].value_counts().to_dict()
    target_props = df[target_col].value_counts(normalize=True).to_dict()
    
    # Class imbalance ratio (majority : minority)
    max_cls_count = max(target_counts.values())
    min_cls_count = min(target_counts.values())
    imbalance_ratio = float(max_cls_count / min_cls_count) if min_cls_count > 0 else float('inf')
    
    # Range / Out-of-bounds checks
    out_of_range_report = {}
    total_out_of_range = 0
    for col, (min_val, max_val) in VALID_COLUMN_RANGES.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_cnt = int(invalid_mask.sum())
            if invalid_cnt > 0:
                out_of_range_report[col] = {
                    "invalid_count": invalid_cnt,
                    "expected_range": [min_val, max_val],
                    "actual_min": float(df[col].min()),
                    "actual_max": float(df[col].max())
                }
                total_out_of_range += invalid_cnt

    report = {
        "metadata": DATASET_METADATA,
        "validation_metrics": {
            "num_rows": n_rows,
            "num_cols": n_cols,
            "num_features": n_cols - 1,
            "column_names": cols,
            "data_types": dtypes,
            "total_missing_values": total_missing,
            "missing_values_per_column": missing_per_col,
            "duplicate_records": duplicate_count,
            "duplicate_percentage": round(duplicate_pct * 100, 2),
            "target_variable": target_col,
            "target_class_counts": {str(k): int(v) for k, v in target_counts.items()},
            "target_class_proportions": {str(k): round(float(v) * 100, 2) for k, v in target_props.items()},
            "class_imbalance_ratio": round(imbalance_ratio, 2),
            "out_of_range_violations": out_of_range_report,
            "total_out_of_range_records": total_out_of_range,
            "validation_status": "PASSED" if total_missing == 0 and total_out_of_range == 0 else "WARNINGS_FOUND"
        }
    }
    
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "dataset_ingestion_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n" + "=" * 60, flush=True)
    print(" DATASET INGESTION & VALIDATION REPORT ", flush=True)
    print("=" * 60, flush=True)
    print(f"Dataset Name:          {DATASET_METADATA['name']}", flush=True)
    print(f"Official Source:       {DATASET_METADATA['source']}", flush=True)
    print(f"License:               {DATASET_METADATA['license']}", flush=True)
    print(f"Total Records (Rows):  {n_rows:,}", flush=True)
    print(f"Total Features (Cols): {n_cols - 1} (+1 Target: {target_col})", flush=True)
    print(f"Missing Values:        {total_missing}", flush=True)
    print(f"Duplicate Rows:        {duplicate_count:,} ({duplicate_pct:.2%})", flush=True)
    print(f"Class Distribution:    {target_counts}", flush=True)
    print(f"Class Proportions:     {report['validation_metrics']['target_class_proportions']}", flush=True)
    print(f"Imbalance Ratio:       {imbalance_ratio:.2f}:1", flush=True)
    print(f"Out-of-range Values:   {total_out_of_range}", flush=True)
    print(f"Validation Status:     {report['validation_metrics']['validation_status']}", flush=True)
    print(f"Detailed JSON Report:  {report_path}", flush=True)
    print("=" * 60, flush=True)
    
    return df

