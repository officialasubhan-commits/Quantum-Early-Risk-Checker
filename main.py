import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data_ingestion.ingest_data import fetch_raw_dataset, load_and_validate_data, DATASET_METADATA
from src.preprocessing.pipeline import preprocess_and_split
from src.classical_ml.models import get_classical_models, train_and_cross_validate
from src.evaluation.metrics import evaluate_model_performance

def main():
    print("=" * 70, flush=True)
    print(" Hybrid Quantum Machine Learning Platform for Early Disease Detection ", flush=True)
    print(" Phase 1: Classical Machine Learning Model Pipeline Execution ", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Directories Setup
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 2. Data Ingestion & Validation
    raw_data_path = fetch_raw_dataset(raw_dir="data/raw")
    raw_df = load_and_validate_data(raw_data_path)
    
    target_col = DATASET_METADATA["target_variable"]
    
    # 3. Data Pipeline & Preprocessing (Stratified Split, Fit Scaler strictly on Train)
    data_dict = preprocess_and_split(
        df=raw_df,
        target_col=target_col,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val     = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test   = data_dict["X_test"], data_dict["y_test"]
    preprocessor     = data_dict["preprocessor"]
    feature_names    = data_dict["feature_names"]
    
    # 4. Classical ML Models Instantiation & 5-Fold Cross Validation
    models = get_classical_models(random_state=42)
    cv_results = train_and_cross_validate(models, X_train, y_train, cv_splits=5)
    
    # 5. Model Evaluation on Held-Out Test Set
    metrics_summary = {}
    trained_models = {}
    
    best_model_name = None
    best_f1_score = -1.0
    best_model_obj = None
    
    for model_name, model in models.items():
        metrics, trained_model = evaluate_model_performance(
            model=model,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        # Attach cross-validation stats
        metrics["cv_mean_f1_macro"] = cv_results[model_name]["mean_f1_macro"]
        metrics["cv_std_f1_macro"] = cv_results[model_name]["std_f1_macro"]
        
        metrics_summary[model_name] = metrics
        trained_models[model_name] = trained_model
        
        if metrics["f1_score"] > best_f1_score:
            best_f1_score = metrics["f1_score"]
            best_model_name = model_name
            best_model_obj = trained_model

    # 6. Save Artifacts into models/ and reports/
    print("\n" + "=" * 70, flush=True)
    print(" Saving Trained Artifacts & Evaluation Reports ", flush=True)
    print("=" * 70, flush=True)
    
    best_model_path = os.path.join("models", "best_classical_model.joblib")
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    
    joblib.dump(best_model_obj, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f" -> Best Model Saved ({best_model_name}): {best_model_path}", flush=True)
    print(f" -> Preprocessing Pipeline Saved:           {preprocessor_path}", flush=True)
    
    # Save metrics report JSON
    report_json_path = os.path.join("reports", "experiment_metrics.json")
    with open(report_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # Save CSV metrics summary table
    summary_rows = []
    for m_name, m_data in metrics_summary.items():
        summary_rows.append({
            "Model": m_name,
            "Accuracy": f"{m_data['accuracy']:.4f}",
            "Precision": f"{m_data['precision']:.4f}",
            "Recall/Sensitivity": f"{m_data['recall_sensitivity']:.4f}",
            "Specificity": f"{m_data['specificity']:.4f}",
            "F1-Score": f"{m_data['f1_score']:.4f}",
            "ROC-AUC": f"{m_data['roc_auc']:.4f}",
            "Training Time (s)": f"{m_data['training_time_sec']:.3f}",
            "Inference Time (ms/sample)": f"{m_data['inference_ms_per_sample']:.4f}"
        })
    summary_df = pd.DataFrame(summary_rows)
    report_csv_path = os.path.join("reports", "metrics_summary.csv")
    summary_df.to_csv(report_csv_path, index=False)
    
    print(f" -> Detailed Metrics JSON Saved: {report_json_path}", flush=True)
    print(f" -> Summary CSV Saved:          {report_csv_path}", flush=True)
    
    # 7. Print Final Summary Table
    print("\n" + "=" * 70, flush=True)
    print(" Final Model Comparison Summary Table ", flush=True)
    print("=" * 70, flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
