import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.explainability.feature_attribution import compute_global_feature_attributions
from src.explainability.patient_explainer import explain_patient_prediction

FEATURE_NAMES = [
    "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
# 21 Predictor features (excluding target Diabetes_binary)
PREDICTOR_NAMES = [f for f in FEATURE_NAMES if f != "Diabetes_binary"]

def run_explainability_pipeline():
    print("=" * 70, flush=True)
    print(" Phase 6: Model Explainability & Feature Attribution Pipeline ", flush=True)
    print("=" * 70, flush=True)
    
    os.makedirs("reports/explainability", exist_ok=True)
    os.makedirs("experiments/explainability", exist_ok=True)
    
    # 1. Load Preprocessed Data & Models
    val_data  = np.load("data/processed/val_data.npz")
    test_data = np.load("data/processed/test_data.npz")
    
    X_val, y_val   = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]
    
    classical_model = joblib.load("models/best_classical_model.joblib")
    pca_reducer     = joblib.load("models/qml_pca_reducer.joblib")
    hybrid_model    = joblib.load("models/hybrid_fusion_model.joblib")
    
    print("\n[EXPLAINABILITY] Computing Global Feature Attributions across 21 Features...", flush=True)
    # 2. Compute Global Feature Attributions
    global_attributions = compute_global_feature_attributions(
        classical_model=classical_model,
        pca_reducer=pca_reducer,
        X_val=X_val,
        y_val=y_val,
        feature_names=PREDICTOR_NAMES
    )
    
    print(" -> Top 5 Global Clinical Risk Drivers:", global_attributions["top_features"], flush=True)
    
    # 3. Individual Patient Prediction Explanations
    print("\n[EXPLAINABILITY] Generating Individual Patient Risk Explanations...", flush=True)
    
    # Sample high risk patient (positive class in test set)
    pos_indices = np.where(y_test == 1)[0]
    high_risk_sample = X_test[pos_indices[0]]
    high_risk_explanation = explain_patient_prediction(
        hybrid_model=hybrid_model,
        sample_vector=high_risk_sample,
        feature_names=PREDICTOR_NAMES,
        patient_id="PATIENT_HIGH_RISK_101"
    )
    
    # Sample low risk patient (negative class in test set)
    neg_indices = np.where(y_test == 0)[0]
    low_risk_sample = X_test[neg_indices[0]]
    low_risk_explanation = explain_patient_prediction(
        hybrid_model=hybrid_model,
        sample_vector=low_risk_sample,
        feature_names=PREDICTOR_NAMES,
        patient_id="PATIENT_LOW_RISK_202"
    )
    
    print(f"\n--- Clinical Narrative for High Risk Patient ---")
    print(high_risk_explanation["clinical_narrative"])
    print(f"\n--- Clinical Narrative for Low Risk Patient ---")
    print(low_risk_explanation["clinical_narrative"])
    
    # 4. Save JSON Reports
    report_payload = {
        "global_feature_attributions": global_attributions,
        "sample_patient_explanations": [
            high_risk_explanation,
            low_risk_explanation
        ]
    }
    
    with open("reports/explainability/explainability_report.json", "w") as f:
        json.dump(report_payload, f, indent=4)
        
    with open("reports/explainability/patient_explanation_sample.json", "w") as f:
        json.dump(high_risk_explanation, f, indent=4)
        
    with open("experiments/explainability/explainability_report.json", "w") as f:
        json.dump(report_payload, f, indent=4)
        
    with open("experiments/explainability/patient_explanation_sample.json", "w") as f:
        json.dump(high_risk_explanation, f, indent=4)

    print("\n -> Saved Explainability JSON Reports to reports/explainability/ & experiments/explainability/", flush=True)
    
    # 5. Generate Technical Plots
    generate_explainability_plots(global_attributions, high_risk_explanation)
    
    return report_payload

def generate_explainability_plots(global_attributions, sample_explanation):
    print("\n[VISUALIZATION] Rendering Explainability Charts...", flush=True)
    
    attributions = global_attributions["feature_attributions"]
    features = [a["feature"] for a in attributions[:10]][::-1] # Top 10 reversed for horizontal plot
    classical_imp = [a["classical_importance"] for a in attributions[:10]][::-1]
    quantum_imp = [a["quantum_importance"] for a in attributions[:10]][::-1]
    hybrid_imp = [a["hybrid_importance"] for a in attributions[:10]][::-1]
    
    # Plot 1: Global Hybrid Feature Importance Bar Chart
    plt.figure(figsize=(9, 5.5))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, hybrid_imp, color="#8b5cf6", edgecolor="#6d28d9", height=0.6)
    plt.yticks(y_pos, features, fontsize=10)
    plt.xlabel("Hybrid Feature Attribution Weight", fontsize=10)
    plt.title("Top 10 Global Disease Risk Drivers (Hybrid Model)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/explainability/global_feature_importance.png", dpi=300)
    plt.savefig("experiments/explainability/global_feature_importance.png", dpi=300)
    plt.close()
    
    # Plot 2: Classical vs Quantum Feature Contribution Comparison
    plt.figure(figsize=(10, 6))
    width = 0.35
    plt.barh(y_pos - width/2, classical_imp, width, label="Classical RF Importance", color="#3b82f6")
    plt.barh(y_pos + width/2, quantum_imp, width, label="Quantum PCA Projection Importance", color="#06b6d4")
    plt.yticks(y_pos, features, fontsize=10)
    plt.xlabel("Feature Contribution Weight", fontsize=10)
    plt.title("Classical vs Quantum Feature Contribution Comparison", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/explainability/classical_vs_quantum_feature_contributions.png", dpi=300)
    plt.savefig("experiments/explainability/classical_vs_quantum_feature_contributions.png", dpi=300)
    plt.close()
    
    # Plot 3: Individual Sample Explanation Bar Chart
    top_contribs = sample_explanation["top_contributing_features"]
    sample_feat_names = [item["display_name"] for item in top_contribs][::-1]
    sample_scores = [item["contribution_score"] for item in top_contribs][::-1]
    
    plt.figure(figsize=(8, 4))
    y_pos_sample = np.arange(len(sample_feat_names))
    plt.barh(y_pos_sample, sample_scores, color="#ec4899", height=0.5)
    plt.yticks(y_pos_sample, sample_feat_names, fontsize=9)
    plt.xlabel("Patient Local Feature Risk Contribution", fontsize=10)
    plt.title(f"Patient Risk Factor Attribution ({sample_explanation['patient_id']})", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/explainability/individual_sample_explanation.png", dpi=300)
    plt.savefig("experiments/explainability/individual_sample_explanation.png", dpi=300)
    plt.close()
    
    print(" -> Saved 3 Explainability Plots under reports/explainability/ & experiments/explainability/", flush=True)

if __name__ == "__main__":
    run_explainability_pipeline()
