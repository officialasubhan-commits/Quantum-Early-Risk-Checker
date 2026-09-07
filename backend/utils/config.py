import os

# Project Root Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths to trained model artifacts
MODEL_PATHS = {
    "preprocessor": os.path.join(BASE_DIR, "models", "preprocessor.joblib"),
    "classical_model": os.path.join(BASE_DIR, "models", "best_classical_model.joblib"),
    "pca_reducer": os.path.join(BASE_DIR, "models", "qml_pca_reducer.joblib"),
    "qml_params": os.path.join(BASE_DIR, "models", "qml_vqc_params.npz"),
    "hybrid_model": os.path.join(BASE_DIR, "models", "hybrid_fusion_model.joblib"),
    "dataset_ingestion_report": os.path.join(BASE_DIR, "reports", "dataset_ingestion_report.json"),
    "classical_report": os.path.join(BASE_DIR, "reports", "classical_ml_experiment_report.json"),
    "qml_report": os.path.join(BASE_DIR, "reports", "qml_experiment_report.json"),
    "hybrid_report": os.path.join(BASE_DIR, "reports", "hybrid_experiment_report.json"),
    "explainability_report": os.path.join(BASE_DIR, "reports", "explainability", "explainability_report.json"),
}

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'sih26139.db')}")

# API Metadata
API_TITLE = "SIH26139 — Hybrid Quantum Machine Learning Platform API"
API_DESCRIPTION = (
    "AI-Assisted Early Disease Risk Decision Support System for Diabetes/Prediabetes.\n\n"
    "**MEDICAL DISCLAIMER**: This API provides early-stage probabilistic risk decision-support "
    "estimates derived from classical, quantum, and hybrid machine learning models. "
    "It is NOT a diagnostic tool and DOES NOT replace professional clinical diagnosis, advice, or treatment."
)
API_VERSION = "1.0.0"

# Target feature ordering expected by preprocessor
FEATURE_NAMES_IN = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# Order of features output by preprocessor
FEATURE_NAMES_OUT = [
    "BMI", "MentHlth", "PhysHlth", "HighBP", "HighChol", "CholCheck",
    "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
