import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database.connection import engine, Base, SessionLocal
from backend.database.models import AssessmentRecord, ModelRegistryRecord
from backend.database.repository import register_model
from backend.utils.config import MODEL_PATHS

from sqlalchemy import text

def init_database():
    print("=" * 70, flush=True)
    print(" [DATABASE INITIALIZATION] Creating database tables & model registry ", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Create all defined tables
    Base.metadata.create_all(bind=engine)
    
    # Perform SQLite Column Migrations if needed
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE patient_assessments ADD COLUMN disease_id VARCHAR(64) DEFAULT 'diabetes'"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE patient_assessments ADD COLUMN disease_name VARCHAR(128) DEFAULT 'Diabetes / Prediabetes'"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN disease_id VARCHAR(64) DEFAULT 'diabetes'"))
        except Exception:
            pass

    print(" -> Database tables created/migrated successfully: 'patient_assessments', 'model_registry'", flush=True)
    
    # 2. Seed Model Registry from verified experiment reports
    db = SessionLocal()
    try:
        comparison_path = os.path.join(PROJECT_ROOT, "reports", "hybrid_vs_all_comparison.json")
        if os.path.exists(comparison_path):
            with open(comparison_path, "r") as f:
                comp_data = json.load(f)
                
            rf_info = comp_data.get("classical_rf", {})
            qml_info = comp_data.get("quantum_vqc", {})
            hyb_info = comp_data.get("hybrid_ensemble", {})
            test_samples = comp_data.get("experiment_metadata", {}).get("test_samples", 38052)
            
            # Register Classical Baseline
            register_model(
                db,
                model_name=rf_info.get("model_name", "Random Forest Baseline"),
                model_type="Classical",
                version="1.0.0",
                artifact_path=MODEL_PATHS["classical_model"],
                accuracy=rf_info.get("accuracy", 0.7348),
                f1_score=rf_info.get("f1_score", 0.4444),
                roc_auc=rf_info.get("roc_auc", 0.8240),
                recall=rf_info.get("recall_sensitivity", 0.7612),
                specificity=rf_info.get("specificity", 0.7306),
                test_samples=test_samples
            )
            
            # Register QML VQC
            register_model(
                db,
                model_name=qml_info.get("model_name", "6-Qubit Variational Quantum Classifier"),
                model_type="Quantum",
                version="1.0.0",
                artifact_path=MODEL_PATHS["qml_params"],
                accuracy=qml_info.get("accuracy", 0.2050),
                f1_score=qml_info.get("f1_score", 0.2451),
                roc_auc=qml_info.get("roc_auc", 0.5027),
                recall=qml_info.get("recall_sensitivity", 0.9261),
                specificity=qml_info.get("specificity", 0.0883),
                test_samples=test_samples
            )
            
            # Register Hybrid Model
            register_model(
                db,
                model_name=hyb_info.get("model_name", "Hybrid Classical-Quantum Ensemble"),
                model_type="Hybrid",
                version="1.0.0",
                artifact_path=MODEL_PATHS["hybrid_model"],
                accuracy=hyb_info.get("accuracy", 0.7879),
                f1_score=hyb_info.get("f1_score", 0.4622),
                roc_auc=hyb_info.get("roc_auc", 0.8351),
                recall=hyb_info.get("recall_sensitivity", 0.7845),
                specificity=hyb_info.get("specificity", 0.8095),
                test_samples=test_samples
            )
            print(" -> Seeded Model Registry Records from verified reports/hybrid_vs_all_comparison.json", flush=True)
    except Exception as e:
        print(f" Warning: Could not seed model registry: {e}", flush=True)
    finally:
        db.close()
        
    print(" [DATABASE INITIALIZATION] Database initialization complete.", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    init_database()
