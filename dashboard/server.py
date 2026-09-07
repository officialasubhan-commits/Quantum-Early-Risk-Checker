import os
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CHECKLIST_FILE = os.path.join(REPORTS_DIR, "task_checklist.json")

DEFAULT_PHASES = [
    {"id": "dataset_discovery", "name": "Dataset Discovery", "status": "COMPLETED", "description": "Researched and identified CDC BRFSS 2015 dataset"},
    {"id": "dataset_verification", "name": "Dataset Verification", "status": "COMPLETED", "description": "Verified 253,680 records & open domain CC0 license"},
    {"id": "data_ingestion", "name": "Data Ingestion", "status": "COMPLETED", "description": "10-point validation report generated in reports/"},
    {"id": "preprocessing", "name": "Preprocessing", "status": "COMPLETED", "description": "Stratified 70/15/15 split with leak-free StandardScaler"},
    {"id": "classical_ml", "name": "Classical ML", "status": "COMPLETED", "description": "Trained & cross-validated LR, RF, and SVM models"},
    {"id": "qml", "name": "QML", "status": "COMPLETED", "description": "6-Qubit VQC with PCA feature reduction evaluated on test set"},
    {"id": "hybrid_model", "name": "Hybrid Model", "status": "COMPLETED", "description": "Probability-blended meta-classifier ensemble evaluated on 38,052 test set"},
    {"id": "model_evaluation", "name": "Model Evaluation", "status": "COMPLETED", "description": "Classical, Quantum, and Hybrid models evaluated on 38,052 test set"},
    {"id": "explainability", "name": "Explainability", "status": "COMPLETED", "description": "Global feature attributions & patient clinical narrative generator"},
    {"id": "fastapi", "name": "FastAPI", "status": "COMPLETED", "description": "REST API backend prediction endpoints with Pydantic validation"},
    {"id": "database", "name": "Database", "status": "COMPLETED", "description": "Production-ready SQLite/SQLAlchemy ORM database layer with model registry"},
    {"id": "web_application", "name": "Web Application", "status": "COMPLETED", "description": "Internal SIH26139 project monitoring dashboard"},
    {"id": "testing", "name": "Testing", "status": "NOT STARTED", "description": "Unit tests and integration test suites"},
    {"id": "sih_demo", "name": "SIH Demo", "status": "NOT STARTED", "description": "Final SIH competition presentation package"}
]

DEFAULT_CHECKLIST = [
    {"id": 1, "task": "Inspect existing codebase and data ingestion", "phase": "Data Ingestion", "status": "COMPLETED"},
    {"id": 2, "task": "10-point data validation & ingestion report generation", "phase": "Data Ingestion", "status": "COMPLETED"},
    {"id": 3, "task": "Stratified 70/15/15 train/val/test data pipeline split", "phase": "Preprocessing", "status": "COMPLETED"},
    {"id": 4, "task": "Train and cross-validate classical ML baseline models", "phase": "Classical ML", "status": "COMPLETED"},
    {"id": 5, "task": "Save best classical model & preprocessor artifacts", "phase": "Classical ML", "status": "COMPLETED"},
    {"id": 6, "task": "Develop internal project monitoring dashboard UI", "phase": "Web Application", "status": "COMPLETED"},
    {"id": 7, "task": "Implement PCA dimensionality reduction (21 -> 6 quantum features)", "phase": "QML", "status": "COMPLETED"},
    {"id": 8, "task": "Construct 6-qubit Variational Quantum Circuit (VQC)", "phase": "QML", "status": "COMPLETED"},
    {"id": 9, "task": "Train VQC on simulator & evaluate on 38,052 test set", "phase": "QML", "status": "COMPLETED"},
    {"id": 10, "task": "Train Hybrid Classical-Quantum Classifier & Meta-Ensemble", "phase": "Hybrid Model", "status": "COMPLETED"},
    {"id": 11, "task": "Compute global feature attributions & patient explanations", "phase": "Explainability", "status": "COMPLETED"},
    {"id": 12, "task": "Build FastAPI prediction & monitoring endpoints", "phase": "FastAPI", "status": "COMPLETED"},
    {"id": 13, "task": "Validate all REST API endpoints & OpenAPI schema", "phase": "FastAPI", "status": "COMPLETED"},
    {"id": 14, "task": "Build patient-facing Web Application interface (http://localhost:8080/)", "phase": "Web Application", "status": "COMPLETED"},
    {"id": 15, "task": "Integrate Web Application with FastAPI backend & end-to-end testing", "phase": "Web Application", "status": "COMPLETED"},
    {"id": 16, "task": "Configure SQLAlchemy ORM engine & SQLite schema (data/sih26139.db)", "phase": "Database", "status": "COMPLETED"},
    {"id": 17, "task": "Implement patient_assessments & model_registry tables with CRUD repository", "phase": "Database", "status": "COMPLETED"},
    {"id": 18, "task": "Integrate Database persistence into FastAPI endpoints (/assessments, /registry)", "phase": "Database", "status": "COMPLETED"}
]

def load_checklist():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if os.path.exists(CHECKLIST_FILE):
        try:
            with open(CHECKLIST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    with open(CHECKLIST_FILE, "w") as f:
        json.dump(DEFAULT_CHECKLIST, f, indent=4)
    return DEFAULT_CHECKLIST

def save_checklist(checklist):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(CHECKLIST_FILE, "w") as f:
        json.dump(checklist, f, indent=4)

def get_ml_status():
    ingestion_file = os.path.join(REPORTS_DIR, "dataset_ingestion_report.json")
    metrics_file = os.path.join(REPORTS_DIR, "experiment_metrics.json")
    qml_file = os.path.join(REPORTS_DIR, "qml_experiment_report.json")
    
    num_records = "Not available"
    num_features = "Not available"
    best_model_name = "Not available"
    accuracy = "Not available"
    precision = "Not available"
    recall = "Not available"
    f1_score = "Not available"
    roc_auc = "Not available"
    qml_status = "Not available"
    
    if os.path.exists(ingestion_file):
        try:
            with open(ingestion_file, "r") as f:
                ingest_data = json.load(f)
                vm = ingest_data.get("validation_metrics", {})
                num_records = f"{vm.get('num_rows', 0):,}"
                num_features = str(vm.get("num_features", "Not available"))
        except Exception:
            pass
            
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                
            best_f1 = -1.0
            best_m_data = None
            for m_name, m_data in metrics_data.items():
                if m_data.get("f1_score", 0.0) > best_f1:
                    best_f1 = m_data.get("f1_score", 0.0)
                    best_m_data = m_data
                    best_model_name = m_name
                    
            if best_m_data:
                accuracy = f"{best_m_data.get('accuracy', 0):.4f} ({best_m_data.get('accuracy', 0)*100:.2f}%)"
                precision = f"{best_m_data.get('precision', 0):.4f} ({best_m_data.get('precision', 0)*100:.2f}%)"
                recall = f"{best_m_data.get('recall_sensitivity', 0):.4f} ({best_m_data.get('recall_sensitivity', 0)*100:.2f}%)"
                f1_score = f"{best_m_data.get('f1_score', 0):.4f}"
                roc_auc = f"{best_m_data.get('roc_auc', 0):.4f}"
        except Exception:
            pass

    if os.path.exists(qml_file):
        try:
            with open(qml_file, "r") as f:
                q_data = json.load(f)
                qml_status = f"VQC (6 Qubits, F1: {q_data.get('f1_score', 0):.4f}, Recall: {q_data.get('recall_sensitivity', 0)*100:.1f}%)"
        except Exception:
            pass
            
    return {
        "dataset_records": num_records,
        "num_features": num_features,
        "best_classical_model": best_model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "roc_auc": roc_auc,
        "qml_status": qml_status
    }

class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/api/data":
            checklist = load_checklist()
            ml_status = get_ml_status()
            
            # Audit-based dynamic progress calculation across 14 phases
            completed_phases = sum(1 for p in DEFAULT_PHASES if p["status"] == "COMPLETED")
            in_progress_phases = sum(1 for p in DEFAULT_PHASES if p["status"] == "IN PROGRESS")
            total_phases = len(DEFAULT_PHASES)
            progress_pct = int(((completed_phases + 0.5 * in_progress_phases) / total_phases) * 100)

            response_payload = {
                "project_info": {
                    "name": "Hybrid Quantum Machine Learning Platform",
                    "code": "SIH26139",
                    "status": "Development",
                    "environment": "Local Development Server"
                },
                "current_task": {
                    "current_phase": "Phase 13: End-to-End Automated Testing Suites",
                    "current_task": "Integration & Regression Unit Test Execution",
                    "progress_percentage": progress_pct,
                    "last_completed_task": "Phase 12 Database Layer, Model Registry & SQLite ORM Integration",
                    "next_task": "Integration & Regression Unit Test Suite Execution"
                },
                "phases": DEFAULT_PHASES,
                "ml_status": ml_status,
                "architecture_flow": [
                    {"step": 1, "title": "Dataset", "desc": "CDC BRFSS 2015 (253,680 Records)"},
                    {"step": 2, "title": "Data Validation", "desc": "10-Point Missingness & Range Check"},
                    {"step": 3, "title": "Preprocessing", "desc": "Stratified Split & Fit StandardScaler"},
                    {"step": 4, "title": "Classical ML", "desc": "LR, RF & SVM Baseline Evaluation"},
                    {"step": 5, "title": "Feature Reduction", "desc": "PCA 21 to 6 Quantum Features"},
                    {"step": 6, "title": "QML", "desc": "6-Qubit Variational Quantum Circuit (VQC)"},
                    {"step": 7, "title": "Hybrid Model", "desc": "Classical-Quantum Ensemble Fusion"},
                    {"step": 8, "title": "Evaluation", "desc": "F1, ROC-AUC, Latency Metrics"},
                    {"step": 9, "title": "Explainability", "desc": "SHAP & LIME Feature Attribution"},
                    {"step": 10, "title": "API", "desc": "FastAPI Model Inference Service"},
                    {"step": 11, "title": "Web Dashboard", "desc": "SIH26139 Development Monitor"}
                ],
                "checklist": checklist
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload, indent=2).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/api/tasks":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                task_id = data.get("id")
                new_status = data.get("status")
                
                checklist = load_checklist()
                updated = False
                for item in checklist:
                    if item["id"] == task_id:
                        item["status"] = new_status
                        updated = True
                        break
                        
                if updated:
                    save_checklist(checklist)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "checklist": checklist}).encode("utf-8"))
                else:
                    self.send_response(400)
                    self.end_headers()
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port=PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    print(f"=" * 70)
    print(f" SIH26139 Project Monitoring Dashboard Server Running ")
    print(f" Access URL: http://localhost:{port}/")
    print(f" Press Ctrl+C to stop the server")
    print(f"=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
