import os
import sys
from fastapi import FastAPI, HTTPException, status, Request, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import uuid
import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

# Add project root to sys.path to ensure module imports resolve correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils.config import API_TITLE, API_DESCRIPTION, API_VERSION
from backend.schemas import (
    PatientFeatures,
    PredictionRequest,
    GenericDiseasePredictionRequest,
    PredictionResponse,
    ExplainabilityResponse,
    MultiDiseasePredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MEDICAL_DISCLAIMER_TEXT
)
from backend.services.model_service import ModelService
from src.utils.disease_registry import DiseaseRegistry

from backend.database.init_db import init_database
from backend.database.connection import get_db
from backend.database.repository import get_assessment_records, get_registered_models, get_dataset_records

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[FASTAPI SERVER] Starting API service, initializing Database & loading models...", flush=True)
    init_database()
    ModelService()
    yield

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS Middleware for Frontend & Dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Multilingual Voice AI Router
try:
    from voice_ai.app.api.router import router as voice_router
    app.include_router(voice_router)
except Exception as e:
    print(f"[FASTAPI SERVER] Voice AI router mount warning: {e}")


# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": request.url.path
        }
    )

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Returns API health status, disease registry count, and loaded state of model artifacts.
    """
    service = ModelService()
    return service.get_health()

@app.get("/api/v1/info", response_model=ModelInfoResponse, tags=["System"])
def get_model_information():
    """
    Retrieves verified model metadata, baseline performance comparison metrics, and dataset attributes.
    """
    service = ModelService()
    return service.get_model_info()

@app.get("/api/v1/diseases", tags=["Multi-Disease Registry"])
def get_registered_diseases():
    """
    Returns metadata for all targeted medical disease models registered in the platform registry.
    """
    registry = DiseaseRegistry()
    return {
        "registered_diseases": registry.get_all_diseases(),
        "active_count": len(registry.get_active_diseases()),
        "total_count": len(registry.get_all_diseases())
    }

@app.get("/api/v1/assessments", tags=["Database"])
def get_db_assessments(limit: int = 50, patient_id: Optional[str] = None, disease_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Retrieves stored patient assessment records from the database with optional disease filtering.
    """
    records = get_assessment_records(db, limit=limit, patient_id=patient_id, disease_id=disease_id)
    return [
        {
            "id": r.id,
            "request_id": r.request_id,
            "patient_id": r.patient_id,
            "disease_id": r.disease_id,
            "disease_name": r.disease_name,
            "model_used": r.model_used,
            "predicted_class": r.predicted_class,
            "predicted_label": r.predicted_label,
            "risk_probability": r.risk_probability,
            "is_high_risk": r.is_high_risk,
            "feature_inputs": r.feature_inputs,
            "top_contributing_features": r.top_contributing_features,
            "clinical_narrative": r.clinical_narrative,
            "timestamp": r.timestamp.isoformat() if r.timestamp is not None else None
        }
        for r in records
    ]

@app.get("/api/v1/registry", tags=["Database"])
def get_model_registry(disease_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Retrieves registered model versions and benchmark metrics from the database.
    """
    records = get_registered_models(db, disease_id=disease_id)
    return [
        {
            "id": r.id,
            "disease_id": r.disease_id,
            "model_name": r.model_name,
            "model_type": r.model_type,
            "version": r.version,
            "artifact_path": r.artifact_path,
            "accuracy": r.accuracy,
            "f1_score": r.f1_score,
            "roc_auc": r.roc_auc,
            "recall": r.recall,
            "specificity": r.specificity,
            "test_samples": r.test_samples,
            "created_at": r.created_at.isoformat() if r.created_at is not None else None
        }
        for r in records
    ]

@app.get("/api/v1/datasets", tags=["Database"])
def get_biomedical_datasets(disease_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Retrieves registered biomedical datasets, sources, modalities, and sample counts linked to target diseases.
    """
    records = get_dataset_records(db, disease_id=disease_id)
    return [
        {
            "id": r.id,
            "dataset_id": r.dataset_id,
            "disease_id": r.disease_id,
            "name": r.name,
            "source": r.source,
            "modality": r.modality,
            "target_name": r.target_name,
            "sample_count": r.sample_count,
            "feature_count": r.feature_count,
            "access_info": r.access_info,
            "is_acquired": r.is_acquired,
            "created_at": r.created_at.isoformat() if r.created_at is not None else None
        }
        for r in records
    ]


@app.post("/api/v1/predict/disease/{disease_id}/{model_type}", response_model=PredictionResponse, tags=["Multi-Disease Predictions"])
def predict_disease_endpoint(disease_id: str, model_type: str, request: GenericDiseasePredictionRequest):
    """
    Executes prediction for a specific disease and model architecture (classical, qml, hybrid).
    """
    try:
        service = ModelService()
        return service.predict_disease(disease_id, model_type, request.features, patient_id=request.patient_id or "PATIENT_001")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction for '{disease_id}' ({model_type}) failed: {str(e)}"
        )

@app.post("/api/v1/predict/all", response_model=MultiDiseasePredictionResponse, tags=["Multi-Disease Predictions"])
def predict_all_diseases_endpoint(request: GenericDiseasePredictionRequest):
    """
    Evaluates patient features against all registered active disease models for which inputs are valid.
    """
    try:
        service = ModelService()
        return service.predict_all_diseases(request.features, patient_id=request.patient_id or "PATIENT_001")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Multi-disease risk assessment failed: {str(e)}"
        )

@app.post("/api/v1/explain/disease/{disease_id}", response_model=ExplainabilityResponse, tags=["Explainability"])
def explain_disease_endpoint(disease_id: str, request: GenericDiseasePredictionRequest):
    """
    Generates feature attributions and a human-readable clinical explanation narrative for a specific disease model.
    """
    try:
        service = ModelService()
        return service.explain_disease(disease_id, request.features, patient_id=request.patient_id or "PATIENT_001")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Explainability generation for '{disease_id}' failed: {str(e)}"
        )

# Legacy Single-Disease Endpoints (Maintained for Backward Compatibility)
@app.post("/api/v1/predict/classical", response_model=PredictionResponse, tags=["Legacy Predictions"])
def predict_classical_baseline(request: PredictionRequest):
    """
    Executes prediction using the Classical Baseline (Random Forest) model.
    """
    try:
        service = ModelService()
        return service.predict_classical(request.features, patient_id=request.patient_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classical prediction failed: {str(e)}"
        )

@app.post("/api/v1/predict/qml", response_model=PredictionResponse, tags=["Legacy Predictions"])
def predict_quantum_vqc(request: PredictionRequest):
    """
    Executes prediction using the standalone 6-Qubit Variational Quantum Classifier (QML).
    """
    try:
        service = ModelService()
        return service.predict_qml(request.features, patient_id=request.patient_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QML prediction failed: {str(e)}"
        )

@app.post("/api/v1/predict/hybrid", response_model=PredictionResponse, tags=["Legacy Predictions"])
def predict_hybrid_ensemble(request: PredictionRequest):
    """
    Executes prediction using the Hybrid Classical-Quantum Ensemble model with optimal thresholding.
    """
    try:
        service = ModelService()
        return service.predict_hybrid(request.features, patient_id=request.patient_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid prediction failed: {str(e)}"
        )

@app.post("/api/v1/explain", response_model=ExplainabilityResponse, tags=["Explainability"])
def explain_patient_risk(request: PredictionRequest):
    """
    Generates global & local feature attributions along with a human-readable clinical explanation narrative.
    """
    try:
        service = ModelService()
        return service.explain(request.features, patient_id=request.patient_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability generation failed: {str(e)}"
        )

@app.post("/api/v1/upload-report", tags=["Medical Reports"])
async def upload_medical_report(file: UploadFile = File(...)):
    """
    Accepts patient medical report files (PDF, PNG, JPG, JPEG).
    Validates file format and size constraint (Max 10MB).
    """
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed_extensions)}"
        )
        
    contents = await file.read()
    max_bytes = 10 * 1024 * 1024  # 10 MB
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size ({len(contents) / (1024*1024):.2f} MB) exceeds maximum allowed threshold of 10 MB."
        )
        
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filename": filename,
        "content_type": file.content_type,
        "file_size_kb": round(len(contents) / 1024.0, 2),
        "status": "Uploaded Successfully",
        "extraction_status": "PENDING_CLINICAL_NLP_INTEGRATION",
        "message": (
            "Medical report received successfully. Automated OCR & clinical NLP extraction is "
            "currently PENDING pipeline integration. Please complete the health profile indicators manually."
        ),
        "disclaimer": MEDICAL_DISCLAIMER_TEXT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8001, reload=True)
