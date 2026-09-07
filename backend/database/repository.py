from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from backend.database.models import AssessmentRecord, ModelRegistryRecord, DiseaseRegistryRecord, DatasetRegistryRecord

def save_assessment_record(
    db: Session,
    request_id: str,
    patient_id: str,
    model_used: str,
    predicted_class: int,
    predicted_label: str,
    risk_probability: float,
    is_high_risk: bool,
    disease_id: str = "diabetes",
    disease_name: str = "Diabetes / Prediabetes",
    feature_inputs: Optional[Dict[str, Any]] = None,
    top_contributing_features: Optional[List[Dict[str, Any]]] = None,
    clinical_narrative: Optional[str] = None,
    disclaimer: Optional[str] = None
) -> AssessmentRecord:
    """
    Saves an assessment record to the database.
    If a record with the same request_id exists, updates it with explainability data.
    """
    existing = db.query(AssessmentRecord).filter(AssessmentRecord.request_id == request_id).first()
    if existing:
        if top_contributing_features is not None:
            existing.top_contributing_features = top_contributing_features
        if clinical_narrative is not None:
            existing.clinical_narrative = clinical_narrative
        db.commit()
        db.refresh(existing)
        return existing

    record = AssessmentRecord(
        request_id=request_id,
        patient_id=patient_id,
        disease_id=disease_id,
        disease_name=disease_name,
        model_used=model_used,
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        risk_probability=risk_probability,
        is_high_risk=is_high_risk,
        feature_inputs=feature_inputs,
        top_contributing_features=top_contributing_features,
        clinical_narrative=clinical_narrative,
        disclaimer=disclaimer
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_assessment_records(db: Session, limit: int = 50, patient_id: Optional[str] = None, disease_id: Optional[str] = None) -> List[AssessmentRecord]:
    query = db.query(AssessmentRecord)
    if patient_id:
        query = query.filter(AssessmentRecord.patient_id == patient_id)
    if disease_id:
        query = query.filter(AssessmentRecord.disease_id == disease_id)
    return query.order_by(AssessmentRecord.timestamp.desc()).limit(limit).all()

def get_assessment_by_request_id(db: Session, request_id: str) -> Optional[AssessmentRecord]:
    return db.query(AssessmentRecord).filter(AssessmentRecord.request_id == request_id).first()

def register_model(
    db: Session,
    model_name: str,
    model_type: str,
    version: str,
    artifact_path: str,
    accuracy: float,
    f1_score: float,
    roc_auc: float,
    recall: float,
    specificity: float,
    test_samples: int,
    disease_id: str = "diabetes"
) -> ModelRegistryRecord:
    existing = db.query(ModelRegistryRecord).filter(
        ModelRegistryRecord.model_name == model_name,
        ModelRegistryRecord.version == version,
        ModelRegistryRecord.disease_id == disease_id
    ).first()
    
    if existing:
        existing.accuracy = accuracy
        existing.f1_score = f1_score
        existing.roc_auc = roc_auc
        existing.recall = recall
        existing.specificity = specificity
        existing.test_samples = test_samples
        db.commit()
        db.refresh(existing)
        return existing

    record = ModelRegistryRecord(
        disease_id=disease_id,
        model_name=model_name,
        model_type=model_type,
        version=version,
        artifact_path=artifact_path,
        accuracy=accuracy,
        f1_score=f1_score,
        roc_auc=roc_auc,
        recall=recall,
        specificity=specificity,
        test_samples=test_samples
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_registered_models(db: Session, disease_id: Optional[str] = None) -> List[ModelRegistryRecord]:
    query = db.query(ModelRegistryRecord)
    if disease_id:
        query = query.filter(ModelRegistryRecord.disease_id == disease_id)
    return query.all()

def register_disease_record(
    db: Session,
    disease_id: str,
    name: str,
    category: str,
    description: str,
    symptoms: str,
    risk_factors: str,
    dataset_name: str,
    target_variable: str,
    data_modality: str,
    features: List[str],
    qml_config: Optional[Dict[str, Any]] = None,
    status: str = "ACTIVE",
    model_version: str = "1.0.0"
) -> DiseaseRegistryRecord:
    existing = db.query(DiseaseRegistryRecord).filter(DiseaseRegistryRecord.disease_id == disease_id).first()
    if existing:
        existing.name = name
        existing.category = category
        existing.description = description
        existing.symptoms = symptoms
        existing.risk_factors = risk_factors
        existing.dataset_name = dataset_name
        existing.target_variable = target_variable
        existing.data_modality = data_modality
        existing.features = features
        existing.qml_config = qml_config or {}
        existing.status = status
        existing.model_version = model_version
        db.commit()
        db.refresh(existing)
        return existing

    record = DiseaseRegistryRecord(
        disease_id=disease_id,
        name=name,
        category=category,
        description=description,
        symptoms=symptoms,
        risk_factors=risk_factors,
        dataset_name=dataset_name,
        target_variable=target_variable,
        data_modality=data_modality,
        features=features,
        qml_config=qml_config or {},
        status=status,
        model_version=model_version
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_disease_records(db: Session, active_only: bool = False) -> List[DiseaseRegistryRecord]:
    query = db.query(DiseaseRegistryRecord)
    if active_only:
        query = query.filter(DiseaseRegistryRecord.status == "ACTIVE")
    return query.all()

def register_dataset_record(
    db: Session,
    dataset_id: str,
    disease_id: str,
    name: str,
    source: str,
    modality: str,
    target_name: str,
    sample_count: int,
    feature_count: int,
    access_info: str,
    raw_path: Optional[str] = None,
    processed_path: Optional[str] = None,
    is_acquired: bool = True
) -> DatasetRegistryRecord:
    existing = db.query(DatasetRegistryRecord).filter(DatasetRegistryRecord.dataset_id == dataset_id).first()
    if existing:
        existing.name = name
        existing.source = source
        existing.modality = modality
        existing.target_name = target_name
        existing.sample_count = sample_count
        existing.feature_count = feature_count
        existing.access_info = access_info
        existing.raw_path = raw_path
        existing.processed_path = processed_path
        existing.is_acquired = is_acquired
        db.commit()
        db.refresh(existing)
        return existing

    record = DatasetRegistryRecord(
        dataset_id=dataset_id,
        disease_id=disease_id,
        name=name,
        source=source,
        modality=modality,
        target_name=target_name,
        sample_count=sample_count,
        feature_count=feature_count,
        access_info=access_info,
        raw_path=raw_path,
        processed_path=processed_path,
        is_acquired=is_acquired
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_dataset_records(db: Session, disease_id: Optional[str] = None) -> List[DatasetRegistryRecord]:
    query = db.query(DatasetRegistryRecord)
    if disease_id:
        query = query.filter(DatasetRegistryRecord.disease_id == disease_id)
    return query.all()

