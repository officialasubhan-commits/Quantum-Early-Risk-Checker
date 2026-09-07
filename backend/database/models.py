import datetime
from typing import Optional, Any
from sqlalchemy import Integer, String, Float, Boolean, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base

class AssessmentRecord(Base):
    """
    Database Table: patient_assessments
    Stores validated patient risk predictions, model outputs, and feature attributions.
    """
    __tablename__ = "patient_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    disease_id: Mapped[str] = mapped_column(String(64), index=True, default="diabetes", nullable=False)
    disease_name: Mapped[str] = mapped_column(String(128), default="Diabetes / Prediabetes", nullable=False)
    model_used: Mapped[str] = mapped_column(String(128), nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(128), nullable=False)
    risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    is_high_risk: Mapped[bool] = mapped_column(Boolean, nullable=False)
    
    feature_inputs: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    top_contributing_features: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    clinical_narrative: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    disclaimer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    timestamp: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True
    )

class ModelRegistryRecord(Base):
    """
    Database Table: model_registry
    Stores verified performance metrics and versioning metadata for trained models.
    """
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    disease_id: Mapped[str] = mapped_column(String(64), index=True, default="diabetes", nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)  # "Classical", "Quantum", "Hybrid"
    version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    artifact_path: Mapped[str] = mapped_column(String(256), nullable=False)
    
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    specificity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    test_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

class DiseaseRegistryRecord(Base):
    """
    Database Table: disease_registry
    Stores metadata, categories, symptoms, risk factors, and feature configurations for target diseases.
    """
    __tablename__ = "disease_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    disease_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    symptoms: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_factors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dataset_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    target_variable: Mapped[str] = mapped_column(String(64), nullable=False, default="target")
    data_modality: Mapped[str] = mapped_column(String(64), nullable=False, default="Tabular Health Indicators")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    features: Mapped[Any] = mapped_column(JSON, nullable=False)
    qml_config: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

class DatasetRegistryRecord(Base):
    """
    Database Table: dataset_registry
    Stores biomedical dataset sources, modalities, sample counts, and access info linked to diseases.
    """
    __tablename__ = "dataset_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    disease_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    modality: Mapped[str] = mapped_column(String(128), nullable=False, default="Tabular Clinical Features")
    target_name: Mapped[str] = mapped_column(String(64), nullable=False, default="target")
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    access_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_path: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    processed_path: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    is_acquired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
