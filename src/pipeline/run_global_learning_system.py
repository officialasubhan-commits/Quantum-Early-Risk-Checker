import os
import sys
import json
from typing import List, Dict, Any
import pandas as pd
from sqlalchemy.orm import Session

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.database.init_db import init_database
from backend.database.connection import SessionLocal
from backend.database.repository import register_disease_record, register_dataset_record, get_disease_records, get_dataset_records, get_registered_models
from src.data_ingestion.ingest_global_datasets import ingest_all_global_datasets
from src.pipeline.auto_training_pipeline import AutoDiseaseMLPipeline
from src.utils.disease_registry import DiseaseRegistry

def seed_existing_baseline_diseases(db: Session) -> None:
    """
    Seeds database records for the 7 baseline disease targets.
    """
    baseline_diseases: List[Dict[str, Any]] = [
        {
            "id": "diabetes", "name": "Diabetes / Prediabetes Risk Assessment", "category": "Endocrine & Metabolic",
            "desc": "AI-assisted early-risk assessment for prediabetes and type-2 diabetes based on 21 clinical health indicators.",
            "symptoms": "Polyuria, polydipsia, unexplained weight loss, fatigue, blurred vision",
            "risk_factors": "High BMI, hypertension, high cholesterol, physical inactivity, smoking, family history",
            "dataset": "CDC BRFSS 2015 Diabetes Health Indicators", "target": "Diabetes_binary",
            "source": "CDC Behavioral Risk Factor Surveillance System", "samples": 253680,
            "raw": "data/raw/cdc_diabetes_health_indicators.csv"
        },
        {
            "id": "heart_disease", "name": "Cardiovascular / Heart Disease Risk Assessment", "category": "Cardiology",
            "desc": "Early risk assessment for coronary artery disease and heart failure based on cardiovascular biomarkers.",
            "symptoms": "Chest discomfort/angina, shortness of breath, palpitations, fatigue, dizziness",
            "risk_factors": "Elevated blood pressure, high serum cholesterol, age, smoking, abnormal ECG, exercise angina",
            "dataset": "UCI Heart Disease Cleveland Benchmark Dataset", "target": "target",
            "source": "UCI Machine Learning Repository / Cleveland Clinic", "samples": 303,
            "raw": "data/raw/heart_disease/heart_disease_cleveland.csv"
        },
        {
            "id": "kidney_disease", "name": "Chronic Kidney Disease Risk Assessment", "category": "Nephrology",
            "desc": "Early risk screening for chronic kidney disease (CKD) using renal biomarkers and clinical indicators.",
            "symptoms": "Edema, changes in urination, fatigue, nausea, persistent pruritus, anemia",
            "risk_factors": "Hypertension, diabetes mellitus, elevated serum creatinine, proteinuria, age",
            "dataset": "UCI Chronic Kidney Disease Dataset", "target": "target",
            "source": "UCI Machine Learning Repository / Apollo Hospitals", "samples": 400,
            "raw": "data/raw/kidney_disease/raw_data.csv"
        },
        {
            "id": "liver_disease", "name": "Liver Disease Risk Assessment", "category": "Hepatology",
            "desc": "Early screening for hepatic dysfunction and liver disease using serum bilirubin and enzyme markers.",
            "symptoms": "Jaundice, abdominal pain/swelling, dark urine, pale stool, persistent fatigue",
            "risk_factors": "Alcohol consumption, elevated liver enzymes (ALT, AST, ALP), hyperbilirubinemia, age",
            "dataset": "UCI Indian Liver Patient Dataset", "target": "target",
            "source": "UCI Machine Learning Repository", "samples": 583,
            "raw": "data/raw/liver_disease/raw_data.csv"
        },
        {
            "id": "stroke", "name": "Stroke Risk Assessment", "category": "Neurology",
            "desc": "Cerebrovascular accident and stroke prediction using clinical risk factors and lifestyle indicators.",
            "symptoms": "Sudden numbness/weakness, confusion, speech difficulty, visual impairment, severe headache",
            "risk_factors": "Hypertension, heart disease, elevated average glucose, advanced age, smoking, high BMI",
            "dataset": "Kaggle Healthcare Stroke Dataset", "target": "target",
            "source": "Kaggle Healthcare Datasets", "samples": 5110,
            "raw": "data/raw/stroke/raw_data.csv"
        },
        {
            "id": "breast_cancer", "name": "Breast Cancer Risk Assessment", "category": "Oncology",
            "desc": "Biomarker and cell nucleus morphometric analysis for early breast cancer risk stratification.",
            "symptoms": "Painless breast mass, skin changes, nipple retraction or discharge, axillary lymphadenopathy",
            "risk_factors": "Age, genetic mutations (BRCA), dense breast tissue, hormonal exposure, family history",
            "dataset": "UCI Breast Cancer Wisconsin (Diagnostic) Dataset", "target": "target",
            "source": "UCI Machine Learning Repository / Univ of Wisconsin", "samples": 569,
            "raw": "data/raw/breast_cancer/raw_data.csv"
        },
        {
            "id": "parkinsons", "name": "Parkinson's Disease Risk Assessment", "category": "Neurology & Movement",
            "desc": "Acoustic and voice frequency analysis for early detection of Parkinsonian neurological tremor.",
            "symptoms": "Resting tremor, bradykinesia, rigidity, postural instability, vocal dysphonia",
            "risk_factors": "Advanced age, male sex, environmental toxin exposure, acoustic jitter/shimmer variations",
            "dataset": "UCI Parkinson's Disease Vocal Dataset", "target": "target",
            "source": "UCI Machine Learning Repository / Oxford University", "samples": 195,
            "raw": "data/raw/parkinsons/raw_data.csv"
        }
    ]

    reg = DiseaseRegistry()
    all_reg = reg.get_all_diseases()

    for d in baseline_diseases:
        d_info: Dict[str, Any] = all_reg.get(str(d["id"]), {})
        features: List[str] = d_info.get("features", [])
        register_disease_record(
            db,
            disease_id=str(d["id"]),
            name=str(d["name"]),
            category=str(d["category"]),
            description=str(d["desc"]),
            symptoms=str(d["symptoms"]),
            risk_factors=str(d["risk_factors"]),
            dataset_name=str(d["dataset"]),
            target_variable=str(d["target"]),
            data_modality="Tabular Clinical Features",
            features=features,
            qml_config={"n_qubits": 6, "n_layers": 2, "pca_dim": 6},
            status="ACTIVE",
            model_version="1.0.0"
        )
        register_dataset_record(
            db,
            dataset_id=f"ds_{d['id']}",
            disease_id=str(d["id"]),
            name=str(d["dataset"]),
            source=str(d["source"]),
            modality="Tabular Clinical Biomarkers",
            target_name=str(d["target"]),
            sample_count=int(d["samples"]),
            feature_count=len(features),
            access_info="Open Benchmark Medical Repository",
            raw_path=os.path.join(BASE_DIR, str(d["raw"])),
            processed_path=os.path.join(BASE_DIR, "data", "processed", str(d["id"]), "test.csv"),
            is_acquired=True
        )

def run_global_system():
    print("=" * 80)
    print(" [STARTING GLOBAL DISEASE & DATASET LEARNING SYSTEM (SIH26139 PLATFORM)]")
    print("=" * 80)

    # Step 1: Initialize Database Tables
    init_database()
    db = SessionLocal()
    seed_existing_baseline_diseases(db)
    db.close()

    # Step 2: Acquire/Generate Datasets for New Medical Categories
    ingest_all_global_datasets()

    # Step 3: Define New Disease Configurations for Automated Training
    new_diseases: List[Dict[str, Any]] = [
        {
            "id": "thyroid",
            "name": "Thyroid Disease Risk Assessment",
            "category": "Endocrine & Thyroid",
            "desc": "AI-assisted screening for hypothyroidism and thyroid dysfunction using TSH, T3, T4, and clinical indicators.",
            "symptoms": "Fatigue, weight gain/loss, temperature intolerance, dry skin, goitre, mood changes",
            "risk_factors": "Female sex, age > 50, personal/family history of autoimmune disease, iodine levels",
            "target": "target",
            "dataset": "UCI Thyroid Disease Benchmark Dataset (Garavan Institute)",
            "source": "UCI Machine Learning Repository / Garavan Institute",
            "csv_path": os.path.join(BASE_DIR, "data", "raw", "thyroid", "raw_data.csv")
        },
        {
            "id": "lung_cancer",
            "name": "Thoracic & Lung Cancer Risk Assessment",
            "category": "Pulmonology & Oncology",
            "desc": "Risk stratification for thoracic surgery complications and early lung cancer screening using pulmonary spirometry (FVC, FEV1).",
            "symptoms": "Persistent cough, dyspnoea, haemoptysis, thoracic chest pain, unexplained weight loss",
            "risk_factors": "Smoking history, occupational toxin exposure, low FEV1/FVC ratio, advanced age, large tumor size",
            "target": "target",
            "dataset": "UCI Thoracic Surgery & Lung Cancer Dataset",
            "source": "UCI Machine Learning Repository",
            "csv_path": os.path.join(BASE_DIR, "data", "raw", "lung_cancer", "raw_data.csv")
        },
        {
            "id": "alzheimers",
            "name": "Alzheimer's & Dementia Risk Assessment",
            "category": "Neurology & Geriatrics",
            "desc": "Early cognitive impairment and Alzheimer's dementia risk evaluation based on MMSE scores, CDR ratings, and brain MRI volume markers.",
            "symptoms": "Memory loss, cognitive disorientation, executive dysfunction, language impairment, behavioral changes",
            "risk_factors": "Advanced age, low Mini-Mental State Exam (MMSE) score, reduced whole brain volume (nWBV), APOE-e4 allele",
            "target": "target",
            "dataset": "OASIS Longitudinal MRI Dementia Dataset",
            "source": "Open Access Series of Imaging Studies (OASIS) / Washington University",
            "csv_path": os.path.join(BASE_DIR, "data", "raw", "alzheimers", "raw_data.csv")
        },
        {
            "id": "hypertension",
            "name": "Clinical Hypertension Risk Assessment",
            "category": "Cardiology & Vascular",
            "desc": "Early risk detection for primary and secondary hypertension using blood pressure dynamics, electrolyte balances, and lifestyle markers.",
            "symptoms": "Occipital headache, dizziness, visual disturbances, palpitations, epistaxis",
            "risk_factors": "High SBP/DBP, elevated BMI, high dietary sodium, physical inactivity, smoking, family history",
            "target": "target",
            "dataset": "Clinical Hypertension & Vascular Health Dataset",
            "source": "NHANES / Clinical Health Indicators",
            "csv_path": os.path.join(BASE_DIR, "data", "raw", "hypertension", "raw_data.csv")
        }
    ]

    # Step 4: Execute Automated Pipeline for Each New Disease Target
    pipeline_results = {}
    for d in new_diseases:
        pipeline = AutoDiseaseMLPipeline(
            disease_id=str(d["id"]),
            disease_name=str(d["name"]),
            category=str(d["category"]),
            description=str(d["desc"]),
            target_col=str(d["target"]),
            dataset_name=str(d["dataset"]),
            source=str(d["source"]),
            raw_csv_path=str(d["csv_path"]),
            symptoms=str(d["symptoms"]),
            risk_factors=str(d["risk_factors"]),
            n_qubits=6
        )
        res = pipeline.run_pipeline()
        pipeline_results[str(d["id"])] = res

    # Step 5: Generate Final System Audit Summary
    db = SessionLocal()
    disease_recs = get_disease_records(db)
    dataset_recs = get_dataset_records(db)
    model_recs = get_registered_models(db)
    db.close()

    print("\n" + "=" * 80)
    print(" [GLOBAL DISEASE & DATASET LEARNING SYSTEM AUDIT SUMMARY]")
    print("=" * 80)
    print(f" Total Diseases Registered: {len(disease_recs)}")
    print(f" Total Datasets Acquired & Linked: {len(dataset_recs)}")
    print(f" Total Trained Models Saved: {len(model_recs)}")
    print("-" * 80)
    print(" Disease Target Summary:")
    for dr in disease_recs:
        features_list = getattr(dr, "features", []) or []
        print(f"  * [{dr.status}] {dr.name} ({dr.disease_id.upper()}) | Category: {dr.category} | Features: {len(features_list)}")
    print("=" * 80)

if __name__ == "__main__":
    run_global_system()

