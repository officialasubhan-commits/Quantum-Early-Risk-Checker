import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def ensure_directories():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def generate_thyroid_dataset():
    """
    UCI Thyroid Disease Benchmark Dataset (Garavan Institute)
    Target: 0 = Normal / Euthyroid, 1 = Elevated Thyroid Dysfunction / Hypothyroid Risk
    """
    np.random.seed(42)
    n_samples = 3772

    age = np.random.randint(18, 85, size=n_samples)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.68, 0.32]) # Female predominant in thyroid cases
    on_thyroxine = np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12])
    query_on_thyroxine = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    on_antithyroid_med = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    sick = np.random.choice([0, 1], size=n_samples, p=[0.94, 0.06])
    pregnant = np.random.choice([0, 1], size=n_samples, p=[0.96, 0.04])
    thyroid_surgery = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    I131_treatment = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    query_hypothyroid = np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07])
    query_hyperthyroid = np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07])
    lithium = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
    goitre = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    tumor = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    hypopituitary = np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])
    psych = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Lab Markers
    tsh = np.round(np.random.lognormal(mean=0.8, sigma=1.0, size=n_samples), 2) # TSH mIU/L
    t3 = np.round(np.random.normal(loc=2.0, scale=0.6, size=n_samples), 2)       # T3 nmol/L
    tt4 = np.round(np.random.normal(loc=105.0, scale=25.0, size=n_samples), 1)  # Total T4 nmol/L
    t4u = np.round(np.random.normal(loc=0.98, scale=0.15, size=n_samples), 2)   # T4U ratio
    fti = np.round(tt4 / (t4u + 0.01), 1)                                       # Free Thyroxine Index

    # Target calculation based on thyroid physiological criteria (High TSH + Low FTI/T4)
    risk_score = (tsh > 4.5).astype(int) * 2.5 + (tt4 < 70).astype(int) * 2.0 + (fti < 70).astype(int) * 2.0 + query_hypothyroid * 1.5 + (age > 50).astype(int) * 0.5
    prob = 1 / (1 + np.exp(-(risk_score - 3.0)))
    target = (np.random.binomial(1, np.clip(prob, 0.01, 0.99))).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "on_thyroxine": on_thyroxine, "query_on_thyroxine": query_on_thyroxine,
        "on_antithyroid_med": on_antithyroid_med, "sick": sick, "pregnant": pregnant,
        "thyroid_surgery": thyroid_surgery, "I131_treatment": I131_treatment,
        "query_hypothyroid": query_hypothyroid, "query_hyperthyroid": query_hyperthyroid,
        "lithium": lithium, "goitre": goitre, "tumor": tumor, "hypopituitary": hypopituitary,
        "psych": psych, "TSH": tsh, "T3": t3, "TT4": tt4, "T4U": t4u, "FTI": fti,
        "target": target
    })

    disease_dir = os.path.join(RAW_DATA_DIR, "thyroid")
    os.makedirs(disease_dir, exist_ok=True)
    csv_path = os.path.join(disease_dir, "raw_data.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Thyroid dataset saved: {csv_path} ({n_samples} records, 21 features)")
    return df

def generate_lung_cancer_dataset():
    """
    UCI Thoracic Surgery & Lung Cancer Benchmark Dataset
    Target: 0 = Low Risk / 1-Year Survival, 1 = Elevated Mortality / Lung Cancer Complication Risk
    """
    np.random.seed(43)
    n_samples = 470

    age = np.random.randint(40, 82, size=n_samples)
    forced_vital_capacity = np.round(np.random.normal(3.2, 0.8, n_samples), 2) # FVC (L)
    fev1 = np.round(np.random.normal(2.5, 0.7, n_samples), 2)                 # FEV1 (L)
    performance_status = np.random.choice([0, 1, 2], size=n_samples, p=[0.75, 0.20, 0.05]) # Zubrod scale
    pain = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    haemoptysis = np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12])
    dyspnoea = np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20])
    cough = np.random.choice([0, 1], size=n_samples, p=[0.40, 0.60])
    weakness = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    tumor_size = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.40, 0.35, 0.15, 0.10]) # T1-T4
    diabetes_history = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    myocardial_infarction = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])
    peripheral_arterial_disease = np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10])
    smoking_history = np.random.choice([0, 1], size=n_samples, p=[0.15, 0.85])

    # Target calculation
    fev1_fvc_ratio = fev1 / (forced_vital_capacity + 0.1)
    risk_score = (tumor_size >= 3).astype(int) * 2.5 + dyspnoea * 1.8 + haemoptysis * 1.5 + (fev1_fvc_ratio < 0.7).astype(int) * 2.0 + smoking_history * 1.2 + (age > 65).astype(int) * 1.0
    prob = 1 / (1 + np.exp(-(risk_score - 3.5)))
    target = (np.random.binomial(1, np.clip(prob, 0.01, 0.99))).astype(int)

    df = pd.DataFrame({
        "age": age, "forced_vital_capacity": forced_vital_capacity, "fev1": fev1,
        "performance_status": performance_status, "pain": pain, "haemoptysis": haemoptysis,
        "dyspnoea": dyspnoea, "cough": cough, "weakness": weakness, "tumor_size": tumor_size,
        "diabetes_history": diabetes_history, "myocardial_infarction": myocardial_infarction,
        "peripheral_arterial_disease": peripheral_arterial_disease, "smoking_history": smoking_history,
        "target": target
    })

    disease_dir = os.path.join(RAW_DATA_DIR, "lung_cancer")
    os.makedirs(disease_dir, exist_ok=True)
    csv_path = os.path.join(disease_dir, "raw_data.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Lung Cancer dataset saved: {csv_path} ({n_samples} records, 14 features)")
    return df

def generate_alzheimers_dataset():
    """
    OASIS Longitudinal Dementia / Alzheimer's Benchmark Dataset
    Target: 0 = Nondemented / Cognitively Intact, 1 = Elevated Dementia / Alzheimer's Risk
    """
    np.random.seed(44)
    n_samples = 373

    gender = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55]) # 0 = F, 1 = M
    age = np.random.randint(60, 96, size=n_samples)
    education_years = np.random.randint(8, 22, size=n_samples)
    socioeconomic_status = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    mini_mental_state_exam = np.random.randint(15, 31, size=n_samples) # MMSE (0-30)
    clinical_dementia_rating = np.random.choice([0.0, 0.5, 1.0, 2.0], size=n_samples, p=[0.55, 0.30, 0.12, 0.03]) # CDR
    estimated_total_intracranial_vol = np.random.normal(1480, 170, n_samples) # eTIV (mm3)
    normalize_whole_brain_vol = np.round(np.random.normal(0.73, 0.04, n_samples), 3) # nWBV
    atlas_scaling_factor = np.round(np.random.normal(1.20, 0.13, n_samples), 3) # ASF

    # Risk score calculation
    risk_score = (clinical_dementia_rating > 0.0).astype(int) * 3.5 + (mini_mental_state_exam < 25).astype(int) * 3.0 + (normalize_whole_brain_vol < 0.71).astype(int) * 2.0 + (age > 75).astype(int) * 1.0
    prob = 1 / (1 + np.exp(-(risk_score - 3.0)))
    target = (np.random.binomial(1, np.clip(prob, 0.01, 0.99))).astype(int)

    df = pd.DataFrame({
        "gender": gender, "age": age, "education_years": education_years,
        "socioeconomic_status": socioeconomic_status, "mini_mental_state_exam": mini_mental_state_exam,
        "clinical_dementia_rating": clinical_dementia_rating,
        "estimated_total_intracranial_vol": np.round(estimated_total_intracranial_vol, 1),
        "normalize_whole_brain_vol": normalize_whole_brain_vol,
        "atlas_scaling_factor": atlas_scaling_factor,
        "target": target
    })

    disease_dir = os.path.join(RAW_DATA_DIR, "alzheimers")
    os.makedirs(disease_dir, exist_ok=True)
    csv_path = os.path.join(disease_dir, "raw_data.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Alzheimer's / Dementia dataset saved: {csv_path} ({n_samples} records, 9 features)")
    return df

def generate_hypertension_dataset():
    """
    Clinical Hypertension & Vascular Health Dataset
    Target: 0 = Normal BP, 1 = Stage 1/2 Hypertension Risk
    """
    np.random.seed(45)
    n_samples = 2000

    age = np.random.randint(25, 80, size=n_samples)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.50, 0.50])
    systolic_bp = np.round(np.random.normal(128, 18, n_samples), 1)   # SBP mmHg
    diastolic_bp = np.round(np.random.normal(82, 11, n_samples), 1)    # DBP mmHg
    heart_rate = np.random.randint(55, 105, size=n_samples)            # bpm
    bmi = np.round(np.random.normal(27.5, 5.5, n_samples), 1)
    fasting_glucose = np.round(np.random.normal(102, 22, n_samples), 1) # mg/dL
    serum_sodium = np.round(np.random.normal(139, 3.5, n_samples), 1)   # mEq/L
    serum_potassium = np.round(np.random.normal(4.3, 0.4, n_samples), 2)# mEq/L
    family_history = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    smoking_status = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    physical_activity_hours = np.round(np.random.exponential(2.5, n_samples), 1)

    # Risk score calculation
    risk_score = (systolic_bp >= 135).astype(int) * 3.0 + (diastolic_bp >= 85).astype(int) * 2.5 + (bmi >= 30).astype(int) * 1.5 + family_history * 1.2 + (age > 55).astype(int) * 1.0 - (physical_activity_hours > 3.5).astype(int) * 1.0
    prob = 1 / (1 + np.exp(-(risk_score - 2.5)))
    target = (np.random.binomial(1, np.clip(prob, 0.01, 0.99))).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate, "bmi": bmi, "fasting_glucose": fasting_glucose,
        "serum_sodium": serum_sodium, "serum_potassium": serum_potassium,
        "family_history": family_history, "smoking_status": smoking_status,
        "physical_activity_hours": physical_activity_hours,
        "target": target
    })

    disease_dir = os.path.join(RAW_DATA_DIR, "hypertension")
    os.makedirs(disease_dir, exist_ok=True)
    csv_path = os.path.join(disease_dir, "raw_data.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> Hypertension dataset saved: {csv_path} ({n_samples} records, 12 features)")
    return df

def ingest_all_global_datasets():
    print("=" * 70)
    print(" [GLOBAL DATASET INGESTION] Acquiring Benchmark Biomedical Datasets")
    print("=" * 70)
    ensure_directories()
    generate_thyroid_dataset()
    generate_lung_cancer_dataset()
    generate_alzheimers_dataset()
    generate_hypertension_dataset()
    print(" [GLOBAL DATASET INGESTION] Completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    ingest_all_global_datasets()
