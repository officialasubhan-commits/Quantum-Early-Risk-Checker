from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

MEDICAL_DISCLAIMER_TEXT = (
    "AI-Assisted Early Disease Risk Decision Support: "
    "This output is an automated probabilistic risk estimation designed solely for decision support. "
    "It is NOT a medical diagnosis and should not replace professional clinical evaluation by a licensed healthcare practitioner."
)

class PatientFeatures(BaseModel):
    BMI: float = Field(..., ge=10.0, le=100.0, description="Body Mass Index (BMI)", examples=[28.5])
    GenHlth: int = Field(..., ge=1, le=5, description="General Health Assessment (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)", examples=[3])
    MentHlth: float = Field(..., ge=0.0, le=30.0, description="Number of days of poor mental health in past 30 days", examples=[2.0])
    PhysHlth: float = Field(..., ge=0.0, le=30.0, description="Number of days of poor physical health in past 30 days", examples=[5.0])
    HighBP: int = Field(..., ge=0, le=1, description="High Blood Pressure Diagnosis (0=No, 1=Yes)", examples=[1])
    HighChol: int = Field(..., ge=0, le=1, description="High Blood Cholesterol Diagnosis (0=No, 1=Yes)", examples=[1])
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol check in past 5 years (0=No, 1=Yes)", examples=[1])
    Smoker: int = Field(..., ge=0, le=1, description="Smoked at least 100 cigarettes in lifetime (0=No, 1=Yes)", examples=[1])
    Stroke: int = Field(..., ge=0, le=1, description="History of Stroke (0=No, 1=Yes)", examples=[0])
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Coronary Heart Disease or Myocardial Infarction (0=No, 1=Yes)", examples=[0])
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity in past 30 days (0=No, 1=Yes)", examples=[1])
    Fruits: int = Field(..., ge=0, le=1, description="Consume fruit 1 or more times per day (0=No, 1=Yes)", examples=[1])
    Veggies: int = Field(..., ge=0, le=1, description="Consume vegetables 1 or more times per day (0=No, 1=Yes)", examples=[1])
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Heavy alcohol drinkers (adult men >= 14 drinks/week, women >= 7 drinks/week) (0=No, 1=Yes)", examples=[0])
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Has any health care coverage (0=No, 1=Yes)", examples=[1])
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Could not see doctor due to cost in past 12 months (0=No, 1=Yes)", examples=[0])
    DiffWalk: int = Field(..., ge=0, le=1, description="Serious difficulty walking or climbing stairs (0=No, 1=Yes)", examples=[0])
    Sex: int = Field(..., ge=0, le=1, description="Biological Sex (0=Female, 1=Male)", examples=[1])
    Age: int = Field(..., ge=1, le=13, description="13-level age category (1=18-24 ... 13=80+)", examples=[9])
    Education: int = Field(..., ge=1, le=6, description="Education level (1=Never attended school ... 6=College 4+ years)", examples=[5])
    Income: int = Field(..., ge=1, le=8, description="Income scale (1=LessThan$10k ... 8=$75k+)", examples=[7])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "BMI": 32.0, "GenHlth": 4, "MentHlth": 5.0, "PhysHlth": 10.0,
                "HighBP": 1, "HighChol": 1, "CholCheck": 1, "Smoker": 1,
                "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0,
                "Fruits": 0, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
                "NoDocbcCost": 0, "DiffWalk": 1, "Sex": 1, "Age": 11,
                "Education": 4, "Income": 5
            }
        }
    )

# Alias for backward compatibility
DiabetesFeatures = PatientFeatures

class PredictionRequest(BaseModel):
    patient_id: Optional[str] = Field("PATIENT_001", description="Patient Identifier or Request Tag")
    features: PatientFeatures

class GenericDiseasePredictionRequest(BaseModel):
    patient_id: Optional[str] = Field("PATIENT_001", description="Patient Identifier or Request Tag")
    disease_id: Optional[str] = Field("diabetes", description="Target Disease Identifier (e.g. diabetes, heart_disease)")
    features: Dict[str, Any] = Field(..., description="Dictionary of clinical feature key-values")

class SingleDiseasePredictionResult(BaseModel):
    disease_id: str
    disease_name: str
    model_used: str
    predicted_class: int
    predicted_label: str
    risk_probability: float
    is_high_risk: bool
    status: str = "SUCCESS"
    missing_features: List[str] = []

class MultiDiseasePredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    patient_id: str
    disease_results: List[SingleDiseasePredictionResult]
    unsupported_diseases: List[Dict[str, Any]] = []
    disclaimer: str = MEDICAL_DISCLAIMER_TEXT

class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    patient_id: str
    disease_id: str = "diabetes"
    disease_name: str = "Diabetes / Prediabetes"
    model_used: str
    predicted_class: int
    predicted_label: str
    risk_probability: float
    is_high_risk: bool
    disclaimer: str = MEDICAL_DISCLAIMER_TEXT

class FeatureContributionItem(BaseModel):
    feature: str
    display_name: str
    feature_value: float
    contribution_score: float

class ExplainabilityResponse(BaseModel):
    request_id: str
    timestamp: str
    patient_id: str
    disease_id: str = "diabetes"
    disease_name: str = "Diabetes / Prediabetes"
    model_used: str
    predicted_class: int
    predicted_label: str
    risk_probability: float
    is_high_risk: bool
    top_contributing_features: List[FeatureContributionItem]
    full_feature_contributions: List[FeatureContributionItem]
    clinical_narrative: str
    disclaimer: str = MEDICAL_DISCLAIMER_TEXT

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    registered_diseases_count: int = 1
    active_diseases_count: int = 1
    version: str

class ModelInfoResponse(BaseModel):
    baseline_model: Dict[str, Any]
    qml_model: Dict[str, Any]
    hybrid_model: Dict[str, Any]
    dataset_info: Dict[str, Any]
    medical_disclaimer: str = MEDICAL_DISCLAIMER_TEXT
