import numpy as np

# Feature display mapping and clinical interpretations
CLINICAL_FEATURE_DESCRIPTIONS = {
    "BMI": "Body Mass Index (BMI)",
    "HighBP": "High Blood Pressure Diagnosis",
    "HighChol": "High Blood Cholesterol Diagnosis",
    "GenHlth": "General Health Assessment (1=Excellent, 5=Poor)",
    "Age": "Age Category (1=18-24, 13=80+)",
    "PhysHlth": "Physical Illness Days in Past Month",
    "MentHlth": "Mental Unwellness Days in Past Month",
    "DiffWalk": "Difficulty Walking or Climbing Stairs",
    "HeartDiseaseorAttack": "History of Coronary Heart Disease/Myocardial Infarction",
    "Stroke": "History of Stroke",
    "Smoker": "Tobacco Smoking History",
    "PhysActivity": "Physical Activity Level",
    "Fruits": "Daily Fruit Consumption",
    "Veggies": "Daily Vegetable Consumption",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Healthcare Coverage Status",
    "NoDocbcCost": "Inability to See Doctor Due to Cost",
    "Sex": "Biological Sex",
    "Education": "Education Level",
    "Income": "Income Bracket",
    "CholCheck": "Cholesterol Check within 5 Years"
}

def explain_patient_prediction(hybrid_model, sample_vector: np.ndarray, feature_names: list, patient_id: str = "PATIENT_001") -> dict:
    """
    Computes local feature contributions for an individual patient sample
    and generates a human-readable clinical explanation narrative.
    """
    # 1. Model Risk Prediction
    risk_prob = float(hybrid_model.predict_proba(sample_vector.reshape(1, -1))[0])
    is_high_risk = risk_prob >= hybrid_model.optimal_threshold
    risk_label = "Elevated Diabetes/Prediabetes Risk" if is_high_risk else "Low Diabetes Risk"
    
    # 2. Local Feature Attributions (Sample feature value * Global hybrid feature weight)
    rf_importances = hybrid_model.classical_model.feature_importances_
    
    # Feature deviation from baseline
    feature_abs_vals = np.abs(sample_vector)
    local_weights = feature_abs_vals * rf_importances
    if np.sum(local_weights) > 0:
        local_weights /= np.sum(local_weights)
        
    contributions = []
    for i, name in enumerate(feature_names):
        raw_val = float(sample_vector[i])
        contributions.append({
            "feature": name,
            "display_name": CLINICAL_FEATURE_DESCRIPTIONS.get(name, name),
            "feature_value": round(raw_val, 2),
            "contribution_score": float(local_weights[i])
        })
        
    # Sort by local contribution descending
    contributions = sorted(contributions, key=lambda x: x["contribution_score"], reverse=True)
    top_3 = contributions[:3]
    
    # 3. Generate Human-Readable Clinical Explanation Narrative
    narrative_lines = [
        f"Diagnostic Assessment for {patient_id}:",
        f"• Risk Prediction: {risk_label} ({risk_prob*100:.1f}% estimated probability).",
        f"• Key Clinical Risk Factor Contributions:"
    ]
    
    for item in top_3:
        narrative_lines.append(f"  - {item['display_name']} (Value: {item['feature_value']}) contributed {item['contribution_score']*100:.1f}% to this prediction.")
        
    if is_high_risk:
        narrative_lines.append("• Recommendation: Patient exhibits strong clinical indicators associated with prediabetes/diabetes risk. Consult healthcare provider for HbA1c screening.")
    else:
        narrative_lines.append("• Recommendation: Patient risk profile remains within routine monitoring thresholds. Maintain healthy diet and active lifestyle.")
        
    human_narrative = "\n".join(narrative_lines)
    
    return {
        "patient_id": patient_id,
        "risk_probability": round(risk_prob, 4),
        "predicted_label": risk_label,
        "is_high_risk": is_high_risk,
        "top_contributing_features": top_3,
        "full_feature_contributions": contributions,
        "clinical_narrative": human_narrative
    }
