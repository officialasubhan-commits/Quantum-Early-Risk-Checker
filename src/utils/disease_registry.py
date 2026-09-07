import os
import json
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_PATH = os.path.join(BASE_DIR, "disease_registry", "registry.json")

class DiseaseRegistry:
    """
    Central Disease Registry Manager for SIH26139 Multi-Disease Platform.
    Manages metadata, required feature schemas, model artifact paths, and status 
    for all targeted medical disease models.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiseaseRegistry, cls).__new__(cls)
            cls._instance._load_registry()
        return cls._instance

    def _load_registry(self):
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.diseases = data.get("diseases", {})
        else:
            self.diseases = {}

    def get_all_diseases(self) -> Dict[str, Any]:
        return self.diseases

    def get_active_diseases(self) -> Dict[str, Any]:
        return {d_id: info for d_id, info in self.diseases.items() if info.get("status") == "ACTIVE"}

    def get_disease_info(self, disease_id: str) -> Optional[Dict[str, Any]]:
        return self.diseases.get(disease_id.lower())

    def is_disease_active(self, disease_id: str) -> bool:
        info = self.get_disease_info(disease_id)
        return bool(info and info.get("status") == "ACTIVE")

    def get_required_features(self, disease_id: str) -> List[str]:
        info = self.get_disease_info(disease_id)
        if info:
            return info.get("features", [])
        return []

    def get_model_paths(self, disease_id: str) -> Dict[str, str]:
        info = self.get_disease_info(disease_id)
        if not info:
            return {}
        paths = info.get("model_paths", {})
        abs_paths = {}
        for key, rel_path in paths.items():
            abs_paths[key] = os.path.join(BASE_DIR, rel_path)
        return abs_paths

    def validate_features_for_disease(self, disease_id: str, input_features: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validates if input feature dict contains all required features for a disease model.
        Returns (is_valid, list_of_missing_features). Never invents or substitutes missing values.
        """
        required = self.get_required_features(disease_id)
        if not required:
            return False, ["Disease model feature schema not defined or inactive"]
        
        missing = [f for f in required if f not in input_features or input_features[f] is None]
        return (len(missing) == 0, missing)

    def register_disease_model(self, disease_id: str, name: str, category: str, description: str,
                               dataset_name: str, target_variable: str, features: List[str],
                               model_paths: Dict[str, str], qml_config: Optional[Dict[str, Any]] = None,
                               symptoms: Optional[str] = None, risk_factors: Optional[str] = None,
                               data_modality: str = "Tabular Health Indicators", status: str = "ACTIVE"):
        """
        Registers or updates a disease model entry in registry.json.
        """
        self.diseases[disease_id.lower()] = {
            "id": disease_id.lower(),
            "name": name,
            "category": category,
            "description": description,
            "symptoms": symptoms or "Clinical symptoms and health risk biomarkers.",
            "risk_factors": risk_factors or "General clinical and demographic risk factors.",
            "dataset_name": dataset_name,
            "target_variable": target_variable,
            "data_modality": data_modality,
            "status": status,
            "model_version": "1.0.0",
            "features": features,
            "model_paths": model_paths,
            "qml_config": qml_config or {"n_qubits": 6, "n_layers": 2, "pca_dim": 6}
        }
        self.save_registry()

    def save_registry(self):
        os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump({"diseases": self.diseases}, f, indent=2)

