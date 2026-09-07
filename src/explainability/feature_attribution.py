import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def compute_global_feature_attributions(classical_model, pca_reducer, X_val, y_val, feature_names: list) -> dict:
    """
    Computes mathematically rigorous global feature attributions:
    1. Classical Attributions: Random Forest feature importances + Permutation Importance.
    2. Quantum Attributions: PCA Component Loading Projection (Mapping 6 quantum features back to 21 original features).
    3. Hybrid Attributions: Blended ensemble weighted feature attributions (0.83 Classical + 0.17 Quantum).
    """
    # 1. Classical Random Forest Feature Importances
    rf_importances = classical_model.feature_importances_
    
    # Permutation importance on validation set (sample of 2,000 for speed)
    val_sample_size = min(2000, len(X_val))
    perm_res = permutation_importance(
        classical_model,
        X_val[:val_sample_size],
        y_val[:val_sample_size],
        n_repeats=5,
        random_state=42
    )
    perm_importances = perm_res.importances_mean
    perm_importances = np.maximum(0, perm_importances)
    if np.sum(perm_importances) > 0:
        perm_importances /= np.sum(perm_importances)
        
    classical_combined_importance = 0.5 * (rf_importances / np.sum(rf_importances)) + 0.5 * perm_importances
    classical_combined_importance /= np.sum(classical_combined_importance)
    
    # 2. Quantum Feature Attributions via PCA Loading Matrix Projection
    # PCA components shape: (6, 21) -> 6 quantum features x 21 input features
    pca_loadings = np.abs(pca_reducer.components_)  # (6, 21)
    explained_var = pca_reducer.explained_variance_ratio_  # (6,)
    
    # Weight loading vectors by explained variance ratio of each principal component
    weighted_pca_loadings = np.dot(explained_var, pca_loadings)  # (21,)
    quantum_importance = weighted_pca_loadings / np.sum(weighted_pca_loadings)
    
    # 3. Hybrid Ensemble Feature Importance (0.830 Classical + 0.170 Quantum)
    w_class = 0.830
    w_quant = 0.170
    hybrid_importance = w_class * classical_combined_importance + w_quant * quantum_importance
    hybrid_importance /= np.sum(hybrid_importance)
    
    # Build clean structured dictionary
    attributions = []
    for i, name in enumerate(feature_names):
        attributions.append({
            "feature": name,
            "classical_importance": float(classical_combined_importance[i]),
            "quantum_importance": float(quantum_importance[i]),
            "hybrid_importance": float(hybrid_importance[i])
        })
        
    # Sort by hybrid importance descending
    attributions = sorted(attributions, key=lambda x: x["hybrid_importance"], reverse=True)
    
    return {
        "feature_attributions": attributions,
        "top_features": [a["feature"] for a in attributions[:5]]
    }
