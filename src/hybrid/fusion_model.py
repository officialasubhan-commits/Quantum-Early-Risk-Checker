from typing import Any
import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from src.qml.vqc_model import VariationalQuantumClassifier

class HybridEnsembleClassifier:
    """
    SIH26139 Hybrid Classical-Quantum Ensemble Classifier:
    - Classical Branch: Random Forest baseline (21 features)
    - Quantum Branch: 6-Qubit Variational Quantum Classifier (6 PCA features)
    - Fusion Strategy: Optimal Probability-Weighted Meta-Classifier fitted on Validation Set
    """
    def __init__(self, classical_model: Any, qml_model: Any, pca_reducer: Any):
        self.classical_model = classical_model
        self.qml_model = qml_model
        self.pca_reducer = pca_reducer
        
        self.meta_classifier = LogisticRegression(class_weight="balanced", random_state=42)
        self.optimal_weight = 0.85
        self.optimal_threshold = 0.50

    def fit_fusion(self, X_val: Any, y_val: Any):
        """
        Fits fusion meta-classifier and optimal probability weighting strictly on Validation Data.
        Zero exposure to Held-Out Test Set.
        """
        print("\n[HYBRID FUSION] Extracting validation probability predictions...", flush=True)
        # 1. Classical Probabilities
        P_class_val = self.classical_model.predict_proba(X_val)[:, 1]
        
        # 2. Quantum Probabilities (6 PCA Features)
        X_val_q = self.pca_reducer.transform(X_val)
        P_quantum_val = self.qml_model.predict_proba(X_val_q)
        
        # 3. Fit Meta-Classifier on validation probability pair [P_class, P_quantum]
        P_stack_val: Any = np.column_stack([P_class_val, P_quantum_val])
        self.meta_classifier.fit(P_stack_val, y_val)
        
        # 4. Search optimal linear blending weight and decision threshold on validation set
        best_f1 = -1.0
        best_w = 0.85
        best_thresh = 0.50
        
        for w in np.linspace(0.50, 0.99, 50):
            P_blend = w * P_class_val + (1.0 - w) * P_quantum_val
            for thresh in np.linspace(0.30, 0.70, 41):
                y_pred = (P_blend >= thresh).astype(int)
                score = f1_score(y_val, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_w = float(w)
                    best_thresh = float(thresh)
                    
        self.optimal_weight = best_w
        self.optimal_threshold = best_thresh
        
        print(f" -> Validation Meta-Classifier Trained.", flush=True)
        print(f" -> Optimal Blend Weights: {best_w:.3f} Classical + {1.0-best_w:.3f} Quantum", flush=True)
        print(f" -> Optimal Decision Threshold: {best_thresh:.3f} (Validation F1: {best_f1:.4f})", flush=True)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        """
        Computes hybrid probability predictions for input feature matrix X.
        """
        # Classical probabilities
        P_class = self.classical_model.predict_proba(X)[:, 1]
        
        # Quantum probabilities
        X_q = self.pca_reducer.transform(X)
        P_quantum = self.qml_model.predict_proba(X_q)
        
        # Meta-classifier stack probabilities
        P_stack: Any = np.column_stack([P_class, P_quantum])
        P_meta = self.meta_classifier.predict_proba(P_stack)[:, 1]
        
        # Blended ensemble probabilities
        P_hybrid = 0.5 * P_meta + 0.5 * (self.optimal_weight * P_class + (1.0 - self.optimal_weight) * P_quantum)
        return P_hybrid

    def predict(self, X: Any) -> NDArray[np.int_]:
        """
        Predicts binary classes using calibrated optimal threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= self.optimal_threshold).astype(int)
