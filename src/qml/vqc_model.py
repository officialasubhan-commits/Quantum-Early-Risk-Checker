import os
import time
from typing import Any
import numpy as np
from numpy.typing import NDArray

class VariationalQuantumClassifier:
    """
    Statevector Variational Quantum Classifier (VQC) simulator:
    - Qubits: 6 (matching 6 PCA quantum features)
    - Feature Map: Angle encoding Ry(pi * tanh(x_i))
    - Variational Circuit: 2 layers of Ry(theta) and Rz(phi) rotations with CNOT ring entanglement
    - Measurement: Pauli-Z expectation value on Qubit 0 (<Z_0>) mapped to probability via sigmoid
    - Parameters: 24 quantum circuit angles + 2 classical output weights (26 total parameters)
    """
    def __init__(self, n_qubits: int = 6, n_layers: int = 2, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 1 << n_qubits  # 2^6 = 64
        
        np.random.seed(seed)
        # 2 layers * 6 qubits * 2 rotations (Ry, Rz) = 24 parameters
        self.weights_ry = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        self.weights_rz = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        self.bias = 0.0
        self.scale = 1.0

    def _ry(self, theta: float) -> np.ndarray:
        cos_t = np.cos(theta / 2.0)
        sin_t = np.sin(theta / 2.0)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.complex128)

    def _rz(self, phi: float) -> np.ndarray:
        exp_neg = np.exp(-1j * phi / 2.0)
        exp_pos = np.exp(1j * phi / 2.0)
        return np.array([[exp_neg, 0.0], [0.0, exp_pos]], dtype=np.complex128)

    def _apply_gate_1q(self, state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
        shape = [2] * self.n_qubits
        state_tensor = state.reshape(shape)
        # Apply gate on specified qubit axis
        state_tensor = np.tensordot(gate, state_tensor, axes=([1], [qubit]))
        # Move axis back to original position
        state_tensor = np.moveaxis(state_tensor, 0, qubit)
        return state_tensor.ravel()

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        shape = [2] * self.n_qubits
        state_tensor = state.reshape(shape).copy()
        
        # Swap target index slicing where control qubit == 1
        slices_control_1 = [slice(None)] * self.n_qubits
        slices_control_1[control] = 1  # type: ignore[call-overload]
        
        slices_target_0 = list(slices_control_1)
        slices_target_0[target] = 0  # type: ignore[call-overload]
        
        slices_target_1 = list(slices_control_1)
        slices_target_1[target] = 1  # type: ignore[call-overload]
        
        tmp = state_tensor[tuple(slices_target_0)].copy()
        state_tensor[tuple(slices_target_0)] = state_tensor[tuple(slices_target_1)]
        state_tensor[tuple(slices_target_1)] = tmp
        
        return state_tensor.ravel()

    def execute_circuit(self, x: np.ndarray) -> float:
        """
        Executes the quantum circuit for a single 6-dimensional feature vector x:
        1. Initialize |000000>
        2. Feature Encoding: Ry(pi * tanh(x_i))
        3. Variational Layers: Ry + Rz + CNOT Ring Entanglement
        4. Measure <Z_0>
        """
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0 + 0.0j  # |0...0>
        
        # 1. Feature Map / Angle Encoding
        for i in range(self.n_qubits):
            val = np.pi * np.tanh(x[i])
            gate = self._ry(val)
            state = self._apply_gate_1q(state, gate, i)
            
        # 2. Variational Circuit Layers
        for l in range(self.n_layers):
            # Rotations
            for i in range(self.n_qubits):
                gate_y = self._ry(self.weights_ry[l, i])
                state = self._apply_gate_1q(state, gate_y, i)
                gate_z = self._rz(self.weights_rz[l, i])
                state = self._apply_gate_1q(state, gate_z, i)
                
            # CNOT Ring Entanglement
            for i in range(self.n_qubits):
                target = (i + 1) % self.n_qubits
                state = self._apply_cnot(state, control=i, target=target)
                
        # 3. Measurement: Multi-qubit average Pauli-Z expectation value
        probs = np.abs(state) ** 2
        indices = np.arange(self.dim)
        
        expval_sum = 0.0
        for q in range(self.n_qubits):
            q_bits = (indices >> (self.n_qubits - 1 - q)) & 1
            z_vals = np.where(q_bits == 0, 1.0, -1.0)
            expval_sum += np.sum(probs * z_vals)
            
        expval_avg = float(expval_sum / self.n_qubits)
        return expval_avg

    def predict_proba_sample(self, x: Any) -> float:
        expval_avg = self.execute_circuit(x)
        logit = self.scale * expval_avg + self.bias
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(prob)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        probs = np.array([self.predict_proba_sample(x) for x in X], dtype=np.float64)
        return probs

    def predict(self, X: Any, threshold: float = 0.5) -> NDArray[np.int_]:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def fit(self, X_train: Any, y_train: Any, epochs: int = 40, lr: float = 0.1, batch_size: int = 256):
        """
        Trains the VQC using batch gradient descent / parameter shift on batch loss.
        """
        n_samples = len(X_train)
        loss_history = []
        
        print(f"\n[QML TRAINING] Starting VQC Training: {self.n_qubits} qubits, {self.n_layers} layers, 24 quantum params", flush=True)
        print(f" -> Training Samples: {n_samples:,} | Epochs: {epochs} | LR: {lr}", flush=True)
        
        # Calculate balanced class weights for BCE loss (smoothed)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        pos_weight = min(2.0, float(n_neg / n_pos)) if n_pos > 0 else 1.0
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Mini-batch sampling
            indices = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            
            # Subsample for gradient estimation
            sub_idx = np.random.choice(len(X_batch), size=min(64, len(X_batch)), replace=False)
            X_sub = X_batch[sub_idx]
            y_sub = y_batch[sub_idx]
            weights_sub = np.where(y_sub == 1, pos_weight, 1.0)
            
            def compute_batch_loss(probs_in):
                eps = 1e-12
                p_c = np.clip(probs_in, eps, 1.0 - eps)
                return -np.mean(weights_sub * (y_sub * np.log(p_c) + (1.0 - y_sub) * np.log(1.0 - p_c)))
            
            p_sub = self.predict_proba(X_sub)
            loss = compute_batch_loss(p_sub)
            loss_history.append(float(loss))
            
            # Classical Affine Gradients (scale & bias)
            eps = 1e-12
            p_c = np.clip(p_sub, eps, 1.0 - eps)
            dloss_dlogit = (p_c - y_sub) * weights_sub
            expvals_sub = np.array([self.execute_circuit(x) for x in X_sub])
            
            dloss_dbias = float(np.mean(dloss_dlogit))
            dloss_dscale = float(np.mean(dloss_dlogit * expvals_sub))
            
            self.bias -= lr * dloss_dbias
            self.scale -= lr * dloss_dscale
            
            # Direct Finite Difference Parameter Shift Gradients on Batch Loss
            delta = 0.05
            grad_ry = np.zeros_like(self.weights_ry)
            grad_rz = np.zeros_like(self.weights_rz)
            
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    # Shift Ry
                    self.weights_ry[l, q] += delta
                    loss_plus = compute_batch_loss(self.predict_proba(X_sub))
                    self.weights_ry[l, q] -= 2 * delta
                    loss_minus = compute_batch_loss(self.predict_proba(X_sub))
                    self.weights_ry[l, q] += delta # restore
                    
                    grad_ry[l, q] = (loss_plus - loss_minus) / (2.0 * delta)
                    
                    # Shift Rz
                    self.weights_rz[l, q] += delta
                    loss_plus = compute_batch_loss(self.predict_proba(X_sub))
                    self.weights_rz[l, q] -= 2 * delta
                    loss_minus = compute_batch_loss(self.predict_proba(X_sub))
                    self.weights_rz[l, q] += delta # restore
                    
                    grad_rz[l, q] = (loss_plus - loss_minus) / (2.0 * delta)
            
            # Update Quantum Rotation Parameters
            self.weights_ry -= lr * grad_ry
            self.weights_rz -= lr * grad_rz
            
            if epoch % 10 == 0 or epoch == 1:
                print(f" -> Epoch {epoch:02d}/{epochs} | Loss: {loss:.4f} | Bias: {self.bias:.3f} | Scale: {self.scale:.3f}", flush=True)
                
        total_time = time.time() - start_time
        print(f" -> VQC Training Completed in {total_time:.2f}s", flush=True)
        return loss_history
