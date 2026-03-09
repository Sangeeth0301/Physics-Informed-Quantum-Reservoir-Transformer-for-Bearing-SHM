
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class QuantumReadoutSVM:
    """
    Expert One-Class SVM Quantum Readout Layer.
    Learns the 'Healthy' hypersphere boundary in Quantum Hilbert Space.
    """
    def __init__(self, kernel='precomputed', nu=0.05):
        self.model = OneClassSVM(kernel=kernel, nu=nu)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, K):
        """
        Fit ONLY on the healthy baseline data.
        If kernel is precomputed, K should be [n_healthy, n_healthy].
        """
        self.model.fit(K)
        self.is_fitted = True
        print("QuantumReadoutSVM: Boundary learned successfully.")

    def score(self, K_test):
        """
        Inference: Compute continuous divergence scores.
        K_test: Kernel matrix between test points and training points [m_test, n_healthy].
        Outputs: Continuous score [0, 1] where 1 is highly anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted on healthy data before scoring.")

        # Distance from separating hyperplane (signed)
        # Decision function: > 0 for inliers, < 0 for outliers.
        dist = self.model.decision_function(K_test)

        # Map to Anomaly Score: Invert distance (Inliers high -> Score low)
        anomaly_score = -dist

        # Min-Max Normalization to [0, 1] reference
        # We use a robust clamp/sigmoid or linear scaling
        score_min = np.min(anomaly_score)
        score_max = np.max(anomaly_score)
        
        if score_max > score_min:
            normalized_score = (anomaly_score - score_min) / (score_max - score_min)
        else:
            normalized_score = np.zeros_like(anomaly_score)

        return normalized_score

# Usage Example Demonstrator
if __name__ == "__main__":
    # Simulate a Quantum Kernel Matrix (Fidelity)
    n_healthy = 50
    n_fault = 20
    
    # Healthy-Healthy Kernel (Strong correlation, values close to 1)
    K_healthy = np.random.uniform(0.7, 1.0, (n_healthy, n_healthy))
    
    # Fault-Healthy Kernel (Weak correlation, values close to 0)
    K_test_H = np.random.uniform(0.6, 0.9, (10, n_healthy))
    K_test_F = np.random.uniform(0.1, 0.4, (n_fault, n_healthy))
    
    readout = QuantumReadoutSVM(kernel='precomputed')
    readout.fit(K_healthy)
    
    scores_h = readout.score(K_test_H)
    scores_f = readout.score(K_test_F)
    
    print(f"Healthy Mean Score: {scores_h.mean():.4f}")
    print(f"Fault Mean Score: {scores_f.mean():.4f}")
