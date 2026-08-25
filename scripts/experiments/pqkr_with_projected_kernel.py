import pennylane as qml
import numpy as np

class PQKR:
    """
    Projected Quantum Kernel Reservoir (PQKR) Module
    Implements a deterministic quantum reservoir for computing fidelity kernels.
    """
    def __init__(self, n_qubits=4, n_layers=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        
        # CPU only, statevector mode REQUIRED
        self.dev = qml.device('default.qubit', wires=self.n_qubits, shots=None)
        
        # Fixed random parameters seeded deterministically (NOT trainable)
        rng = np.random.default_rng(seed)
        self.theta_rx = rng.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits))
        self.theta_ry = rng.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits))
        self.theta_rz = rng.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits))
        
        def _apply_reservoir(features):
            # Encoding Layer (deterministic angle encoding)
            for i in range(self.n_qubits):
                qml.RX(features[i], wires=i)

            # Reservoir Layers
            for l in range(self.n_layers):
                # Random Pauli Rotations
                for i in range(self.n_qubits):
                    qml.RX(self.theta_rx[l, i], wires=i)
                    qml.RY(self.theta_ry[l, i], wires=i)
                    qml.RZ(self.theta_rz[l, i], wires=i)

                # Entanglement (Ladder CNOT + ring closure)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

        @qml.qnode(self.dev, interface="autograd")
        def _circuit(features):
            _apply_reservoir(features)
            return qml.state()

        self.circuit = _circuit

        # --- Projected-kernel readout (Huang et al. 2021 "Power of data in
        # quantum machine learning") ---
        # Instead of returning the full 2^n statevector and comparing by
        # fidelity (which is the construction shown to exponentially
        # concentrate for expressive/entangled circuits, Thanasilp et al.
        # 2024), we measure local single-qubit Pauli expectation values
        # <X_i>, <Y_i>, <Z_i> from the SAME reservoir circuit and build the
        # kernel on that 3*n_qubits-dimensional real, classically-comparable
        # feature vector instead.
        @qml.qnode(self.dev, interface="autograd")
        def _circuit_projected(features):
            _apply_reservoir(features)
            obs = []
            for i in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliX(i)))
            for i in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliY(i)))
            for i in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliZ(i)))
            return obs

        self.circuit_projected = _circuit_projected

    def get_state(self, features):
        """Returns the full statevector for the given features."""
        return self.circuit(features)

    def get_projected_features(self, features):
        """Returns the local Pauli-expectation feature vector (3*n_qubits,)
        used by the projected quantum kernel readout."""
        return np.array(self.circuit_projected(features), dtype=float)

    def compute_kernel(self, X1, X2):
        """
        Compute fidelity kernel K(i,j) = |<psi_i | psi_j>|^2
        Memory safe, vectorized over statevectors.
        """
        states1 = np.array([self.get_state(x) for x in X1])
        if X1 is X2:
            states2 = states1
        else:
            states2 = np.array([self.get_state(x) for x in X2])

        # Inner products: <psi_i | psi_j>
        inner_prods = states1 @ states2.conj().T

        # Fidelity kernel
        fidelity = np.abs(inner_prods) ** 2
        return fidelity

    def compute_projected_kernel(self, X1, X2, gamma=None):
        """
        Projected quantum kernel (Huang et al. 2021):
            k^PQ(x_i, x_j) = exp( -gamma * sum_k || rho_k(x_i) - rho_k(x_j) ||^2_F )
        approximated here via local Pauli-expectation vectors instead of full
        reduced density matrices (equivalent information for single-qubit
        marginals up to the identity component).
        """
        feats1 = np.array([self.get_projected_features(x) for x in X1])
        if X1 is X2:
            feats2 = feats1
        else:
            feats2 = np.array([self.get_projected_features(x) for x in X2])

        if gamma is None:
            gamma = 1.0 / feats1.shape[1]

        sq_dists = (
            np.sum(feats1 ** 2, axis=1)[:, None]
            + np.sum(feats2 ** 2, axis=1)[None, :]
            - 2 * feats1 @ feats2.T
        )
        sq_dists = np.clip(sq_dists, 0, None)
        return np.exp(-gamma * sq_dists)
