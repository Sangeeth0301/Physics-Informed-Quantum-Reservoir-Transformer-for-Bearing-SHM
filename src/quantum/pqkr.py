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
        
        @qml.qnode(self.dev, interface="autograd")
        def _circuit(features):
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
                
            return qml.state()
            
        self.circuit = _circuit

    def get_state(self, features):
        """Returns the full statevector for the given features."""
        return self.circuit(features)
    
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
