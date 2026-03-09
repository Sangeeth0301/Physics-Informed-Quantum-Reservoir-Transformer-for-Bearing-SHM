
import torch
import torch.nn as nn

class LearnedFusionNetwork(nn.Module):
    """
    UPGRADE 2: Non-linear MLP for Fusion.
    Learns to combine Physics, Quantum, Transformer, and Koopman scores.
    """
    def __init__(self, n_inputs=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # Squashes Global Instability Score (SI) between 0 and 1
        )

    def forward(self, x):
        """
        x: [batch, 4] -> [Koopman, Quantum, Transformer, Physics] scores
        """
        return self.net(x)

    def train_fusion(self, scores_matrix, labels, epochs=100, lr=0.01):
        """
        Semi-supervised training: If labels are available (Healthy vs Fault),
        learn the non-linear boundary for SI.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        X = torch.tensor(scores_matrix, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"LearnedFusionNetwork: Non-linear weights optimized. Final Loss: {loss.item():.6f}")
