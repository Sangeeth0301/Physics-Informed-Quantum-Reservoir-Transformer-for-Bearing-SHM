import torch
import torch.nn as nn

class ContinuousNeuralODE(nn.Module):
    """
    Q1-Grade Continuous Neural ODE using 4th-Order Runge-Kutta (RK4).
    Unlike discrete MLPs, this models the true continuous vector field layer:
    dz/dt = f_theta(z)
    """
    def __init__(self, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, z):
        """
        Outputs the continuous rate of change dz/dt.
        """
        return self.net(z)

    def integrate_rk4(self, z0, dt):
        """
        Explicit true 4th-Order Runge-Kutta Integrator.
        Eliminates the 'trivial' Euler method, enabling highly stable 
        and physically rigorous numerical approximation.
        """
        k1 = self.forward(z0)
        k2 = self.forward(z0 + 0.5 * dt * k1)
        k3 = self.forward(z0 + 0.5 * dt * k2)
        k4 = self.forward(z0 + dt * k3)
        return z0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
