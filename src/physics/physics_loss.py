import torch

def compute_physics_loss_pinn(model, z_t, c=0.1, k1=0.05, k2=0.05):
    """
    Q1-Grade Physics-Informed Neural Network (PINN) Loss using Autograd.
    This completely replaces the trivial finite-difference approach.
    By using PyTorch's computational graph (autograd), we calculate the EXACT
    continuous physical accelerations directly from the Neural ODE vector field.

    Full Modified Jeffcott Rotor Equation:
    x_ddot + c*x_dot + k1*x + k2*|x|^(1.5)*sign(x) = 0
    """
    # Detach and require grad to query the exact continuous gradient
    z = z_t.clone().detach().requires_grad_(True)
    
    # First derivative from the ODE continuous vector field
    z_dot = model(z) # shape: [batch, latent_dim]
    
    # We assign physical displacement `x` to the 1st latent dimension
    x = z[:, 0:1]         # shape: [batch, 1]
    x_dot = z_dot[:, 0:1] # shape: [batch, 1]
    
    # Exact second derivative via PyTorch Autograd (Jacobian-Vector Product)
    # Since x_dot is a function of z, dx_dot/dt = (dx_dot/dz) * (dz/dt)
    x_dot_grad = torch.autograd.grad(
        outputs=x_dot,
        inputs=z,
        grad_outputs=torch.ones_like(x_dot),
        create_graph=True,
        retain_graph=True
    )[0] # shape: [batch, latent_dim]
    
    # x_ddot = sum_i ( d_x_dot/d_z_i * z_dot_i )
    x_ddot = torch.sum(x_dot_grad * z_dot, dim=1, keepdim=True) # shape: [batch, 1]
    
    # Physics definitions (Hertzian nonlinear contact stress + Linear stiffness + Damping)
    linear_stiffness = k1 * x
    hertzian_stress = k2 * torch.pow(torch.abs(x), 1.5) * torch.sign(x)
    damping = c * x_dot
    
    # The complete PINN continuous exact residual
    r_phys = x_ddot + damping + linear_stiffness + hertzian_stress
    
    # L_phys is the MSE of violating this law of thermodynamics
    loss_phys = torch.mean(r_phys**2)
    
    return loss_phys, torch.abs(r_phys).detach().squeeze()

def get_pinn_anomaly_score(model, Z):
    """Utility to quickly score an sequence of latents without tracking gradients"""
    with torch.enable_grad():
        _, residuals = compute_physics_loss_pinn(model, Z.clone().detach())
    return residuals.numpy()
