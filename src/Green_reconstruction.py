import numpy as np
from scipy.special import roots_legendre
import torch

def Phi(t):
    """
    Computes the Phi function.

    Parameters:
    - t: The Gauss-Legendre node.

    Returns:
    - The value of the Phi function at t, shape preserving.
    """
    
    # Clamp the input to avoid overflow in tan
    return torch.tan(0.5 * np.pi * t).clamp(min=1e-6, max=1e6) 

def GreenFunctionIntegrand(t, tau, a, b, eps, gam, beta):
    """
    Computes the integrand of the Green's function.

    Parameters:
    - tau:  The imaginary time variable of the Green's function. Shape: (1, 1, N_tau, 1).
    - t:    The Gauss-Legendre node. Shape: (1, 1, 1, N_t).
    - a + ib:  The residues. Shape: (B, M, 1).
    - eps + igam: The poles. Shape: (B, M, 1).
    - beta: The inverse temperature. Scalar (1,) tensor.

    Returns:
    - The value of the integrand at t and tau.
    """
    
    # # Define numerator
    # num = a - b * Phi(t)  # Shape: (B, M, 1, N_t)
    
    # # Build stable denominator
    # arg1 = tau * (eps + gam * Phi(t))
    # arg2 = (tau - beta) * (eps + gam * Phi(t))
    # max_exp = 20.0
    # # Clamp the exponentials to avoid overflow
    # arg1 = torch.clamp(arg1, max=max_exp, min=-max_exp)
    # arg2 = torch.clamp(arg2, max=max_exp, min=-max_exp)
    # denom = torch.exp(arg1) + torch.exp(arg2)
    # denom.clamp(min=1e-12)  # Avoid division by zero
    
    # return 0.5 * num / denom  # Shape: (B, M, N_tau, N_t)
    
    # Map t in [-1, 1] to q in [0, 1] (quantile)
    q = 0.5 * (t + 1)
    
    # Map q in [0, 1] to energy omega using inverse CDF
    omega = eps - gam * torch.tan(np.pi * (q - 0.5))
    
    # return 0.5 * kernel_tau 
    
    num1 = 1 + (b * eps) / (a * gam)
    num2 = b * omega
    
    # Build stable denominator
    arg1 = tau * omega
    arg2 = (tau - beta) * omega
    max_exp = 50.0
    # Clamp the exponentials to avoid overflow
    arg1 = torch.clamp(arg1, max=max_exp, min=-max_exp)
    arg2 = torch.clamp(arg2, max=max_exp, min=-max_exp)
    # Avoid division by zero
    denom = torch.exp(arg1) + torch.exp(arg2)
    # denom.clamp(min=1e-12)  # Avoid division by zero

    kernel_tau = (num1 * denom ** (-1)) - (num2 * (a * gam * denom) ** (-1))
    
    return 0.5 * kernel_tau
    
def GaussLegendreQuadrature(batch_size, num_poles, N_t, tau_array, poles, residues, beta):
    """
    Reconstruct G(τ) from complex poles and residues using Gauss-Legendre quadrature.

    Args:
        - N_t: Number of Gauss-Legendre nodes/quadrature points.
        - tau: Imaginary time variable.
        - a, b: Residues (real and imaginary parts).
        - eps, gam: Poles (real and imaginary parts).
        - beta: Inverse temperature.

    Returns:
        G_tau: (B, N_tau) real tensor.
    """
    # Get the nodes and weights
    device = tau_array.device
    dtype = tau_array.dtype
    nodes, weights = roots_legendre(N_t)  # Both are numpy arrays of shape (N_t,)
    
    t_nodes = torch.tensor(nodes, dtype=dtype, device=device).reshape(1, 1, 1, -1)  # Shape: (1, 1, 1, N_t)
    t_weights = torch.tensor(weights, dtype=dtype, device=device).reshape(1, 1, 1, -1)  # Shape: (1, 1, 1, N_t)
    
    # Extract real and imaginary parts of poles and residues
    a = residues.real.reshape(-1, num_poles, 1, 1)   # Real part of the residues (batch_size, M, 1, 1)
    b = residues.imag.reshape(-1, num_poles, 1, 1)   # Imaginary part of the residues (batch_size, num_poles, 1, 1)
    eps = poles.real.reshape(-1, num_poles, 1, 1)    # Real part of the poles (batch_size, num_poles, 1, 1)
    gam = poles.imag.reshape(-1, num_poles, 1, 1)    # Imaginary part of the poles (batch_size, num_poles, 1, 1)
    
    if __name__== "__main__":
        print(f"\na shape: {a.shape}")
        print(f"\nb shape: {b.shape}")
        print(f"\neps shape: {eps.shape}")
        print(f"\ngam shape: {gam.shape}")
        print(f"\na: {a}")
        print(f"\nb: {b}")
        print(f"\neps: {eps}")
        print(f"\ngam: {gam}", end='\n\n')
    
    # Reshape tau_array to match the shape of the integrand
    tau_array = tau_array.reshape(1, 1, -1, 1)                     # Shape: (batch_size, 1, N_tau, 1)
    
    # Compute the Green's function
    # Apply quadrature over t (N_t nodes)
    integrand = GreenFunctionIntegrand(t_nodes, tau_array, a, b, eps, gam, beta)  # Shape: (B, M, N_tau, N_t)
    
    if __name__ == "__main__":
        print(f"\nintegrand shape: {integrand.shape}")
    
    # Apply the weights to the integrand and sum over the Gauss-Legendre nodes
    integral_per_pole = integrand * t_weights  # Shape: (B, M, N_tau, N_t)
    
    if __name__ == "__main__":
        print(f"\nintegral_per_pole shape: {integral_per_pole.shape}")
    
    # Sum over the t-axis (N_t nodes) and the residues (M poles)
    G_tau = integral_per_pole.sum(dim=-1)        # sum over N_t → (B, M, N_tau)
    G_tau = G_tau.sum(dim=1)                     # sum over poles → (B, N_tau)
    
    return G_tau


# Example usage

if __name__ == "__main__":
    
    # Define the parameters

    # Batch size and number of poles 
    BATCH_SIZE, NUM_POLES = 1, 1

    # Number of time slices
    N_TAU = 201

    # Number of Gauss-Legendre nodes/quadrature points
    N_t = 200

    BETA = torch.tensor(10.0)
    TAU_ARRAY = torch.linspace(
            start=0,
            end=BETA,
            steps=N_TAU,
            dtype=torch.float32,
        )
    
    # Complex poles shape: (batch_size, num_poles)
    poles = torch.tensor([0 - 0.5j], dtype=torch.complex64)
    
    # Complex residues shape: (batch_size, num_poles)
    residues = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
    
    # Compute G(τ)
    G_tau = GaussLegendreQuadrature(batch_size=BATCH_SIZE, num_poles=NUM_POLES, N_t=N_t, tau_array=TAU_ARRAY, poles=poles, residues=residues, beta=BETA)

    # Print the resulting Green's function
    print("Green's Function G(τ):")
    print(G_tau)