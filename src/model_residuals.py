"""
    A Pytorch module that defines a VAE with a residual structure. In the forward pass, the model encodes the input to a latent space using a convolutional network. 
    It then decodes the latent space to poles and residues instead of reconstructing the input. 
    The model uses a sample method to generate a latent vector from the encoded mean and log variance.
    Additionally, it uses an RNN cell to process the latent vector and extract poles and residues iteratively.
    The residual part of the model ensures that the poles and residues are output one at a time. At each iteration, the residual Green's function is calculated 
    by G_res = G_out - G_in, where G_out is the output of the RNN cell using only one pole and residue. Then G_res is used as the input for the next iteration.
    At the end, the poles and residues are returned as complex numbers, ensuring that the poles are in the lower half-plane and residues have positive real parts.
"""

import torch
from torch import nn
import torch.nn.functional as F

from Green_reconstruction import GaussLegendreQuadrature # type: ignore

class ModelResiduals(nn.Module):
    
    def __init__(self, input_dim, 
                 hidden_dim1, hidden_dim2,
                 latent_dim, hidden_dim3, hidden_dim4,
                 num_poles, eps,
                 N_t, tau_array, beta):
        
        super(ModelResiduals, self).__init__()
        
        # Initialize all variables
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1; self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim; self.hidden_dim3 = hidden_dim3; self.hidden_dim4 = hidden_dim4
        self.num_poles = num_poles; self.eps = eps
        
        # Initialize the Gauss-Legendre quadrature for the Green's function reconstruction
        self.N_t = N_t; self.tau_array = tau_array; self.beta = beta
        
        # Encoder
        # Convolutional layers to encode the input
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Flatten the output of the conv layers for the linear layer
        self.dense1 = nn.Linear(128 * input_dim, hidden_dim1)
        self.dense2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # Dense layers for mu and logvar
        self.mu_encoder = nn.Linear(hidden_dim2, latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dim2, latent_dim)
        
        # Decoder
        self.z_projector = nn.Linear(latent_dim, hidden_dim3)
        self.second_projector = nn.Linear(hidden_dim3, hidden_dim4)
        
        # RNN cell to process the latent vector and extract poles and residues iteratively
        self.rnn_cell = nn.LSTMCell(input_size=hidden_dim4, hidden_size=hidden_dim4, bias=True)
        self.out_head = nn.Linear(hidden_dim4, 4)
        # self.out_in = nn.Linear(4, hidden_dim4)
        
    def encode(self, x):
        
        B = x.shape[0]
        
        # Apply convolutional layers
        h = F.relu(self.conv1(x.unsqueeze(1))) # Add channel dimension
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        # Flatten the output of the conv layers for the linear layer
        h = h.view(B, -1)
        h = F.relu(self.dense1(h))
        h = F.relu(self.dense2(h))
        
        # No activation function for mu and logvar!
        mu = self.mu_encoder(h)
        logvar = self.logvar_encoder(h)
        
        return mu, logvar
    
    
    def sample(self, mu, logvar):
        
        stdv = torch.exp(0.5 * logvar)
        eps = torch.randn_like(stdv)
        
        return eps.mul(stdv).add_(mu)
    
    
    def project_sample(self, z):
        
        """
        Given latent z, extract poles and residues.
        """
        
        B = z.shape[0]
        
        # Project the latent vector to hidden_dim3
        h = F.relu(self.z_projector(z))
        h = F.relu(self.second_projector(h))
        
        return h
    
    def reconstruct(self, batch_size, num_poles, N_t, tau_array, poles, residues, beta):
        
        G_out = GaussLegendreQuadrature(batch_size, num_poles, N_t, tau_array, poles, residues, beta)
        
        return G_out
    
    def forward(self, G_in):
        
        B = G_in.shape[0]
        
        poles, residues = [], []
        
        # Initialize G_res as the input for the first iteration
        G_res = G_in
        
        # Initialize a loss variable to accumulate the residuals
        loss = 0.0
        
        for _ in range(self.num_poles):
        
            # Encode the input to get mu and logvar
            mu, logvar = self.encode(G_res)
            
            # Sample from the latent space
            z = self.sample(mu, logvar)
            
            # Project the sampled latent vector 
            h = self.project_sample(z)
            
            # Initialize hidden state for RNN cell
            hx = torch.zeros(B, self.hidden_dim4)
            cx = torch.zeros(B, self.hidden_dim4)
            
            # Process the current latent vector through the RNN cell
            hx, cx = self.rnn_cell(h, (hx, cx))
            pr = self.out_head(hx)
            
            pole, residue = torch.chunk(pr, 2, dim=-1)
            pole_re, pole_im = torch.chunk(pole, 2, dim=-1)
            residue_re, residue_im = torch.chunk(residue, 2, dim=-1)
            
            # Ensure poles are in the lower half-plane and residues have positive real part
            pole_im = -1 * torch.abs(pole_im)
            residue_re = torch.abs(residue_re)
            
            # Concatenate real and imaginary parts to form complex numbers
            pole = torch.complex(pole_re, pole_im)
            residue = torch.complex(residue_re, residue_im)
            
            # Store the poles and residues
            poles.append(pole)
            residues.append(residue)
            
            # Reconstruct the Green's function using the current poles and residues
            G_out = self.reconstruct(B, 1, self.N_t, self.tau_array, pole, residue, self.beta)
            
            # Calculate the residual Green's function
            G_res -= G_out

            # Update the loss with the norm of the residual Green's function
            # This norm treats the tensors as vectors, summing the squares of all elements
            loss += torch.norm(G_res, p=2)  # L2 norm of the residual
            
            # Stop if the residual is small enough
            if torch.norm(G_res, p=2) < self.eps:
                break
            
        # Stack the poles and residues
        poles = torch.cat(poles, dim=1)         # Shape: (B, num_poles)
        residues = torch.cat(residues, dim=1)   # Shape: (B, num_poles)
        
        # Normalize the loss
        loss /= self.num_poles
        
        return mu, logvar, z, poles, residues, G_res, loss
    
if __name__ == "__main__":
    
    model = ModelResiduals(input_dim=200,
                            hidden_dim1=128, hidden_dim2=64,
                            latent_dim=32, hidden_dim3=16, hidden_dim4=8,
                            num_poles=6, eps = 1e-6,
                            N_t=200, tau_array=torch.linspace(0, 10, 200), beta=10)
    
    G_in = torch.randn(1000, 200)  # Example input
    
    # mu, logvar = model.encode(G_in)
    # z = model.sample(mu, logvar)
    
    mu, logvar, z, poles, residues, G_res, loss = model(G_in)
    
    print("\n")
    print(f"mu shape: {mu.shape}")
    print(f"logvar shape: {logvar.shape}")
    print(f"z shape: {z.shape}")
    print(f"poles shape: {poles.shape}")
    print(f"residues shape: {residues.shape}")
    print(f"G_res shape: {G_res.shape}")
    print("\n")