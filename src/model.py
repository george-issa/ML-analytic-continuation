# Important libraries
import torch
import torch.nn.functional as F
from torch import nn

# Define the VAE class
# Input -> Hidden -> Mu, Logvar -> Sampling -> Hidden -> Output
class VariationalAutoEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, latent_dim, num_poles):
        
        super(VariationalAutoEncoder, self).__init__()
        
        # Initalize number of poles
        self.num_poles = num_poles
        
        # Encoder
        # 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Another 1D convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Another 1D convolutional layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Flatten the input and pass through a linear layer
        self.conv_hidden1 = nn.Linear(128 * (input_dim), hidden_dim1)  # Adjusted for conv1d output size
        self.hidden1_hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # KL divergence will push the latent distribution to be close to a standard normal distribution
        self.hidden_mu = nn.Linear(hidden_dim2, latent_dim)
        self.hidden_logvar = nn.Linear(hidden_dim2, latent_dim)
        
        # Decoder shares layer between poles and residues
        self.z_projector = nn.Linear(latent_dim, hidden_dim3)
        self.second_projector = nn.Linear(hidden_dim3, hidden_dim4)
        
        # Instead of reconstructing the input, we decode z to the poles and residues
        # 2 * num_poles for complex poles and residues
        self.hidden_poles = nn.Linear(hidden_dim4, 2 * num_poles)
        self.hidden_residues = nn.Linear(hidden_dim4, 2 * num_poles)
        
    def encode(self, x):
        # q_phi(z|x)
        
        h = F.relu(self.conv1(x.unsqueeze(1))) # Add channel dimension
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        # Flatten the output of the conv layers for the linear layer
        h = h.view(h.size(0), -1)
        h = F.relu(self.conv_hidden1(h))
        h = F.relu(self.hidden1_hidden2(h))
        
        # No activation function for mu and logvar!
        mu = self.hidden_mu(h)
        logvar = self.hidden_logvar(h)
        
        return mu, logvar
    
    def sampling(self, mu, logvar):
    
        stdv = torch.exp(0.5 * logvar)
        eps = torch.randn_like(stdv)
        
        return eps.mul(stdv).add_(mu)
    
    def extract_poles_and_residues(self, z):
        
        """
        Given latent z, extract poles and residues.
        """
        
        B = z.shape[0]
        
        h = F.relu(self.z_projector(z))
        
        # Additional constraints on poles and residues can be added here, 
        # e.g., ensuring poles are within a certain range using tanh or clamp,
        # or ensuring residues are positive using softplus or abs
        
        h = F.relu(self.second_projector(h))
        
        poles_real, poles_imag = torch.chunk(self.hidden_poles(h), 2, dim=-1)
        poles_imag = -1 * torch.abs(poles_imag)  # Ensure poles are in the lower half-plane
        
        residues_real, residues_imag = torch.chunk(self.hidden_residues(h), 2, dim=-1)
        residues_real = torch.abs(residues_real) # Ensure residues have positive real part
        
        poles = torch.complex(poles_real, poles_imag)
        residues = torch.complex(residues_real, residues_imag)
        
        return poles, residues
    
    def decode(self, z):
        # p_theta(x|z)
        
        """
        Decode z to poles and residues now instead of reconstructing the input.
        """
        
        return self.extract_poles_and_residues(z)
    
    def forward(self, x):
    
        # x has to be reshaped to (batch_size, 1, input_dim) with x.view(-1, 1, self.input_dim) before inputting
        
        mu, logvar = self.encode(x)
        
        z = self.sampling(mu, logvar)
        
        poles, residues = self.decode(z)
        
        return mu, logvar, z, poles, residues
    
if __name__ == "__main__":
    
    vae = VariationalAutoEncoder(
        
        input_dim=201, 
        hidden_dim1=256, 
        hidden_dim2=128,
        latent_dim=20, 
        hidden_dim3=64,
        hidden_dim4=32,
        num_poles=1
        )
    
    x = torch.randn(1000, 201) # Flatten since using Linear layer

    mu, logvar, z, poles, residues = vae(x)
    
    print(f'\n{poles.shape}: shape for poles')
    print(f'{residues.shape}: shape for residues')
    print(f'{mu.shape}: shape for mu')
    print(f'{logvar.shape}: shape for logvar')
    print(f'{z.shape}: shape for z\n')