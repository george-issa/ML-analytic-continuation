# Description: This file contains the training loop for the VAE model.
# Import the necessary libraries
import os

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import the VAE model, dataset, and Gauss-Legendre quadrature function
from model_hierarchy import VariationalAutoEncoder # type: ignore
from data_process import GreenFunctionDataset # type: ignore
from Green_reconstruction import GaussLegendreQuadrature # type: ignore


####################################################
################## INITIALIZATION ##################
####################################################


# Special ID for running
sID = 6

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 201;             HIDDEN_DIM1 = 128;                HIDDEN_DIM2 = 64;        HIDDEN_DIM3 = 64;      HIDDEN_DIM4 = 128     

NUM_POLES = 6;               LATENT_DIM = NUM_POLES * 4

NUM_EPOCHS = 50;             BATCH_SIZE = 32;                  LEARNING_RATE = 3e-03 # Kaparthy constant

# DQMC parameters
BETA = torch.tensor(10.0);   DELTA_TAU = torch.tensor(0.05);   N_TAU = (int(BETA.item() / DELTA_TAU) + 1)

TAU_ARRAY = torch.linspace(
            start=0,
            end=BETA,
            steps=N_TAU,
            dtype=torch.float32,
            device=DEVICE,
        )

# Gauss-Legendre quadrature parameters
N_t = 200;  

# Number of bootstrap samples in the input data
NBOOT = 5000

# Set noise boolean to differenciate between datasets with and without noise
NOISE = True

# Data parameters
if NOISE:
    S = 1e-04       # sigma uncertainty level
    XI = 0.5        # xi correlation length in imaginaty time

# Dataset path
MAIN_PATH = "/nfs/home/gissa/AC"
SPECTRAL_TYPE = "gaussian"

if NOISE:
    # Construct the dataset and output paths for Green's functions with noise added
    DATA_PATH = f"{MAIN_PATH}/Data/datasets/half-filled-{SPECTRAL_TYPE}/Gbins_boot_means_s{S:.0e}_xi{XI}_nboot{NBOOT}.csv"
    OUTPUT_PATH = f"{MAIN_PATH}/VAE_Library/out_{SPECTRAL_TYPE}_numpoles{NUM_POLES}_s{S:.0e}_xi{XI}-{sID}"
    
else:
    # Construct the dataset and output paths for Green's functions with no noise added
    DATA_PATH = f"{MAIN_PATH}/Data/datasets/half-filled-{SPECTRAL_TYPE}/Gbins_boot_means_no_noise.csv"
    OUTPUT_PATH = f"{MAIN_PATH}/VAE_Library/out_{SPECTRAL_TYPE}_numpoles{NUM_POLES}_no_noise-{sID}"

# Using the loading class to load the dataset
dataset = GreenFunctionDataset(file_path=DATA_PATH)

print("" * 10)
if NOISE:
    print(f"Special ID: {sID} | S: {S:.0e} | XI: {XI} | NUM_EPOCHS: {NUM_EPOCHS} | LEARNING_RATE: {LEARNING_RATE:.0e} | NUM_POLES: {NUM_POLES} | SPECTRAL_TYPE: {SPECTRAL_TYPE}")
    
else:
    print(f"Special ID: {sID} | NO NOISE | NUM_EPOCHS: {NUM_EPOCHS} | LEARNING_RATE: {LEARNING_RATE:.0e} | NUM_POLES: {NUM_POLES} | SPECTRAL_TYPE: {SPECTRAL_TYPE}")  
print("" * 10)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/model", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/losses", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/poles_residues", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/Greens", exist_ok=True)


##################################################
################ TRAIN PARAMETERS ################
##################################################


# Split the dataset into training, validation sets, and test sets
train_size = int(0.80 * len(dataset))
val_size = int(0.10 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Data loaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Print the dataset size
print(f"\nDataset size: {dataset.data.shape}")
print(f"Number of batches: {len(train_dataloader)}")

# Initialize the model
model = VariationalAutoEncoder(input_dim=INPUT_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2, 
                               hidden_dim3=HIDDEN_DIM3, hidden_dim4=HIDDEN_DIM4,
                               latent_dim=LATENT_DIM, num_poles=NUM_POLES).to(DEVICE)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-05)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',             # Looking for min validation loss
    factor=0.5,             # Multiply LR by 0.5 when plateau detected
    patience=5,             # Wait 5 epochs before reducing LR
    threshold=1e-03,        # min change in validation loss to qualify as improvement
    threshold_mode='rel',   # threshold mode is absolute
    min_lr=1e-06,           # Minimum learning rate
)

# Loss function
def loss_function(x_recon, x, mu, logvar, num_poles=NUM_POLES):
    
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(x_recon, x, reduction="sum")
    
    # KL divergence
    kl_divergence = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reconstruction_loss + kl_divergence

# Innitialize train and validation losses
train_losses = [];  val_losses = []


#################################################
################ TRAINING LOOP ##################
#################################################

for epoch in range(NUM_EPOCHS):
    
    print(f'-' * 75 + f' Epoch {epoch + 1} ' + '-' * 50, end='\n')
    
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
    
    model.train()
    
    # Initialize train loss
    train_loss = 0.0
    
    # Loop through the batches
    for i, x in loop:
    
        x = x.view(x.shape[0], INPUT_DIM).to(DEVICE)
        
        # Forward pass
        mu, logvar, z, poles, residues = model(x)
        
        # Decode z to reconstruct the input
        x_recon = GaussLegendreQuadrature(batch_size=BATCH_SIZE, num_poles=NUM_POLES, N_t=N_t, 
                                            tau_array=TAU_ARRAY, poles=poles, residues=residues, beta=BETA)
        
        # Compute the loss
        loss = loss_function(x_recon, x, mu, logvar)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update train loss
        train_loss += loss.item()
        
        # Update the progress bar
        loop.set_description(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
        loop.set_postfix(loss=loss.item()/len(x))
        
    # Average train loss
    train_loss /= len(train_dataloader.dataset)
    
    # Append train loss to record it for each epoch
    train_losses.append(train_loss)
    
    # Set the model to evaluation mode
    model.eval()
    
    
    # Validation loop
    val_loss = 0.0
    
    with torch.no_grad():
        
        for x in val_dataloader:
            
            x = x.view(x.shape[0], INPUT_DIM).to(DEVICE)
            
            # Forward pass
            mu, logvar, z, poles, residues = model(x)
            
            # Decode z to reconstruct the input
            x_recon = GaussLegendreQuadrature(batch_size=BATCH_SIZE, num_poles=NUM_POLES, N_t=N_t, 
                                                tau_array=TAU_ARRAY, poles=poles, residues=residues, beta=BETA)
            
            # Compute the loss
            loss = loss_function(x_recon, x, mu, logvar)
            
            # Update validation loss
            val_loss += loss.item()
            
    # Average validation loss
    val_loss /= len(val_dataloader.dataset)
    
    # Step the scheduler
    scheduler.step(val_loss)
    
    # Append validation loss to record it for each epoch
    val_losses.append(val_loss)
        
    # Print train and validation losses and learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Learning Rate: {current_lr:.3e}')
    print(f'Train Loss: {train_loss:.3e}')
    print(f'Validation Loss: {val_loss:.3e}', end='\n\n')
        
    # Save the train and validation losses
    np.save(f'{OUTPUT_PATH}/losses/train_losses.npy', np.array(train_losses))
    np.save(f'{OUTPUT_PATH}/losses/val_losses.npy', np.array(val_losses))
    
# Save the model after training is done
torch.save(model.state_dict(), f'{OUTPUT_PATH}/model/vae_model.pth')


###################################################
################ TESTING LOOP #####################
####################################################


model.eval()
test_loss = 0.0

# Initialize array for input and reconstructed Green's functions
Greens_input = [];  Greens_recon = []

# Initialize array for the poles and residues
poles_array = [];   residues_array = []

with torch.no_grad():
    
    for i, x in enumerate(test_dataloader):
        
        x = x.view(x.shape[0], INPUT_DIM).to(DEVICE)
        
        # Forward pass
        mu, logvar, z, poles, residues = model(x)
        
        # Decode z to reconstruct the input
        x_recon = GaussLegendreQuadrature(batch_size=BATCH_SIZE, num_poles=NUM_POLES, N_t=N_t, 
                                            tau_array=TAU_ARRAY, poles=poles, residues=residues, beta=BETA)
        
        # Save the reconstructed Green's function
        Greens_input.append(x.cpu())
        Greens_recon.append(x_recon.cpu())
        
        # Save the poles and residues
        poles_array.append(poles.cpu())
        residues_array.append(residues.cpu())
        
        # Compute the loss
        loss = loss_function(x_recon, x, mu, logvar)
        
        # Update test loss
        test_loss += loss.item()
        
    # Average test loss
    test_loss /= len(test_dataloader.dataset)
    
# Print the test loss
print(f'Test Loss: {test_loss:.5f}', end='\n\n')

# Concatenate the reconstructed Green's functions, poles, and residues
Greens_input = torch.cat(Greens_input, dim=0)
Greens_recon = torch.cat(Greens_recon, dim=0)
poles_array = torch.cat(poles_array, dim=0)
residues_array = torch.cat(residues_array, dim=0)

# Compute the average of the Green's function, poles, and residues
Greens_input_avg = Greens_input.mean(dim=0)
Greens_recon_avg = Greens_recon.mean(dim=0)
poles_avg = torch.mean(poles_array, dim=0)
residues_avg = torch.mean(residues_array, dim=0)

# Save all outputs
torch.save({
    'inputs':           Greens_input,
    'recon':            Greens_recon,
    'poles':            poles_array,
    'residues':         residues_array,
    'inputs_avg':       Greens_input_avg,
    'recon_avg':        Greens_recon_avg,
    'poles_avg':        poles_avg,
    'residues_avg':     residues_avg,
    'tau':              TAU_ARRAY,   
    'beta':             BETA,
}, f'{OUTPUT_PATH}/summary.pt')

# print(f"Greens input avg: {Greens_input_avg}", end='\n\n')
# print(f"Greens recon avg: {Greens_recon_avg}", end='\n\n')
print(f"Poles avg: {poles_avg}", end='\n')
print(f"Residues avg: {residues_avg}", end='\n\n')