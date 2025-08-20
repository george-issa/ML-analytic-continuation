"""
This sctript is used to train a model that predicts poles and residues of a Green's function using residuals from previous iterations. The model uses an RNN cell to process the latent space and extract poles and residues iteratively, updating the residual Green's function at each step. The model is trained using a loss function that combines a KL divergence term and a term collected from the feed forward pass of the model, involving the norm of the residual Green's function.
"""

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
from model_residuals import ModelResiduals # type: ignore
from data_process import GreenFunctionDataset # type: ignore
from Green_reconstruction import GaussLegendreQuadrature # type: ignore


####################################################
################## INITIALIZATION ##################
####################################################


# Special ID for running
sID = 1

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 201;             HIDDEN_DIM1 = 128;                HIDDEN_DIM2 = 64;        HIDDEN_DIM3 = 64;      HIDDEN_DIM4 = 128     

NUM_POLES = 3;               LATENT_DIM = NUM_POLES * 4

NUM_EPOCHS = 50;             BATCH_SIZE = 32;                  LEARNING_RATE = 3e-03 # Kaparthy constant

EPS = 1E-06 # Threshold for stopping resdiual calculation

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
MAIN_PATH = "/Users/georgeissa/Documents/AC"
SPECTRAL_TYPE = "gaussian"

if NOISE:
    # Construct the dataset and output paths for Green's functions with noise added
    # DATA_PATH = f"{MAIN_PATH}/Data/datasets/half-filled-{SPECTRAL_TYPE}/inputs-3/Gbins_boot_means_s{S:.0e}_xi{XI}_nboot{NBOOT}.csv"
    DATA_PATH = "/Users/georgeissa/Documents/AC/Data/datasets/half-filled-gaussian/inputs-3/Gbins_boot_means_s1e-04_xi0.5_nboot5000.csv"
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
model = ModelResiduals(input_dim=INPUT_DIM, 
                        hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2, 
                        latent_dim=LATENT_DIM,
                        hidden_dim3=HIDDEN_DIM3, hidden_dim4=HIDDEN_DIM4,
                        num_poles=NUM_POLES, eps=EPS,
                        N_t=N_t, tau_array=TAU_ARRAY, beta=BETA).to(DEVICE)

# # Print the model architecture
# print("\nModel architecture:")
# print(model)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

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

# Define the KL loss
def kl(mu, logvar):
    
    """
    Compute the KL divergence loss.
    
    Args:
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log variance of the latent distribution.
        
    Returns:
        torch.Tensor: KL divergence loss.
    """
    
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# Initialize train and validation losses
train_losses = []
val_losses = []

#################################################
################ TRAINING LOOP ##################
#################################################

for epoch in range(NUM_EPOCHS):
    
    print(f'-' * 75 + f' Epoch {epoch + 1} ' + '-' * 50, end='\n')
    
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training', leave=False)
    loop.set_description(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    
    model.train()  # Set the model to training mode
    
    # Initialize train loss
    train_loss = 0.0
    
    # Iterate over the training data
    for i, batch in loop:
        
        B = batch.shape[0]  # Get the batch size
        
        # Ensure the batch is a tensor and has the correct shape
        batch = batch.view(B, INPUT_DIM).to(DEVICE)
        
        # Forward pass
        mu, logvar, z, poles, residues, G_res, G_loss = model(batch)
        
        # Define the total loss as the sum of KL divergence and the G_loss
        kl_loss = kl(mu, logvar)
        loss = kl_loss + G_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update train loss
        train_loss += loss.item()
        
        # Update the progress bar
        loop.set_postfix(kl_loss=kl_loss.item()/B, G_loss=G_loss.item()/B, loss=loss.item()/B)
        
    # Average the train loss over the epoch
    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    
    # Initialize validation loss
    val_loss = 0.0
    
    with torch.no_grad():
        
        for i, batch in enumerate(val_dataloader):
            
            B = batch.shape[0]
            
            # Ensure the batch is a tensor and has the correct shape
            batch = batch.view(B, INPUT_DIM).to(DEVICE)
            
            # Forward pass
            mu, logvar, z, poles, residues, G_res, G_loss = model(batch)
            
            # Define the total loss as the sum of KL divergence and the G_loss
            kl_loss = kl(mu, logvar)
            loss = kl_loss + G_loss
            
            # Update validation loss
            val_loss += loss.item()
            
        # Average the validation loss over the epoch
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        # Update the scheduler
        scheduler.step(val_loss)
        
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

model.eval()  # Set the model to evaluation mode

test_loss = 0.0

poles_array, residues_array = [], []

G_in_array, G_res_array = [], []

with torch.no_grad():
    
    for i, batch in enumerate(test_dataloader):
        
        B = batch.shape[0]
        
        # Ensure the batch is a tensor and has the correct shape
        batch = batch.view(B, INPUT_DIM).to(DEVICE)
        
        G_in_array.append(batch.cpu())  # Store the input Green's function
        
        # Forward pass
        mu, logvar, z, poles, residues, G_res, G_loss = model(batch)
        
        # Define the total loss as the sum of KL divergence and the G_loss
        kl_loss = kl(mu, logvar)
        loss = kl_loss + G_loss
        
        # Update test loss
        test_loss += loss.item()
        
        # Store poles and residues
        poles_array.append(poles.cpu())
        residues_array.append(residues.cpu())
        
        # Store the residual Green's function
        G_res_array.append(G_res.cpu())
        
# Average the test loss over the dataset
test_loss /= len(test_dataloader)

print(f'Test Loss: {test_loss:.3e}', end='\n\n')

# Concatenate the poles, residues, and G_res arrays
poles_array = torch.cat(poles_array, dim=0)
residues_array = torch.cat(residues_array, dim=0)
G_in_array = torch.cat(G_in_array, dim=0)
G_res_array = torch.cat(G_res_array, dim=0)

# Compute the average outputs
poles_avg = poles_array.mean(dim=0)
residues_avg = residues_array.mean(dim=0)
G_in_avg = G_in_array.mean(dim=0)
G_res_avg = G_res_array.mean(dim=0)

# Save the poles, residues, and residual Green's function
torch.save({
    'inputs':           G_in_array,
    'residuals':        G_res_array,
    'poles':            poles_array,
    'residues':         residues_array,
    'inputs_avg':       G_in_avg,
    'residuals_avg':    G_res_avg,
    'poles_avg':        poles_avg,
    'residues_avg':     residues_avg,
}, f'{OUTPUT_PATH}/summary.pt')

print(f"\n G_in average: {G_in_avg}")
print(f"\n G_res average: {G_res_avg}")
print(f"\n Poles average: {poles_avg}")
print(f"\n Residues average: {residues_avg}")