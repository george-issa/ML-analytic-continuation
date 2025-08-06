import torch
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    
    'font.size': 12,        # General font size for labels and legends
    'axes.labelsize': 12,   # Font size for axis labels
    'xtick.labelsize': 10,   # Font size for x-axis tick labels
    'ytick.labelsize': 10,   # Font size for y-axis tick labels
    'legend.fontsize': 8,   # Font size for legend
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],  # Use Arial or a similar font
    'text.usetex': True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"

})


# Special ID
sID = 4

# Data parameters
S = 1e-04 # sigma uncertainty level
XI = 0.5  # xi correlation length in imaginaty time

mantissa = S / (10 ** int(np.floor(np.log10(S))))
exponent = int(np.floor(np.log10(S)))  

# Path to summary file
SPECTRAL_TYPE = "Gaussian"
summary_path = f'out_{SPECTRAL_TYPE}_s{S:.0e}_xi{XI}-{sID}/summary.pt'

# Path to exact Green's function
exact_path = f'/Users/georgeissa/Documents/AC/Data/datasets/Gτ_exact.csv'

# Load data
data = torch.load(summary_path, map_location='cpu')
tau = data['tau']  # array-like, shape (N_tau,)

# Load exact Green's function
exact_data = np.loadtxt(exact_path)

# Create path for plots
plots_path = f'out_{SPECTRAL_TYPE}_s{S:.0e}_xi{XI}-{sID}/plots'
os.makedirs(plots_path, exist_ok=True)

G_input = data['inputs']                # shape (N_samples, N_tau)
G_input_avg = data['inputs_avg']        # shape (N_tau,)

G_recon = data['recon']                 # shape (N_samples, N_tau)
G_recon_avg = data['recon_avg']         # shape (N_tau,)

poles = data['poles'].numpy()           # shape (N_samples, M)
residues = data['residues'].numpy()     # shape (N_samples, M) 

# print(f"poles:{poles}")
# print(f"residues:{residues}")

poles_avg = data['poles_avg'].flatten().numpy()         # shape (M,)
residues_avg = data['residues_avg'].flatten().numpy()   # shape (M,)


# Plot input Green's functions in the background
plt.figure(figsize=(8, 6))

for g in G_input:
    plt.plot(tau, g.numpy(), alpha=0.075)

# Plot reconstructed Green's functions
# plt.figure(figsize=(8, 5))

# for g in G_recon:
#     plt.plot(tau, g.numpy(), alpha=0.075)

plt.plot(tau, G_input_avg.numpy(), lw=2, linestyle='-', color='red', label='Input')
plt.plot(tau, G_recon_avg.numpy(), lw=2, linestyle='--', color='black', label='Reconstructed')
plt.plot(tau, exact_data, lw=2, linestyle=':', color='green', label='Exact')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$G(\tau)$')
plt.title("Green's Functions")
plt.legend()
plt.grid(alpha=0.3)

ax = plt.gca()
ax.text(0.80, 0.90, rf'$\sigma = {mantissa} \times 10^{{{exponent}}}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=12)

ax.text(0.80, 0.85, rf'$\xi = {XI}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=12)

plt.savefig(f'{plots_path}/Greens.pdf', bbox_inches='tight')

plt.show()

# Compute spectral function: A(ω) = -1/π Im Σ r / (ω - p)
omega = np.linspace(-6, 6, 1000)

plt.figure(figsize=(8, 5))

######################################################################

# Compute spectral function for input
# A_input = np.zeros_like(omega)

poles_input = [0.0 - 1j * 0.5, 2.0 - 1j * 0.5]
residues_input = [0.5 + 1j * 0.0, 0.5 + 1j * 0.0]  

# for p, r in zip(poles_input, residues_input):
#     # print(f"p_input: {p}, r_input: {r}")
#     A_input += -1/np.pi * np.imag(r / (omega - p))

A_input = np.loadtxt("/Users/georgeissa/Documents/AC/Data/datasets/Spectral_input.csv", delimiter=',')

# Plot all sampled spectral function and mark poles/residues
for i in range(len(poles)):
    A = 0
    # print(f"poles[{i}]: {poles[i]}")
    # print(f"residues[{i}]: {residues[i]}")
    breakpoint()
    for p, r in zip(poles[i], residues[i]):
        # print(f"p: {p}, r: {r}")
        A += -1/np.pi * np.imag(r / (omega - p))
    plt.plot(omega, A, alpha=0.075)

# Compute average spectral function across all samples
A_avg = np.zeros_like(omega)
for p, r in zip(poles_avg, residues_avg):
    # print(f"p_avg: {p}, r_avg: {r}")
    A_avg += -1/np.pi * np.imag(r / (omega - p))
    
plt.plot(omega, A_input, color='red', linestyle='-', label=' Input')
plt.plot(omega, A_avg, color='black', linestyle='--', label=' Extracted')

# mark pole positions
plt.scatter(np.real(poles_input), np.zeros_like(poles_input),
            color='red', marker='o', s=10, label=r'$\Re({\text{Input Poles}}$)')

plt.scatter(np.real(poles_avg), np.zeros_like(poles_avg),
            color='black', marker='o', s=10, label=r'$\Re({\text{Extracted Poles}}$)')

# mark residue positions
plt.scatter(np.real(poles_input), np.abs(residues_input),
            color='red', marker='x', s=10, label=r' $|$Input Residues$|$')

plt.scatter(np.real(poles_avg), np.abs(residues_avg),
            color='black', marker='x', s=10, label=r' $|$Extracted Residues$|$')

plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega)$')
# plt.yticks(np.arange(-5, 5, 1))
plt.title("Spectral Functions with Poles and Residues")
plt.legend()
plt.grid(alpha=0.3)

ax = plt.gca()
ax.text(0.05, 0.90, rf'$\sigma = {mantissa} \times 10^{{{exponent}}}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=12)

ax.text(0.05, 0.85, rf'$\xi = {XI}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=12)

plt.savefig(f'{plots_path}/Spectral.pdf', bbox_inches='tight')

plt.show()



# Plot the poles and residues on a complex plane
plt.figure(figsize=(8, 5))
plt.scatter(np.real(poles), np.imag(poles), color='blue', alpha=0.1, label='Sampled Poles')
plt.scatter(np.real(poles_avg), np.imag(poles_avg), color='black', label='Average Poles')
plt.scatter(np.real(poles_input), np.imag(poles_input), color='red', label='Input Poles')

plt.scatter(np.real(residues), np.imag(residues), color='blue', alpha=0.1, marker='x', label='Sampled Residues')
plt.scatter(np.real(residues_avg), np.imag(residues_avg), color='black', marker='x', label='Average Residues')
plt.scatter(np.real(residues_input), np.imag(residues_input), color='red', marker='x', label='Input Residues')

plt.xlabel(r'$\Re$')
plt.ylabel(r'$\Im$')
plt.title('Poles and Residues in Complex Plane')

plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.axvline(0, color='gray', lw=0.5, ls='--')

ax = plt.gca()
ax.text(0.82, 0.88, rf'$\sigma = {mantissa} \times 10^{{{exponent}}}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=12)

ax.text(0.82, 0.83, rf'$\xi = {XI}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=12)


plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f'{plots_path}/Poles_Residues.pdf', bbox_inches='tight')
plt.show()