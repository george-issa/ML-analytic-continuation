# Green's Function Analytic Continuation with VAE

This project implements a **physics-informed machine learning framework** for the **analytic continuation** of imaginary-time Green’s functions into real-frequency spectral functions. It is built around a **Variational Autoencoder (VAE)** that explicitly learns a **pole-residue decomposition**, providing a robust and interpretable reconstruction of spectral data from Quantum Monte Carlo (QMC) simulations.

---

## 🔍 Overview

Analytic continuation is a notoriously ill-posed problem in many-body Condensed Matter Physics. Traditional approaches like Maximum Entropy can be unstable and sensitive to noise. This project tackles the problem by:

- Learning poles and residues directly via a VAE.
- Reconstructing spectral functions from noisy QMC Green's functions.
- Providing a flexible and extensible framework for physicists and researchers.

---

## 🧠 Key Features

- ✅ Variational Autoencoder architecture
- ✅ Explicit pole and residue extraction from a learned distribution
- ✅ Supports synthetic and QMC Green's function imaginary-time inputs
- ✅ Modular code structure (training, evaluation, visualization)
- ✅ Easy to extend to other continuation tasks

---

## 📁 Project Structure
```text
greens-function-analytic-continuation/
├── README.md
├── LICENSE
├── requirements.txt
├── src/                                      # Main model and training code
│   ├── model.py                              # VAE class and architecture
│   ├── train.py                              # training and testing
│   ├── Green_reconstruction.py               # reconstructing poles and residues to Green's functions
│   ├── data_process.py                       # input data processing before feeding to network
├── synthetic-data/                           # Placeholder for input Green’s functions as sythetic data
│   ├── half-filled-gaussian                  # Green's functions corresponding to Gaussian spectral functions
│   ├── half-filled-lorentzian                # Green's functions corresponding to Lorentzian spectral functions
│   ├── Green_reconstruction.py               # reconstructing poles and residues to Green's functions
├── example-outputs/                          # Output poles, residues, plots
│   ├── out_gaussian_numpoles1_s1e-04_xi0.5-1 
│   ├── out_gaussian_numpoles1_s1e-04_xi0.5-1
├── examples/                                 # Example usage scripts
│   └── usage_example.py
```


## ⚙️ Installation

Clone the repository:
git clone https://github.com/george-issa/greens-function-analytic-continuation.git
cd greens-function-analytic-continuation

Install required packages:
pip install -r requirements.txt

Or with Conda:
conda env create -f environment.yml
conda activate gfvae

## 🚀 Usage

Train a model:
python src/train.py

## 📄 License

This project is licensed under the MIT License.

## 🙋 Contact

For questions, discussions, or collaborations, feel free to:

    Open an issue

    Submit a pull request

    Connect on GitHub
