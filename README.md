# Green's Function Analytic Continuation with VAE

This project implements a **physics-informed machine learning framework** for the **analytic continuation** of imaginary-time Green’s functions into real-frequency spectral functions. It is built around a **Variational Autoencoder (VAE)** that explicitly learns a **pole-residue decomposition**, providing a robust and interpretable reconstruction of spectral data from Quantum Monte Carlo (QMC) simulations.

---

## 🔍 Overview

Analytic continuation is a notoriously ill-posed problem in many-body physics. Traditional approaches like Maximum Entropy can be unstable and sensitive to noise. This project tackles the problem by:

- Learning poles and residues directly via a VAE.
- Reconstructing spectral functions from noisy QMC Green's functions.
- Providing a flexible and extensible framework for physicists and researchers.

---

## 🧠 Key Features

- ✅ Variational Autoencoder architecture
- ✅ Explicit pole and residue extraction
- ✅ Supports synthetic and QMC Green's function input
- ✅ Modular code structure (training, evaluation, visualization)
- ✅ Easy to extend to other continuation tasks

---

## 📁 Project Structure

