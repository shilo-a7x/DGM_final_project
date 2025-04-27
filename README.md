# DGM final project
Final project in Deep Generative Models course at BIU.

# VAE with GMM Clustering

This project trains a Variational Autoencoder (VAE) on MNIST and clusters the latent space using a Gaussian Mixture Model (GMM).  
The VAE and GMM are trained mutually in iterations: the VAE learns better representations, and the GMM clusters the latent space to guide the VAE.

## 🔗 Full Project Report

The full explanation, background, methodology, and results are available in the [Final Project Report](./DGM_final_project___VAE_with_latent_clustering_using_GMM.pdf).

---

## How to Run

```bash
# Install dependencies
pip install torch torchvision scikit-learn matplotlib

# Run training and sampling
python code/main.py
