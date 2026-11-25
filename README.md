# Machine Learning Assignment 3: CIFAR-10 Analysis
**MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR-10**

- **Student Name:** Bhisham Sharma
- **Roll Number:** M26MAC003
- **Random Seed:** 26003 (Numeric digits of roll number in order)
- **Repository:** https://github.com/bhisham-sharma/ML-Assignment-3-CIFAR10

---

## 1. Project Overview & Reproduction Instructions

This repository contains the complete experimental code, configurations, saved split indices, and figure generation scripts for Assignment 3. Standard libraries used: `PyTorch`, `torchvision`, `scikit-learn`, `NumPy`, `Matplotlib`, and `pandas`. No pretrained models, foundation models, or external data were used.

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/bhisham-sharma/ML-Assignment-3-CIFAR10.git
cd ML-Assignment-3-CIFAR10

# Install dependencies
pip install -r requirements.txt

# Verify data split setup (stratified 45,000 train / 5,000 val with seed 26003)
python data_setup.py
```

### Exact Reproduction Commands
Run any question sequentially to reproduce figures and numbers:

```bash
# Question 1: MLP Width vs Depth & Memorization Trajectory
python run_q1.py

# Question 2: CNN vs MLP Fair Comparison, Pixel Permutation & Occlusion
python run_q2.py

# Question 3: Autoencoders (Bottlenecks 64, 256, 1024), DAE & Latent Class Probing
python run_q3.py

# Question 4: SVM Grid Sweep (C, Kernels), Boundary Pairs & K-Means Clustering
python run_q4.py
```

---

## 2. Summary of Experimental Results

### Q1: Multi-Layer Perceptrons
- **Shallow MLP (1-Layer, ~1.18M params):** Final Test Accuracy = `49.67%`
- **Deep MLP (4-Layers, ~1.17M params):** Final Test Accuracy = `52.38%`
- **Key Finding:** Adding depth under an identical parameter budget yields only a +2.71% improvement. Both models heavily overfit after epoch 5 (U-shaped validation loss), indicating that MLP performance on images is fundamentally bottlenecked by the absence of 2D spatial locality rather than representational depth.
- **Memorization Experiment (30% Random Labels):** Useful structure is learned *before* memorization begins. Clean training accuracy surges and validation accuracy peaks at `40.2%` (epochs 1–5) while noisy accuracy remains low (<20%). Memorization of random labels occurs subsequently (epochs 6–25), reaching `92.3%` training fit and degrading validation accuracy down to `33.2%`.

### Q2: CNNs, Inductive Biases & Spatial Topology
| Model | Parameters | Condition | Test Accuracy (%) |
| :--- | :---: | :--- | :---: |
| **Deep MLP** | 1,165,514 | Clean | 52.38% |
| **Simple CNN** | 935,882 | Clean | 84.45% |
| **Permuted MLP** | 1,165,514 | Spatially Permuted | 51.55% |
| **Permuted CNN** | 935,882 | Spatially Permuted | 51.47% |
| **Occluded CNN** | 935,882 | 12x12 Center Occluded | 77.82% |

- **Key Finding:** CNN beats MLP by +32.07% despite having 20% fewer parameters. Under fixed random pixel permutation, CNN crashes from 84.45% to 51.47% (matching the MLP), proving its superiority stems entirely from exploiting 2D spatial correlation. Under center occlusion, CNN retains 77.82% accuracy, indicating reliance on distributed contextual and edge features. Confidence is high on correct predictions (93.73%) and low on errors (74.61%).

### Q3: Autoencoders & Representation Learning
- **Bottleneck Capacity Sweep (Val MSE):**
  - Bottleneck 64: `0.1296`
  - Bottleneck 256: `0.0558`
  - Bottleneck 1024: `0.0425` (Selected best model; Test MSE = `0.0422`)
- **Denoising Autoencoder (DAE) Classification Rescue:**
  - Clean CNN Baseline: `84.45%`
  - Gaussian Noisy Images ($\sigma=0.25$): `50.47%`
  - Reconstructions from Clean AE: `45.78%`
  - Reconstructions from DAE: `37.18%`
  - **Key Finding:** Denoising does *not* recover the information the CNN classifier needs. The MSE pixel-loss smooths away the high-frequency discriminative edges that convolutional filters rely upon.
- **Reconstruction Quality vs Class Information (Linear Probe):**
  - Bottleneck 64: Linear Probe Acc = `36.23%`
  - Bottleneck 256: Linear Probe Acc = `41.22%`
  - Bottleneck 1024: Linear Probe Acc = `44.79%`
  - **Key Finding:** Reconstruction quality and semantic class information do not strictly agree; lower reconstruction MSE prioritizes background and lighting fidelity rather than pure class-discriminative semantics.

### Q4: SVM & Unsupervised Clustering (Bonus)
- **SVM Hyperparameter Sweep (Fixed Latent Representation dim=256):**
  - Evaluated $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ across Linear, Poly, and RBF kernels.
  - Severe underfitting at $C=0.01$ (100% support vector fraction, ~10% accuracy).
  - Selected Best Model on Validation: **RBF Kernel, $C=10.0$** (Validation Acc = `47.20%`, Final Test Acc = `46.40%`).
  - Overfitting observed at $C=100.0$ (100% training acc, validation acc drops to 45.0%).
- **Hard Class Pairs & Boundary Analysis:** Identified near-boundary test samples with decision margins close to zero for `cat vs dog`, `automobile vs truck`, and `deer vs horse`.
- **Unsupervised K-Means Clustering:**
  - `Test ARI = 0.0399`, `Test NMI = 0.0881`.
  - **Disagreement Analysis:** K-Means heavily grouped quadruped animals (cat, deer, dog, frog) into `Cluster_1` due to shared outdoor color palettes and fur textures, whereas supervised SVM successfully discriminates them via geometric decision boundaries.

---

## 3. Directory Structure

```text
├── checkpoints/              # Saved model weights (.pt)
│   ├── clean_cnn.pt
│   ├── ae_64.pt
│   ├── ae_256.pt
│   ├── ae_1024.pt
│   └── dae.pt
├── results/                  # Generated figures and CSV tables
│   ├── q1/                   # q1a_curves.png, q1b_memorization.png
│   ├── q2/                   # q2c_confidence.png, q2_summary_table.csv
│   ├── q3/                   # q3a_reconstructions.png, q3b_denoising.png, q3c_recon_vs_class.png
│   └── q4/                   # q4a_svm_sweep.png, q4b_hard_pairs.png, q4c_contingency_table.csv
├── data_setup.py             # Stratified dataset split & loaders
├── models.py                 # PyTorch architectures (MLPs, CNN, Autoencoders)
├── run_q1.py                 # Question 1 reproduction script
├── run_q2.py                 # Question 2 reproduction script
├── run_q3.py                 # Question 3 reproduction script
├── run_q4.py                 # Question 4 reproduction script
├── train_idx.npy             # Stratified 45k train indices (seed 26003)
├── val_idx.npy               # Stratified 5k val indices (seed 26003)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
