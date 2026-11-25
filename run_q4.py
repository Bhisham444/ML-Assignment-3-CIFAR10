import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from data_setup import get_dataloaders, SEED
from models import ConvAutoencoder

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/q4', exist_ok=True)

train_loader, val_loader, test_loader, _, test_set, _, _ = get_dataloaders()

# 1. extract fixed representation (Autoencoder latent dim=256)
rep_ae = ConvAutoencoder(latent_dim=256).to(device)
rep_ae.load_state_dict(torch.load("checkpoints/ae_256.pt"))
rep_encoder = rep_ae.encoder.eval()

def extract_features(encoder, loader):
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = encoder(x)
            features.append(z.cpu())
            labels.append(y)
    return torch.cat(features), torch.cat(labels)

print("Extracting fixed representations for Q4...")
x_train_all, y_train_all = extract_features(rep_encoder, train_loader)
x_val_all, y_val_all = extract_features(rep_encoder, val_loader)
x_test_all, y_test_all = extract_features(rep_encoder, test_loader)

# stratified subset for fast SVM reproduction
sub_tr = 5000
sub_val = 1000
sub_te = 2000

X_tr = x_train_all[:sub_tr].numpy()
Y_tr = y_train_all[:sub_tr].numpy()
X_val = x_val_all[:sub_val].numpy()
Y_val = y_val_all[:sub_val].numpy()
X_te = x_test_all[:sub_te].numpy()
Y_te = y_test_all[:sub_te].numpy()

# ============================================================
# Q4(a): SVM HYPERPARAMETER SWEEP
# ============================================================
print("="*60)
print("Q4(a): SVM Grid Sweep across 5 C values and 3 Kernels")
print("="*60)

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
kernels = ['linear', 'poly', 'rbf']

val_accs = {k: [] for k in kernels}
sv_fracs = {k: [] for k in kernels}
best_val_acc = -1
best_model = None
best_params = None

for k in kernels:
    for c in C_values:
        svm = SVC(C=c, kernel=k, max_iter=2000)
        svm.fit(X_tr, Y_tr)
        
        tr_acc = accuracy_score(Y_tr, svm.predict(X_tr)) * 100
        val_acc = accuracy_score(Y_val, svm.predict(X_val)) * 100
        sv_frac = len(svm.support_) / len(X_tr)
        
        val_accs[k].append(val_acc)
        sv_fracs[k].append(sv_frac)
        
        print(f"Kernel: {k:6s} | C: {c:6.2f} | Train Acc: {tr_acc:5.1f}% | Val Acc: {val_acc:5.1f}% | SV Frac: {sv_frac:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = svm
            best_params = (k, c)

test_acc_best_svm = accuracy_score(Y_te, best_model.predict(X_te)) * 100

print("\n" + "="*50)
print(f"Selected Best SVM: Kernel={best_params[0]}, C={best_params[1]}")
print(f"Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Test Accuracy: {test_acc_best_svm:.2f}%")
print("="*50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for k in kernels:
    ax1.plot(C_values, val_accs[k], marker='o', label=k)
    ax2.plot(C_values, sv_fracs[k], marker='s', label=k)

ax1.set_xscale('log')
ax1.set_xlabel('C (Log Scale)')
ax1.set_ylabel('Validation Accuracy (%)')
ax1.set_title('Validation Accuracy vs C')
ax1.legend()
ax1.grid(True)

ax2.set_xscale('log')
ax2.set_xlabel('C (Log Scale)')
ax2.set_ylabel('Support Vector Fraction')
ax2.set_title('Support Vector Fraction vs C')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/q4/q4a_svm_sweep.png', dpi=300)
plt.close()
print("Saved plot to results/q4/q4a_svm_sweep.png")

# ============================================================
# Q4(b): HARD CLASS PAIRS & DECISION BOUNDARY ANALYSIS
# ============================================================
print("\n" + "="*60)
print("Q4(b): Analyzing 3 Difficult Class Pairs near Decision Boundary")
print("="*60)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
hard_pairs = [(3, 5), (1, 9), (4, 7)] # (cat vs dog), (automobile vs truck), (deer vs horse)
scores = best_model.decision_function(X_te)

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

fig, axes = plt.subplots(3, 4, figsize=(10, 7))

for row, (c1, c2) in enumerate(hard_pairs):
    pair_label = f"{classes[c1]} vs {classes[c2]}"
    mask = (Y_te == c1) | (Y_te == c2)
    pair_idx = np.where(mask)[0]
    
    diff = scores[pair_idx][:, c1] - scores[pair_idx][:, c2]
    closest = pair_idx[np.argsort(np.abs(diff))[:4]]
    
    for col, idx in enumerate(closest):
        img_tensor = test_set[idx][0]
        label = test_set[idx][1]
        img_disp = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0)
        score_val = diff[np.where(pair_idx == idx)[0][0]]
        
        ax = axes[row, col]
        ax.imshow(img_disp)
        ax.set_title(f"True: {classes[label][:4]}\nScore: {score_val:.2f}", fontsize=9)
        ax.axis('off')
        
    axes[row, 0].text(-35, 16, pair_label, rotation=90, verticalalignment='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/q4/q4b_hard_pairs.png', dpi=300)
plt.close()
print("Saved plot to results/q4/q4b_hard_pairs.png")

# ============================================================
# Q4(c): UNSUPERVISED CLUSTERING (K-MEANS)
# ============================================================
print("\n" + "="*60)
print("Q4(c): Unsupervised K-Means Clustering & Disagreement Analysis")
print("="*60)

kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=10)
kmeans.fit(X_tr)

clusters = kmeans.predict(X_te)
ari = adjusted_rand_score(Y_te, clusters)
nmi = normalized_mutual_info_score(Y_te, clusters)

print(f"Test ARI: {ari:.4f}")
print(f"Test NMI: {nmi:.4f}")

cont_mat = confusion_matrix(Y_te, clusters)
df_cont = pd.DataFrame(cont_mat, index=classes, columns=[f"Cluster_{i}" for i in range(10)])
df_cont.to_csv("results/q4/q4c_contingency_table.csv")

print("\n--- Contingency Table ---")
print(df_cont)
print("Saved contingency table to results/q4/q4c_contingency_table.csv")
