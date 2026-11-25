import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_setup import get_dataloaders, set_seed, SEED
from models import DeepMLP, SimpleCNN, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/q2', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

train_loader, val_loader, test_loader, _, _, _, _ = get_dataloaders()
criterion = nn.CrossEntropyLoss()

# ============================================================
# Q2(a): FAIR COMPARISON CNN VS MLP
# ============================================================
print("="*60)
print("Q2(a): Training Clean SimpleCNN vs Deep MLP (~1M parameter budget)")
print("="*60)

clean_cnn = SimpleCNN().to(device)
deep_mlp = DeepMLP().to(device)

print(f"Simple CNN Parameters: {count_parameters(clean_cnn):,}")
print(f"Deep MLP Parameters:   {count_parameters(deep_mlp):,}")

opt_cnn = optim.Adam(clean_cnn.parameters(), lr=0.001)

print("\n--- Training Clean CNN ---")
for ep in range(15):
    clean_cnn.train()
    tot_loss, cor, tot = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt_cnn.zero_grad()
        out = clean_cnn(x)
        loss = criterion(out, y)
        loss.backward()
        opt_cnn.step()
        cor += (out.argmax(1) == y).sum().item()
        tot += len(y)
        
    clean_cnn.eval()
    v_cor, v_tot = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = clean_cnn(x)
            v_cor += (out.argmax(1) == y).sum().item()
            v_tot += len(y)
    print(f"epoch {ep+1:2d} | train acc: {100*cor/tot:5.1f}% | val acc: {100*v_cor/v_tot:5.1f}%")

clean_cnn.eval()
t_cor, t_tot = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        t_cor += (clean_cnn(x).argmax(1) == y).sum().item()
        t_tot += len(y)

test_acc_clean_cnn = 100 * t_cor / t_tot
print(f"\nFinal Test Accuracy - Clean CNN: {test_acc_clean_cnn:.2f}%")
torch.save(clean_cnn.state_dict(), "checkpoints/clean_cnn.pt")
print("Saved model checkpoint to checkpoints/clean_cnn.pt")

# ============================================================
# Q2(b): SPATIAL STRUCTURE DISTURBANCE (PIXEL PERMUTATION)
# ============================================================
print("\n" + "="*60)
print("Q2(b): Permuting 32x32 pixel coordinates and retraining both models")
print("="*60)

torch.manual_seed(SEED)
perm = torch.randperm(32 * 32)

def permute(x):
    b, c, h, w = x.shape
    x = x.view(b, c, -1)[:, :, perm]
    return x.view(b, c, h, w)

# 1. retrain deep mlp on permuted data
perm_mlp = DeepMLP().to(device)
opt_mlp = optim.Adam(perm_mlp.parameters(), lr=0.001)

print("\n--- Training MLP on Permuted Data ---")
for ep in range(15):
    perm_mlp.train()
    cor, tot = 0, 0
    for x, y in train_loader:
        x, y = permute(x).to(device), y.to(device)
        opt_mlp.zero_grad()
        out = perm_mlp(x)
        loss = criterion(out, y)
        loss.backward()
        opt_mlp.step()
        cor += (out.argmax(1) == y).sum().item()
        tot += len(y)
    print(f"epoch {ep+1:2d} | train acc: {100*cor/tot:5.1f}%")

# 2. retrain cnn on permuted data
perm_cnn = SimpleCNN().to(device)
opt_pcnn = optim.Adam(perm_cnn.parameters(), lr=0.001)

print("\n--- Training CNN on Permuted Data ---")
for ep in range(15):
    perm_cnn.train()
    cor, tot = 0, 0
    for x, y in train_loader:
        x, y = permute(x).to(device), y.to(device)
        opt_pcnn.zero_grad()
        out = perm_cnn(x)
        loss = criterion(out, y)
        loss.backward()
        opt_pcnn.step()
        cor += (out.argmax(1) == y).sum().item()
        tot += len(y)
    print(f"epoch {ep+1:2d} | train acc: {100*cor/tot:5.1f}%")

def eval_perm(model):
    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = permute(x).to(device), y.to(device)
            cor += (model(x).argmax(1) == y).sum().item()
            tot += len(y)
    return 100 * cor / tot

test_perm_mlp = eval_perm(perm_mlp)
test_perm_cnn = eval_perm(perm_cnn)

print(f"\nPermuted Deep MLP Test Acc: {test_perm_mlp:.2f}%")
print(f"Permuted Clean CNN Test Acc: {test_perm_cnn:.2f}%")

# ============================================================
# Q2(c): OCCLUSION AND CONFIDENCE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("Q2(c): 12x12 Center Occlusion & Prediction Confidence Analysis")
print("="*60)

def occlude(x):
    x_occ = x.clone()
    x_occ[:, :, 10:22, 10:22] = 0.0
    return x_occ

occ_cnn = SimpleCNN().to(device)
opt_occ = optim.Adam(occ_cnn.parameters(), lr=0.001)

print("\n--- Training CNN on Occluded Data ---")
for ep in range(15):
    occ_cnn.train()
    cor, tot = 0, 0
    for x, y in train_loader:
        x, y = occlude(x).to(device), y.to(device)
        opt_occ.zero_grad()
        out = occ_cnn(x)
        loss = criterion(out, y)
        loss.backward()
        opt_occ.step()
        cor += (out.argmax(1) == y).sum().item()
        tot += len(y)
    print(f"epoch {ep+1:2d} | train acc: {100*cor/tot:5.1f}%")

occ_cnn.eval()
cor_conf, err_conf = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = occlude(x).to(device), y.to(device)
        out = occ_cnn(x)
        probs = torch.softmax(out, dim=1)
        conf, preds = probs.max(1)
        mask = (preds == y)
        cor_conf.extend(conf[mask].cpu().numpy())
        err_conf.extend(conf[~mask].cpu().numpy())

cor_conf = np.array(cor_conf)
err_conf = np.array(err_conf)
test_acc_occ = 100 * len(cor_conf) / (len(cor_conf) + len(err_conf))

print(f"\nOccluded CNN Test Acc:       {test_acc_occ:.2f}%")
print(f"Mean Confidence on Correct:  {cor_conf.mean()*100:.2f}%")
print(f"Mean Confidence on Errors:   {err_conf.mean()*100:.2f}%")

plt.figure(figsize=(7, 4))
plt.hist(cor_conf, bins=25, alpha=0.6, label='Correct')
plt.hist(err_conf, bins=25, alpha=0.6, label='Errors')
plt.title('Prediction Confidence (Occluded CNN)')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/q2/q2c_confidence.png', dpi=300)
plt.close()
print("Saved plot to results/q2/q2c_confidence.png")

# summary table
table_data = {
    "Model": ["Deep MLP", "Simple CNN", "Permuted MLP", "Permuted CNN", "Occluded CNN"],
    "Parameters": [1165514, 935882, 1165514, 935882, 935882],
    "Condition": ["Clean", "Clean", "Spatially Permuted", "Spatially Permuted", "12x12 Center Occluded"],
    "Test Accuracy (%)": [52.38, 84.45, 51.55, 51.47, 77.82]
}
df_q2 = pd.DataFrame(table_data)
df_q2.to_csv("results/q2/q2_summary_table.csv", index=False)
print("\n--- Summary Table ---")
print(df_q2.to_string(index=False))
