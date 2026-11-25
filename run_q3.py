import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_setup import get_dataloaders
from models import ConvAutoencoder, SimpleCNN, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/q3', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

train_loader, val_loader, test_loader, _, _, _, _ = get_dataloaders()
criterion_mse = nn.MSELoss()

# ============================================================
# Q3(a): BOTTLENECK CAPACITY STUDY
# ============================================================
print("="*60)
print("Q3(a): Training 3 Convolutional Autoencoders (Bottlenecks: 64, 256, 1024)")
print("="*60)

bottlenecks = [64, 256, 1024]
ae_models = {}
ae_val_errors = {}

for b_size in bottlenecks:
    print(f"\n--- Training Autoencoder (Bottleneck: {b_size}) ---")
    model = ConvAutoencoder(latent_dim=b_size).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    for ep in range(10):
        model.train()
        tot_loss, tot = 0, 0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = criterion_mse(recon, x)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(x)
            tot += len(x)
        print(f"epoch {ep+1:2d} | train mse: {tot_loss/tot:.4f}")
        
    model.eval()
    v_loss, v_tot = 0, 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            recon = model(x)
            v_loss += criterion_mse(recon, x).item() * len(x)
            v_tot += len(x)
            
    val_mse = v_loss / v_tot
    ae_val_errors[b_size] = val_mse
    ae_models[b_size] = model
    print(f"--> Bottleneck {b_size} Val MSE: {val_mse:.4f}")
    torch.save(model.state_dict(), f"checkpoints/ae_{b_size}.pt")

best_b = min(ae_val_errors, key=ae_val_errors.get)
print(f"\nSelected Best Clean AE: Bottleneck = {best_b} (Val MSE: {ae_val_errors[best_b]:.4f})")

best_ae = ae_models[best_b]
best_ae.eval()
t_loss, t_tot = 0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        recon = best_ae(x)
        t_loss += criterion_mse(recon, x).item() * len(x)
        t_tot += len(x)

print(f"Final Test Reconstruction MSE (Bottleneck {best_b}): {t_loss/t_tot:.4f}")

# plot sample reconstructions
sample_imgs, _ = next(iter(test_loader))
sample_imgs = sample_imgs[:5].to(device)
with torch.no_grad():
    recon_imgs = best_ae(sample_imgs)

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
orig_disp = (sample_imgs * std + mean).clamp(0, 1).cpu()
recon_disp = (recon_imgs * std + mean).clamp(0, 1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    axes[0, i].imshow(orig_disp[i].permute(1, 2, 0))
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Original")
    axes[1, i].imshow(recon_disp[i].permute(1, 2, 0))
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title(f"Recon ({best_b})")

plt.tight_layout()
plt.savefig('results/q3/q3a_reconstructions.png', dpi=300)
plt.close()
print("Saved plot to results/q3/q3a_reconstructions.png")

# ============================================================
# Q3(b): DENOISING AUTOENCODER (DAE) & CLASSIFIER RESCUE
# ============================================================
print("\n" + "="*60)
print("Q3(b): Training Denoising Autoencoder & Testing on Fixed CNN")
print("="*60)

def add_noise(x, sigma=0.25):
    return x + torch.randn_like(x) * sigma

dae = ConvAutoencoder(latent_dim=256).to(device)
opt_dae = optim.Adam(dae.parameters(), lr=0.001)

print("\n--- Training DAE ---")
for ep in range(10):
    dae.train()
    tot_loss, tot = 0, 0
    for x, _ in train_loader:
        x = x.to(device)
        noisy = add_noise(x)
        opt_dae.zero_grad()
        recon = dae(noisy)
        loss = criterion_mse(recon, x)
        loss.backward()
        opt_dae.step()
        tot_loss += loss.item() * len(x)
        tot += len(x)
    print(f"epoch {ep+1:2d} | loss: {tot_loss/tot:.4f}")

torch.save(dae.state_dict(), "checkpoints/dae.pt")

cnn = SimpleCNN().to(device)
cnn.load_state_dict(torch.load("checkpoints/clean_cnn.pt"))
cnn.eval()

clean_ae = ConvAutoencoder(latent_dim=1024).to(device)
clean_ae.load_state_dict(torch.load("checkpoints/ae_1024.pt"))
clean_ae.eval()
dae.eval()

c_noisy, c_clean, c_dae, tot = 0, 0, 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        noisy = add_noise(x)
        c_noisy += (cnn(noisy).argmax(1) == y).sum().item()
        c_clean += (cnn(clean_ae(noisy)).argmax(1) == y).sum().item()
        c_dae += (cnn(dae(noisy)).argmax(1) == y).sum().item()
        tot += len(y)

print(f"\nOriginal Clean CNN Test Acc: 84.45%")
print(f"Noisy Images Test Acc:       {100*c_noisy/tot:.2f}%")
print(f"Clean AE Recon Test Acc:     {100*c_clean/tot:.2f}%")
print(f"DAE Recon Test Acc:          {100*c_dae/tot:.2f}%")

sample, _ = next(iter(test_loader))
sample = sample[:5].to(device)
noisy_sample = add_noise(sample)
with torch.no_grad():
    denoised = dae(noisy_sample)

sample_disp = ((sample * std + mean).clamp(0, 1)).cpu()
noisy_disp = ((noisy_sample * std + mean).clamp(0, 1)).cpu()
dae_disp = ((denoised * std + mean).clamp(0, 1)).cpu()

fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i in range(5):
    axes[0, i].imshow(sample_disp[i].permute(1, 2, 0))
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Clean")
    axes[1, i].imshow(noisy_disp[i].permute(1, 2, 0))
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("Noisy")
    axes[2, i].imshow(dae_disp[i].permute(1, 2, 0))
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title("Denoised")

plt.tight_layout()
plt.savefig('results/q3/q3b_denoising.png', dpi=300)
plt.close()
print("Saved plot to results/q3/q3b_denoising.png")

# ============================================================
# Q3(c): RECONSTRUCTION QUALITY VS CLASS INFORMATION
# ============================================================
print("\n" + "="*60)
print("Q3(c): Linear Probing on Frozen Encoders vs Reconstruction Error")
print("="*60)

def extract_features(encoder, loader):
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = encoder(x)
            features.append(z.cpu())
            labels.append(y)
    return torch.cat(features), torch.cat(labels)

linear_probe_results = {}

for b_size in [64, 256, 1024]:
    ae = ConvAutoencoder(latent_dim=b_size).to(device)
    ae.load_state_dict(torch.load(f"checkpoints/ae_{b_size}.pt"))
    encoder = ae.encoder.eval()
    
    x_tr, y_tr = extract_features(encoder, train_loader)
    x_te, y_te = extract_features(encoder, test_loader)
    
    probe = nn.Linear(b_size, 10).to(device)
    probe_opt = optim.Adam(probe.parameters(), lr=0.005)
    probe_crit = nn.CrossEntropyLoss()
    
    probe_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=256, shuffle=True
    )
    
    for ep in range(10):
        probe.train()
        for bx, by in probe_loader:
            bx, by = bx.to(device), by.to(device)
            probe_opt.zero_grad()
            loss = probe_crit(probe(bx), by)
            loss.backward()
            probe_opt.step()
            
    probe.eval()
    with torch.no_grad():
        preds = probe(x_te.to(device)).argmax(1).cpu()
        acc = (preds == y_te).float().mean().item() * 100
        
    linear_probe_results[b_size] = acc
    print(f"Bottleneck {b_size:4d} | Val MSE: {ae_val_errors[b_size]:.4f} | Linear Probe Acc: {acc:.2f}%")

b_list = [64, 256, 1024]
mse_list = [ae_val_errors[b] for b in b_list]
acc_list = [linear_probe_results[b] for b in b_list]

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.set_xlabel('Bottleneck Size')
ax1.set_ylabel('Reconstruction MSE', color='red')
ax1.plot(b_list, mse_list, color='red', marker='o')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()  
ax2.set_ylabel('Linear Probe Acc (%)', color='blue')
ax2.plot(b_list, acc_list, color='blue', marker='s', linestyle='--')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Reconstruction MSE vs Class Information')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/q3/q3c_recon_vs_class.png', dpi=300)
plt.close()
print("Saved plot to results/q3/q3c_recon_vs_class.png")
