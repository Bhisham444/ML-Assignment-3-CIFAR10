import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data_setup import get_dataloaders, set_seed, SEED
from models import ShallowMLP, DeepMLP, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/q1', exist_ok=True)

train_loader, val_loader, test_loader, train_full, test_set, train_idx, val_idx = get_dataloaders()

# ============================================================
# Q1(a): SHALLOW VS DEEP MLP (FAIR COMPARISON)
# ============================================================
print("="*60)
print("Q1(a): Training Shallow and Deep MLPs under fixed parameter budget")
print("="*60)

shallow_net = ShallowMLP().to(device)
deep_net = DeepMLP().to(device)

print(f"Shallow MLP Parameters: {count_parameters(shallow_net):,}")
print(f"Deep MLP Parameters:    {count_parameters(deep_net):,}")

def train_and_eval(model, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hist = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    for ep in range(epochs):
        model.train()
        tot_loss, cor, tot = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(y)
            cor += (out.argmax(1) == y).sum().item()
            tot += len(y)
        
        tr_loss = tot_loss / tot
        tr_acc = cor / tot
        
        model.eval()
        v_loss, v_cor, v_tot = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item() * len(y)
                v_cor += (out.argmax(1) == y).sum().item()
                v_tot += len(y)
                
        val_loss = v_loss / v_tot
        val_acc = v_cor / v_tot
        
        hist['train_acc'].append(tr_acc)
        hist['val_acc'].append(val_acc)
        hist['train_loss'].append(tr_loss)
        hist['val_loss'].append(val_loss)
        print(f"epoch {ep+1:2d} | train acc: {100*tr_acc:5.1f}% | val acc: {100*val_acc:5.1f}%")
        
    return hist

print("\n--- Training Shallow MLP ---")
hist_shallow = train_and_eval(shallow_net)

print("\n--- Training Deep MLP ---")
hist_deep = train_and_eval(deep_net)

# final test evaluation
def test_accuracy(model):
    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            cor += (model(x).argmax(1) == y).sum().item()
            tot += len(y)
    return 100 * cor / tot

test_shallow = test_accuracy(shallow_net)
test_deep = test_accuracy(deep_net)

print(f"\nFinal Test Accuracy - Shallow MLP: {test_shallow:.2f}%")
print(f"Final Test Accuracy - Deep MLP:    {test_deep:.2f}%")

# plot q1a curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 16), [a*100 for a in hist_shallow['train_acc']], label='Shallow Train Acc', linestyle='--')
plt.plot(range(1, 16), [a*100 for a in hist_shallow['val_acc']], label='Shallow Val Acc')
plt.plot(range(1, 16), [a*100 for a in hist_deep['train_acc']], label='Deep Train Acc', linestyle='--')
plt.plot(range(1, 16), [a*100 for a in hist_deep['val_acc']], label='Deep Val Acc')
plt.title('Train vs Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, 16), hist_shallow['train_loss'], label='Shallow Train Loss', linestyle='--')
plt.plot(range(1, 16), hist_shallow['val_loss'], label='Shallow Val Loss')
plt.plot(range(1, 16), hist_deep['train_loss'], label='Deep Train Loss', linestyle='--')
plt.plot(range(1, 16), hist_deep['val_loss'], label='Deep Val Loss')
plt.title('Train vs Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/q1/q1a_curves.png', dpi=300)
plt.close()
print("Saved plot to results/q1/q1a_curves.png")

# ============================================================
# Q1(b): MEMORIZATION VS GENERALIZATION TRAJECTORY
# ============================================================
print("\n" + "="*60)
print("Q1(b): Memorization experiment with 30% corrupted random labels")
print("="*60)

sub_idx = train_idx[:10000]
imgs = torch.stack([train_full[i][0] for i in sub_idx])
labs = torch.tensor([train_full.targets[i] for i in sub_idx])

clean_mask = torch.ones(10000, dtype=torch.bool)
clean_mask[:3000] = False
clean_mask = clean_mask[torch.randperm(10000)]

noisy_labs = labs.clone()
noisy_labs[~clean_mask] = torch.randint(0, 10, (3000,))

noisy_loader = DataLoader(
    TensorDataset(imgs, noisy_labs, clean_mask), batch_size=128, shuffle=True
)

mem_net = DeepMLP().to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(mem_net.parameters(), lr=0.001)

clean_hist, noisy_hist, val_hist = [], [], []

for ep in range(25):
    mem_net.train()
    for x, y, _ in noisy_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(mem_net(x), y)
        loss.backward()
        opt.step()
        
    mem_net.eval()
    c_cor, c_tot = 0, 0
    n_cor, n_tot = 0, 0
    with torch.no_grad():
        for x, y, mask in noisy_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            preds = mem_net(x).argmax(1)
            c_cor += (preds[mask] == y[mask]).sum().item()
            c_tot += mask.sum().item()
            n_cor += (preds[~mask] == y[~mask]).sum().item()
            n_tot += (~mask).sum().item()
            
        v_cor, v_tot = 0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            v_cor += (mem_net(x).argmax(1) == y).sum().item()
            v_tot += len(y)
            
    c_acc = 100 * c_cor / c_tot
    n_acc = 100 * n_cor / n_tot
    v_acc = 100 * v_cor / v_tot
    
    clean_hist.append(c_acc)
    noisy_hist.append(n_acc)
    val_hist.append(v_acc)
    
    print(f"epoch {ep+1:2d} | clean: {c_acc:5.1f}% | noisy: {n_acc:5.1f}% | val: {v_acc:5.1f}%")

plt.figure(figsize=(8, 5))
plt.plot(range(1, 26), clean_hist, label='Clean Train Acc (70%)', color='green')
plt.plot(range(1, 26), noisy_hist, label='Noisy Train Acc (30% Random)', color='red')
plt.plot(range(1, 26), val_hist, label='Validation Acc (Clean)', color='blue')
plt.title('Memorisation vs Generalisation Trajectory')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/q1/q1b_memorization.png', dpi=300)
plt.close()
print("Saved plot to results/q1/q1b_memorization.png")
