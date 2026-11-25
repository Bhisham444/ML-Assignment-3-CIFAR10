import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Student: Bhisham Sharma | Roll No: M26MAC003
# Seed extracted from roll number: 26003
SEED = 26003

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloaders(batch_size=128, data_root='./data'):
    set_seed(SEED)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
    # load or create stratified split
    if os.path.exists('train_idx.npy') and os.path.exists('val_idx.npy'):
        train_idx = np.load('train_idx.npy')
        val_idx = np.load('val_idx.npy')
    else:
        targets = np.array(train_full.targets)
        indices = np.arange(len(targets))
        train_idx, val_idx = train_test_split(
            indices, test_size=5000, stratify=targets, random_state=SEED
        )
        np.save('train_idx.npy', train_idx)
        np.save('val_idx.npy', val_idx)
        
    train_subset = torch.utils.data.Subset(train_full, train_idx)
    val_subset = torch.utils.data.Subset(train_full, val_idx)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, train_full, test_set, train_idx, val_idx

if __name__ == '__main__':
    train_l, val_l, test_l, _, _, tr_idx, v_idx = get_dataloaders()
    print(f"Data setup successful! Seed: {SEED}")
    print(f"Train samples: {len(tr_idx)}, Val samples: {len(v_idx)}, Test samples: 10000")
