#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a VGG16-based CNN model on DNA sequences to classify chromatin accessibility.
This script includes data preprocessing, dataset construction, model definition, and training loop with early stopping.
"""

import os
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from sklearn.metrics import roc_auc_score

# Parse input arguments
pos_file = sys.argv[1]
neg_file = sys.argv[2]
model_save_path = sys.argv[3]

# One-hot encoding function for DNA sequences
def seq_to_onehot(seq):
    seq = seq.upper().replace('N', '0')
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    onehot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            onehot[mapping[base], i] = 1.0
    return onehot

def generate_onehot_dataset(seq_list):
    return np.array([seq_to_onehot(seq) for seq in seq_list])

# Pad or trim sequences to length 1000
def pad_sequence(seq, target_len=1000):
    if len(seq) < target_len:
        return seq.center(target_len, 'N')
    center = len(seq) // 2
    return seq[center - target_len // 2 : center + target_len // 2]

# Load sequences and labels from fasta-like files
def load_labeled_data(pos_file, neg_file, subset):
    def load_fasta(file):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
        return {lines[i][1:]: pad_sequence(lines[i+1].upper()) for i in range(0, len(lines), 2)}

    pos_dict = load_fasta(pos_file)
    neg_dict = load_fasta(neg_file)

    def filter_by_chr(seqs, chroms):
        return [s for k, s in seqs.items() if re.match(r'(chr\w+):', k).group(1) in chroms]

    chrom_split = {
        'train': lambda c: c not in {'chr1', 'chr2'},
        'val':   lambda c: c == 'chr1',
        'test':  lambda c: c == 'chr2'
    }

    select_fn = chrom_split[subset]
    get_chr = lambda s: re.match(r'(chr\w+):', s).group(1)

    pos_seqs = [s for k, s in pos_dict.items() if select_fn(get_chr(k))]
    neg_seqs = [s for k, s in neg_dict.items() if select_fn(get_chr(k))]

    labels = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs), dtype=np.int64)
    seqs = generate_onehot_dataset(pos_seqs + neg_seqs)
    return seqs, labels

# Load datasets
train_X, train_y = load_labeled_data(pos_file, neg_file, 'train')
val_X, val_y = load_labeled_data(pos_file, neg_file, 'val')
test_X, test_y = load_labeled_data(pos_file, neg_file, 'test')

# PyTorch dataset wrapper
class SequenceDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

# Model configuration and architecture
CFG = {'VGG16': [280, 280, 'M', 180, 180, 'M', 120, 120, 'M']}

class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG,self).__init__()
        self.feature=self.make_layers(CFG[vgg_name])
        self.fc1=nn.Linear(360,64)
        self.fc2=nn.Linear(64,1)
        
    def forward(self,x):
        out=self.feature(x)          
        out=out.view(out.size(0),-1) 
        out1=F.relu(self.fc1(out))   
        out2=torch.sigmoid(self.fc2(out1))
        out2=out2.view(-1) 
        return out2
    
    def make_layers(self,CFG):
        layers=[]
        in_channels=4
        for x in CFG:
            if x == 'M':
                layers.append(nn.MaxPool1d(kernel_size=2,stride=2))
            elif x == 280:
                layers += [nn.Conv1d(in_channels,x,kernel_size=13),nn.BatchNorm1d(x),nn.ReLU(inplace=True)]
                in_channels=x
            elif x == 180:
                layers += [nn.Conv1d(in_channels,x,kernel_size=11),nn.BatchNorm1d(x),nn.ReLU(inplace=True)]
                in_channels=x
            elif x == 120:
                layers += [nn.Conv1d(in_channels,x,kernel_size=9),nn.BatchNorm1d(x),nn.ReLU(inplace=True)]
                in_channels=x                
        layers += [nn.AdaptiveAvgPool1d(3)]
        return nn.Sequential(*layers)

# Training and evaluation

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            preds.extend(outputs.cpu().numpy())
            labels.extend(batch_y.cpu().numpy())
    auc = roc_auc_score(labels, preds)
    return total_loss / len(loader), auc

# Device setup
model = VGG('VGG16')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Dataloaders
train_loader = data.DataLoader(SequenceDataset(train_X, train_y), batch_size=64, shuffle=True)
val_loader = data.DataLoader(SequenceDataset(val_X, val_y), batch_size=64)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Training loop with early stopping
best_auc = 0
patience, counter = 7, 0

for epoch in range(1, 31):
    print(f"Epoch {epoch}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc
        }, model_save_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
