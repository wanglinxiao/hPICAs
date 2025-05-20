#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulating evolutionary sequence dynamics using Alisim and evaluating functional divergence
via a pretrained CNN model on chromatin accessibility data.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import subprocess
from Bio import SeqIO
from scipy.stats import norm
from decimal import Decimal

# ---------------------------
# model architecture
# ---------------------------

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

# ---------------------------
# one-hot coding + load model
# ---------------------------

def seq_to_hot(seq):
    seq = seq.upper()
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    hot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in base_dict:
            hot[base_dict[base], i] = 1.0
    return torch.tensor(hot).unsqueeze(0)

def load_model(model_path, cuda_id=0):
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    model = VGG('VGG16').to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def model_predict(model, device, seq):
    if len(seq) < 200:
        seq = seq.ljust(200, 'N')
    input_tensor = seq_to_hot(seq).to(device)
    pred = model(input_tensor)
    return pred.item()

# ---------------------------
# perform Alisim and calculate p-value
# ---------------------------

def run_alisim(human_seq, ancestral_seq, evol_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    root_seq_path = os.path.join(output_dir, 'seq.marginal.txt')
    with open(root_seq_path, 'w') as f:
        f.write(">N1\n" + ancestral_seq + "\n")

    output_prefix = os.path.join(output_dir, 'Alisim_mimic')
    cmd = f"""
    iqtree-2.3.6-Linux-intel/bin/iqtree2 \
        -t {evol_file} \
        --alisim {output_prefix} \
        --root-seq {root_seq_path},N1 \
        --seqtype DNA -m GTR+I+G4 \
        --indel 0.05,0.01 --indel-size POW{{1.9/200}},POW{{1.4/200}} \
        --out-format fasta --write-all \
        --num-alignments 1000 -seed 123 --single-output -redo -nt 4 --ancestral
    """
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return os.path.join(output_dir, 'Alisim_mimic.unaligned.fa')

def calculate_pvalue(human_seq, ancestral_seq, evol_file, model_path, output_path, cuda_id=0):
    tmp_dir = os.path.join(os.path.dirname(output_path), "temp_simulation")
    model, device = load_model(model_path, cuda_id)
    fasta_path = run_alisim(human_seq, ancestral_seq, evol_file, tmp_dir)

    null_deltas = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if rec.id == 'Human':
            sim_human_seq = str(rec.seq)
        elif rec.id == '5':
            sim_ancestor_seq = str(rec.seq)
            pred_h = model_predict(model, device, sim_human_seq)
            pred_a = model_predict(model, device, sim_ancestor_seq)
            null_deltas.append(pred_h - pred_a)

    if not null_deltas:
        raise RuntimeError("Alisim simulation failed or returned no sequences.")

    mu, std = np.mean(null_deltas), np.std(null_deltas)
    real_delta = model_predict(model, device, human_seq) - model_predict(model, device, ancestral_seq)
    pval = norm.sf(real_delta, loc=mu, scale=std)
    pval_formatted = '%.2e' % Decimal(pval)

    with open(output_path, 'w') as out:
        out.write(f"{pval_formatted}\n")

    print(f"[✓] P-value written to: {output_path}")


# ---------------------------
# main
# ---------------------------

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage:")
        print("  python 03_Alisim.py <model_path> <human_seq> <ancestor_seq> <tree_file> <output_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    human_seq = sys.argv[2].strip()
    ancestor_seq = sys.argv[3].strip()
    evol_file = sys.argv[4]
    output_file = sys.argv[5]

    calculate_pvalue(human_seq, ancestor_seq, evol_file, model_path, output_file)
