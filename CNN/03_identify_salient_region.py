#!/usr/bin/env python
# coding: utf-8

"""
Salient Region Identification via Integrated Gradients using the CNN.
Description:
    This script identifies salient genomic regions from input DNA sequences using 
    a pre-trained convolutional neural network and the 
    Integrated Gradients method (Captum library).
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from captum.attr import IntegratedGradients

# ----------------------------- Configuration ----------------------------- #
WINDOW_SIZE = 20
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CFG = {'VGG16': [280, 280, 'M', 180, 180, 'M', 120, 120, 'M']}

# ----------------------------- Model Definition ----------------------------- #
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

# ----------------------------- Sequence Encoding ----------------------------- #
def seq_to_onehot(seq: str) -> torch.Tensor:
    """Convert DNA sequence to one-hot encoded tensor."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.upper()
    onehot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            onehot[mapping[nucleotide], i] = 1.0
    return torch.tensor(onehot).unsqueeze(0)

# ----------------------------- Integrated Gradients ----------------------------- #
def compute_salient_region(row):
    """Apply Integrated Gradients and extract salient region information."""
    seq = row['Seq'].upper().rjust(300, 'N') if len(row['Seq']) < 300 else row['Seq'].upper()

    try:
        chrom, start, _ = re.match(r'(.+):(\d+)-(\d+)', row['Peak name']).groups()
        start = int(start)
    except Exception as e:
        raise ValueError(f"Invalid Peak name format: {row['Peak name']}") from e

    input_tensor = seq_to_onehot(seq).to(DEVICE)
    baseline = torch.zeros_like(input_tensor).to(DEVICE)

    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(input_tensor, baseline, n_steps=1000, return_convergence_delta=True)

    df_attr = pd.DataFrame(attributions.squeeze(0).cpu().numpy(), index=['A', 'C', 'G', 'T'])
    df_attr.loc['max'] = df_attr.max()
    df_attr.loc['rolling'] = df_attr.loc['max'].rolling(WINDOW_SIZE).sum()

    max_score = df_attr.loc['rolling'].max()
    sr_end = df_attr.columns[df_attr.loc['rolling'] == max_score][0]
    sr_start = sr_end - (WINDOW_SIZE - 1)

    salient_seq = seq[sr_start:sr_end + 1]
    sr_genome_start = start + sr_start
    sr_genome_end = start + sr_end
    pred_score = model(input_tensor).item()

    return chrom, sr_genome_start, sr_genome_end, salient_seq, pred_score

# ----------------------------- Main Pipeline ----------------------------- #
def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_fasta> <output_csv> <model_file>")
        sys.exit(1)

    input_fasta, output_csv, model_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Load sequences
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_fasta, "fasta")}
    df = pd.DataFrame({
        'Peak name': list(sequences.keys()),
        'Seq': list(sequences.values())
    })
    df['Seq_length'] = df['Seq'].apply(len)

    # Load model
    global model
    model = VGG('VGG16').to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # Run attribution analysis
    df[['SR_chrom', 'SR_start', 'SR_end', 'SR_seq', 'Seq_pred']] = df.apply(
        compute_salient_region, axis=1, result_type='expand'
    )

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
