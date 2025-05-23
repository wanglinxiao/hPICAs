#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script calculates the effect of sequence variations (SNV, insertion, deletion)
based on global pairwise alignment between human and ancestral DNA sequences.
A pre-trained CNN model is used to predict the variation effect on chromatin accessibility.
Only internal indels are considered to avoid alignment artifacts at sequence ends.
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio import pairwise2
import pickle
import sys

def seq_to_onehot(seq):
    seq = seq.upper()
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    hot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in base_dict:
            hot[base_dict[base], i] = 1.0
    return torch.tensor(hot).unsqueeze(0)

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

def load_model(model_path, cuda_id=0):
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    model = VGG('VGG16').to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def align_sequences(human_seq, anc_seq):
    """
    Perform global alignment of two sequences using specified scoring parameters.
    Scoring parameters: match=2, mismatch=-3, open_gap=-5, extend_gap=-2.
    Returns aligned human sequence and aligned ancestral sequence.
    """
    # Global alignment with given scoring scheme
    alignments = pairwise2.align.globalms(human_seq, anc_seq,
                                          2,   # match score
                                         -3,   # mismatch score
                                         -5,   # gap open penalty
                                         -2)   # gap extend penalty
    # Take the highest-scoring alignment
    align_h, align_anc, score, begin, end = alignments[0]
    return align_h, align_anc

def parse_variants(align_h, align_anc):
    # Determine middle region (ignore leading/trailing gaps in either sequence)
    start = 0
    while start < len(align_h) and (align_h[start] == '-' or align_anc[start] == '-'):
        start += 1
    end = len(align_h) - 1
    while end >= 0 and (align_h[end] == '-' or align_anc[end] == '-'):
        end -= 1

    # Identify variants (SNV/INS/DEL) in the middle region
    variants = []
    i = 0
    human_pos = 0  # position in human sequence (1-indexed after increment)
    anc_pos = 0    # position in ancestral sequence
    while i < len(align_h):
        # Update position counters outside mid region (skipping these for variant calls)
        if i < start or i > end:
            if align_h[i] != '-' and align_anc[i] != '-':
                human_pos += 1
                anc_pos += 1
            elif align_h[i] != '-' and align_anc[i] == '-':
                human_pos += 1
            elif align_h[i] == '-' and align_anc[i] != '-':
                anc_pos += 1
            i += 1
            continue

        # Within the middle region of alignment
        if align_h[i] != '-' and align_anc[i] != '-':
            human_pos += 1
            anc_pos += 1
            # Check for SNV (substitution)
            if align_h[i] != align_anc[i]:
                # Record SNV: (type, position, ref_aa, alt_aa)
                variants.append({
                    'type': 'SNV',
                    'human_pos': human_pos,
                    'anc_pos': anc_pos,
                    'human_base': align_h[i],
                    'anc_base':align_anc[i]})
            i += 1

        elif align_h[i] == '-' and align_anc[i] != '-':
            # Deletion in human sequence (present in ancestral, gap in human)
            start_pos = human_pos
            del_seq = ''
            while i < len(align_h) and align_h[i] == '-' and align_anc[i] != '-':
                del_seq += align_anc[i]
                anc_pos += 1
                i += 1
            variants.append({
                'type': 'deletion',
                'human_pos': (start_pos,start_pos+1),
                'anc_pos': anc_pos-len(del_seq),
                'human_base': del_seq,
                'anc_base': ''})    

        elif align_anc[i] == '-' and align_h[i] != '-':
            # Insertion in human sequence (present in human, gap in ancestral)
            start_pos = human_pos
            ins_seq = ''
            while i < len(align_h) and align_anc[i] == '-' and align_h[i] != '-':
                ins_seq += align_h[i]
                human_pos += 1
                i += 1
            variants.append({
                'type': 'insertion',
                'human_pos': (start_pos,start_pos+len(ins_seq)),
                'anc_pos': anc_pos,
                'human_base': '',
                'anc_base': ins_seq})    
                    
        else:
            # Both gaps (should not occur in a valid alignment)
            i += 1
            
    return variants

def predict_variant_effect(model, anc_seq, variant, device):
    ref_seq = anc_seq
    alt_seq = anc_seq  
    pos = variant['anc_pos']
    
    if variant['type'] == 'SNV':
        # substitution
        alt_base = variant['human_base']
        alt_seq = anc_seq[:pos-1] + alt_base + anc_seq[pos:]
    elif variant['type'] == 'insertion':
        # Insertion in human (absent in ancestor)
        ins_seq = variant['anc_base']
        alt_seq = anc_seq[:pos] + ins_seq + anc_seq[pos:]
    elif variant['type'] == 'deletion':
        # Deletion in human (present in ancestor)
        del_seq = variant['human_base']
        alt_seq = anc_seq[:pos] + anc_seq[pos+len(del_seq):]
    else:
        return None  # unknown
    
    X_ref = seq_to_onehot(ref_seq).to(device)
    X_alt = seq_to_onehot(alt_seq).to(device)
    
    pred_ref = model(X_ref).item()
    pred_alt = model(X_alt).item()
    
    return pred_alt - pred_ref

def main_pipeline(human_seq, ancestral_seq, model, device, output_file):
    # 1. global sequence alignment
    aln_h, aln_a = align_sequences(human_seq, ancestral_seq)
    
    # 2. extract variation
    variants = parse_variants(align_h = aln_h, align_anc = aln_a)
    
    # 3. calculate the effect of each variation
    results = []
    for var in variants:
        effect = predict_variant_effect(model, anc_seq = ancestral_seq, variant = var, device = device)
        results.append({
            'type': var['type'],
            'pos': var['human_pos'],
            'pred_change': effect
        })
    
    # 4. save result
    fieldnames = ['type', 'pos', 'pred_change']
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    return results

if __name__ == "__main__":
    # Usage: script.py human_seq anc_seq model_path [cuda_id]
    if len(sys.argv) < 5:
        print("Usage: python script.py <human_seq> <anc_seq> <model_path> <result_file>")
        sys.exit(1)
    human_seq = sys.argv[1]
    anc_seq = sys.argv[2]
    model_path = sys.argv[3]
    result_file = sys.argv[4]
    
    model,device=load_model(model_path=model_path)
    main_pipeline(human_seq, anc_seq, model, device, result_file)

