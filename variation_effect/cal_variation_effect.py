#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant Effect Prediction Script

This script performs global alignment between a human sequence and its ancestral sequence,
identifies variants (substitutions, insertions, deletions) in the aligned region (excluding leading/trailing gaps),
and predicts variant effects using a PyTorch model.
Outputs results in CSV format.

脚本说明：
- 使用 Biopython 进行全局序列比对并找出变异类型（SNV、INS、DEL）。
- 仅对比对中间区域内部的 INS/DEL 进行效应计算（忽略两端的 gaps）。
- 使用预训练的 PyTorch 模型（one-hot 编码输入）进行预测，并输出 CSV 结果。
"""
import sys
from Bio import pairwise2
import torch
import torch.nn.functional as F
import csv

def align_sequences(human_seq: str, anc_seq: str):
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

def one_hot_encode(seq: str, char_map: dict, num_chars: int):
    """
    One-hot encode a sequence into a PyTorch tensor of shape (1, L, num_chars).
    Sequence characters not in char_map are treated as 'X' (unknown).
    """
    # Map sequence to index list, using 'X' index for unknown residues
    indices = [char_map.get(residue, char_map['X']) for residue in seq]
    tensor = torch.tensor(indices, dtype=torch.long)
    one_hot = F.one_hot(tensor, num_classes=num_chars).float()
    # Add batch dimension (1, L, num_chars)
    return one_hot.unsqueeze(0)

def main_pipeline(human_seq: str, anc_seq: str, model_path: str, cuda_id=None):
    """
    Main pipeline: align sequences, identify variants, and predict variant effects.
    """
    # Ensure sequences are uppercase and stripped of whitespace
    human_seq = human_seq.strip().upper()
    anc_seq = anc_seq.strip().upper()

    # 1. Align sequences
    align_h, align_anc = align_sequences(human_seq, anc_seq)

    # 2. Determine middle region (ignore leading/trailing gaps in either sequence)
    start = 0
    while start < len(align_h) and (align_h[start] == '-' or align_anc[start] == '-'):
        start += 1
    end = len(align_h) - 1
    while end >= 0 and (align_h[end] == '-' or align_anc[end] == '-'):
        end -= 1

    # 3. Identify variants (SNV/INS/DEL) in the middle region
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
                variants.append(('SNV', human_pos, align_anc[i], align_h[i]))
            i += 1

        elif align_h[i] == '-' and align_anc[i] != '-':
            # Deletion in human sequence (present in ancestral, gap in human)
            start_pos = human_pos + 1
            del_seq = ''
            while i < len(align_h) and align_h[i] == '-' and align_anc[i] != '-':
                del_seq += align_anc[i]
                anc_pos += 1
                i += 1
            variants.append(('DEL', start_pos, del_seq))

        elif align_anc[i] == '-' and align_h[i] != '-':
            # Insertion in human sequence (present in human, gap in ancestral)
            start_pos = human_pos + 1
            ins_seq = ''
            while i < len(align_h) and align_anc[i] == '-' and align_h[i] != '-':
                ins_seq += align_h[i]
                human_pos += 1
                i += 1
            variants.append(('INS', start_pos, ins_seq))

        else:
            # Both gaps (should not occur in a valid alignment)
            i += 1

    # 4. Load PyTorch model
    device = torch.device(f"cuda:{cuda_id}" if cuda_id is not None and torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    # 5. Define amino acid mapping (20 canonical AAs + 'X' for unknown)
    residues = list("ACDEFGHIKLMNPQRSTVWY") + ['X']
    char_map = {aa: idx for idx, aa in enumerate(residues)}
    num_chars = len(residues)

    # 6. Compute baseline score for ancestral sequence
    base_tensor = one_hot_encode(anc_seq, char_map, num_chars).to(device)
    with torch.no_grad():
        base_score = model(base_tensor)
    base_score = base_score.item() if isinstance(base_score, torch.Tensor) else float(base_score)

    # 7. Predict effect for each variant and write to CSV
    with open('variant_effects.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Type','Position','Ref','Alt','Score_Anc','Score_Var','Delta'])
        for v in variants:
            vtype, pos = v[0], v[1]
            if vtype == 'SNV':
                ref_aa, alt_aa = v[2], v[3]
                idx = pos - 1
                # Replace the ancestral amino acid with the human amino acid at this position
                var_seq = anc_seq[:idx] + alt_aa + anc_seq[idx+1:]
            elif vtype == 'DEL':
                ref_aa = v[2]
                alt_aa = '-'
                idx = pos - 1
                # Delete the segment from the ancestral sequence
                var_seq = anc_seq[:idx] + anc_seq[idx + len(ref_aa):]
            elif vtype == 'INS':
                ref_aa = '-'
                alt_aa = v[2]
                idx = pos - 1
                # Insert the sequence into the ancestral sequence
                var_seq = anc_seq[:idx] + alt_aa + anc_seq[idx:]
            else:
                continue

            # One-hot encode variant sequence and predict score
            var_tensor = one_hot_encode(var_seq, char_map, num_chars).to(device)
            with torch.no_grad():
                var_score = model(var_tensor)
            var_score = var_score.item() if isinstance(var_score, torch.Tensor) else float(var_score)
            delta = var_score - base_score

            writer.writerow([vtype, pos, ref_aa, alt_aa, base_score, var_score, delta])

if __name__ == "__main__":
    # Usage: script.py human_seq anc_seq model_path [cuda_id]
    if len(sys.argv) < 4:
        print("Usage: python script.py <human_seq> <anc_seq> <model_path> [cuda_id]")
        sys.exit(1)
    human_seq_arg = sys.argv[1]
    anc_seq_arg = sys.argv[2]
    model_path_arg = sys.argv[3]
    cuda_id_arg = sys.argv[4] if len(sys.argv) > 4 else None
    main_pipeline(human_seq_arg, anc_seq_arg, model_path_arg, cuda_id_arg)

