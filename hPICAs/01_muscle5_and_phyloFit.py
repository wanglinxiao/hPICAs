#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs multiple sequence alignment using MUSCLE5 and infers phylogenetic trees using PhyloFit
for homologous regions across human, chimpanzee, and gorilla genomes. It is designed to be used in large-scale
comparative epigenomics studies.
"""

import os
import sys
import re
import random
import subprocess
import pandas as pd
from multiprocess import Pool

# Directories for outputs
alignment_dir = sys.argv[1]
phylofit_dir = sys.argv[2]

os.makedirs(alignment_dir, exist_ok=True)
os.makedirs(phylofit_dir, exist_ok=True)
os.chdir(phylofit_dir)

# ============================
# Utility Functions
# ============================

def load_peak_mapping(pairwise_file):
    """Load peak mappings between human and primate from alignment summary."""
    df = pd.read_table(pairwise_file)
    return dict(zip(df['human_peak'], df['primate_peak']))

def load_sequence_dicts(pairwise_file):
    """Load sequence mappings for human and primate."""
    df = pd.read_table(pairwise_file)
    human_seqs = dict(zip(df['human_peak'], df['human_seq']))
    primate_seqs = dict(zip(df['primate_peak'], df['primate_seq']))
    return human_seqs, primate_seqs

# ============================
# Load Alignment Data
# ============================

hc_file = sys.argv[3]
hg_file = sys.argv[4]

hc_peak_map = load_peak_mapping(hc_file)
hg_peak_map = load_peak_mapping(hg_file)

human_seqs, chimp_seqs = load_sequence_dicts(hc_file)
_, gorilla_seqs = load_sequence_dicts(hg_file)

# Filter shared peaks
shared_peaks = hc_peak_map.keys() & hg_peak_map.keys()
peak_triplets = {k: (hc_peak_map[k], hg_peak_map[k]) for k in shared_peaks}

# ============================
# MUSCLE5 Alignment
# ============================

def run_muscle5(human_peak, chimp_peak, gorilla_peak):
    input_fa = os.path.join(alignment_dir, f'{human_peak}.input.fa')
    output_fa = os.path.join(alignment_dir, f'{human_peak}.align.fa')

    with open(input_fa, 'w') as f:
        f.write(f'>Human\n{human_seqs[human_peak]}\n')
        f.write(f'>Chimpanzee\n{chimp_seqs[chimp_peak]}\n')
        f.write(f'>Gorilla\n{gorilla_seqs[gorilla_peak]}\n')

    try:
        cmd = f"muscle -align {input_fa} -output {output_fa} -threads 4"
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=30)
    except subprocess.TimeoutExpired:
        print(f'MUSCLE timeout on peak: {human_peak}')

# ============================
# PhyloFit Execution
# ============================

def run_phylofit(peak_name):
    aligned_fa = os.path.join(alignment_dir, f'{peak_name}.align.fa')
    try:
        cmd = f"phyloFit --tree '((Human,Chimpanzee),Gorilla)' --subst-mod REV --EM --quiet --precision MED --nrate 4 --out-root {peak_name} {aligned_fa}"
        subprocess.run(cmd, shell=True, timeout=30)
    except subprocess.TimeoutExpired:
        print(f'PhyloFit timeout on peak: {peak_name}')
    else:
        extract_newick_tree(peak_name)

def extract_newick_tree(peak_name):
    mod_file = os.path.join(phylofit_dir, f'{peak_name}.mod')
    tree_file = os.path.join(phylofit_dir, f'{peak_name}.newick')

    try:
        with open(mod_file) as f:
            for line in f:
                line=line.strip()
                if line.startswith('TREE'):
                    tree = line[6:]
                    with open(tree_file, 'w') as out:
                        out.write(tree)
    except FileNotFoundError:
        run_phylofit(peak_name)

# ============================
# Execution
# ============================

if __name__ == '__main__':
    # Run MUSCLE alignments
    triplets = [(h, c, g) for h, (c, g) in peak_triplets.items()]
    with Pool(processes=4) as pool:
        pool.starmap(run_muscle5, triplets)

    # Identify completed alignments
    aligned_files = [f for f in os.listdir(alignment_dir) if f.endswith('.align.fa')]
    peak_names = [f.split('.')[0] for f in aligned_files]

    # Run PhyloFit
    with Pool(processes=4) as pool:
        pool.map(run_phylofit, peak_names)
