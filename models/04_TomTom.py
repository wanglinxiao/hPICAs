#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import pysam
import itertools
import numpy as np
import pandas as pd
import subprocess
from Bio import SeqIO

# Input arguments
salient_region_file = sys.argv[1]           # CSV file containing salient regions
query_motif_file = sys.argv[2]              # Output file for k-mer motifs in MEME format
tomtom_output_dir = sys.argv[3]             # Output directory for Tomtom results
motif_database_file = sys.argv[4]           # Reference motif database in MEME format

# Load salient region data and filter high-confidence predictions
df_SR = pd.read_csv(salient_region_file)
df_high_pred = df_SR[df_SR['Seq pred_value'] > 0.8].copy()

"""Extract k-mers from a sequence, excluding ambiguous base 'N'."""
def obtain_kmer_feature_for_one_sequence(tokenizer,input_seq):
    input_seq=input_seq.replace('N','')
    number_of_kmers = len(input_seq) - tokenizer + 1
    kmer_list=[]
    
    for i in range(number_of_kmers):
        this_kmer = input_seq[i:(i+tokenizer)]
        kmer_list.append(this_kmer)
    return kmer_list

"""Generate k-mer frequency dictionary for a list of sequences."""
def kmer_featurization(k,seq_list):
    base_seq='ATGC'
    tokenizer = k
    perm_seq_list=[''.join(i) for i in itertools.product(base_seq,repeat=tokenizer)]
    kmer_result={}
    for perm_seq in perm_seq_list:
        kmer_result[perm_seq] = 0

    for seq in seq_list:
        this_seq_kmer_list=obtain_kmer_feature_for_one_sequence(tokenizer=tokenizer,input_seq=seq)
        for this_seq_kmer in this_seq_kmer_list:
            kmer_result[this_seq_kmer] += 1  
    return kmer_result

# Parameters
k = 7
SR_seq_list = df_high_pred['SR_seq'].tolist()

# Compute top k-mers from salient regions
SR_kmer_dict = kmer_featurization(k, SR_seq_list)
sorted_SR_dict = dict(sorted(SR_kmer_dict.items(), key=lambda x: x[1], reverse=True))
SR_kmers = list(sorted_SR_dict.keys())[:50]  # Top 50 most frequent k-mers

# Convert k-mers to IUPAC MEME format for Tomtom input
SR_kmer_input = ' '.join(SR_kmers)
iupac_command = "iupac2meme -dna %s > %s" % (SR_kmer_input,query_motif_file)
subprocess.run(iupac_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# Run Tomtom to compare discovered motifs to known motif database
tomtom_command = "tomtom -no-ssc -oc %s -verbosity 1 -min-overlap 5 -dist pearson -evalue -thresh 10 -time 300 %s %s" % (tomtom_output_dir,query_motif_file,motif_database_file)
subprocess.run(tomtom_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
