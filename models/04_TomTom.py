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

salient_region_file=sys.argv[1]
query_motif_file=sys.argv[2]
tomtom_output_dir=sys.argv[3]
motif_database_file=sys.argv[4]

df_SR=pd.read_csv(salient_region_file)
df_high_pred=df_SR[df_SR['Seq pred_value'] > 0.8].copy()

def obtain_kmer_feature_for_one_sequence(tokenizer,input_seq):
    
    input_seq=input_seq.replace('N','')
    number_of_kmers = len(input_seq) - tokenizer + 1
    kmer_list=[]
    
    for i in range(number_of_kmers):
        this_kmer = input_seq[i:(i+tokenizer)]
        kmer_list.append(this_kmer)
        
    return kmer_list

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

#k-mer tokenization
k = 7
SR_seq_list = df_high_pred['SR_seq'].tolist()
SR_kmer_dict = kmer_featurization(k=k,seq_list=SR_seq_list)

sorted_SR_dict=dict(sorted(SR_kmer_dict.items(), key=lambda x: x[1] , reverse=True))
SR_kmer=list(sorted_SR_dict.keys())[:50]

#convert k-mers to Tomtom input format.
SR_kmer_input=' '.join(SR_kmer)
command_iupac2meme="iupac2meme                     -dna                     %s > %s" %(SR_kmer_input,query_motif_file)
proc1=subprocess.run(command_iupac2meme,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

#TomTom (meme-suite)
command_tomtom="tomtom         -no-ssc         -oc %s         -verbosity 1         -min-overlap 5         -dist pearson         -evalue         -thresh 10         -time 300 %s %s" %(tomtom_output_dir,query_motif_file,motif_database_file)
proc2=subprocess.run(command_tomtom,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

