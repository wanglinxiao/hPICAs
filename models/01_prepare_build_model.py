#!/usr/bin/env python
# coding: utf-8

"""
Construct positive and negative DNA sequence sets from chromatin accessibility peaks
and genome reference, suitable for supervised learning tasks.

Inputs:
    1. input_peak_file        - Peak regions file (BED format or MACS2 narrowPeak)
    2. input_genome_file      - Reference genome in FASTA format (indexed by samtools faidx)
    3. repeat_masker_file     - RepeatMasker annotated regions (CSV format)
    4. pos_output_file        - Output FASTA file for positive sequences
    5. neg_output_file        - Output FASTA file for GC-matched negative sequences
"""

import os
import sys
import re
import random
import datetime
import numpy as np
import pandas as pd
import portion as P
import pysam

# Parse command-line arguments
input_peak_file = sys.argv[1]
input_genome_file = sys.argv[2]
repeat_masker_file = sys.argv[3]
pos_output_file = sys.argv[4]
neg_output_file = sys.argv[5]

### STEP 1: Construct Positive Set ###

# Load peaks and define standard chromosome list
df_peaks = pd.read_table(input_peak_file, header=None)
df_peaks.rename(columns={0: 'chrom', 1: 'start', 2: 'end', 9: 'summit'}, inplace=True)
df_peaks['peak_name'] = df_peaks['chrom'].astype(str) + ':' + df_peaks['start'].astype(str) + '-' + df_peaks['end'].astype(str)

valid_chroms = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chr2A', 'chr2B', 'chr2a', 'chr2b']
df_peaks = df_peaks[df_peaks['chrom'].isin(valid_chroms)]

# Extract genomic sequences from peaks
fasta = pysam.Fastafile(input_genome_file)
with open(pos_output_file, 'w') as fout:
    for _, row in df_peaks.iterrows():
        try:
            seq = fasta.fetch(row['chrom'], row['start'], row['end']).upper()
            fout.write(f">{row['peak_name']}\n{seq}\n")
        except Exception as e:
            print(f"Skipping: {row['peak_name']} due to error: {e}")
fasta.close()

### STEP 2: Construct Negative Set ###

# Load RepeatMasker low-complexity regions
df_repeat = pd.read_csv(repeat_masker_file)
df_lc = df_repeat[df_repeat['repClass'] == 'Low_complexity'][['genoName', 'genoStart', 'genoEnd']]
df_lc.columns = ['chrom', 'start', 'end']
df_lc = df_lc[df_lc['chrom'].isin(valid_chroms)]

# Merge peaks and low-complexity regions to define exclusion zones
df_merge = pd.concat([df_peaks[['chrom', 'start', 'end']], df_lc])
df_merge = df_merge.set_index('chrom')
merged_regions = df_merge.groupby('chrom')[['start', 'end']].apply(lambda x: x.to_numpy().tolist()).to_dict()

# Compute accessible "gap" intervals not overlapping with peaks or repeats
gap_intervals = {}
for chrom, regions in merged_regions.items():
    if len(regions) < 2:
        continue
    merged = P.empty()
    for start, end in regions:
        merged |= P.closed(start, end)
    all_range = P.closed(merged.lower, merged.upper)
    gaps = all_range - merged
    gap_intervals[chrom] = [[i.lower, i.upper] for i in gaps]

# GC content computation
def gc_content(seq):
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq) if len(seq) > 0 else 0

# Load positive sequences for GC matching
positive_seqs = {}
with open(pos_output_file) as fin:
    for line in fin:
        line = line.strip()
        if line.startswith('>'):
            current_peak = line[1:]
        else:
            positive_seqs[current_peak] = line

# Sample GC-matched negative sequences
fasta = pysam.Fastafile(input_genome_file)
neg_seqs = {}
gc_tolerance = 0.05
max_sampling_time = datetime.timedelta(seconds=180)

for peak_id, pos_seq in positive_seqs.items():
    pos_gc = gc_content(pos_seq)
    seq_len = len(pos_seq)
    chrom = re.match(r'(.+):', peak_id).group(1)

    if chrom not in gap_intervals:
        continue

    start_time = datetime.datetime.now()
    while True:
        if datetime.datetime.now() - start_time > max_sampling_time:
            break
        region = random.choice(gap_intervals[chrom])
        rand_start = random.randint(region[0], region[1] - seq_len)
        rand_end = rand_start + seq_len
        try:
            neg_seq = fasta.fetch(chrom, rand_start, rand_end).upper()
            if abs(gc_content(neg_seq) - pos_gc) <= gc_tolerance:
                neg_id = f"{chrom}:{rand_start}-{rand_end}"
                neg_seqs[neg_id] = neg_seq
                break
        except:
            continue
fasta.close()

# Write negative sequences to FASTA
with open(neg_output_file, 'w') as fout:
    for region_id, seq in neg_seqs.items():
        fout.write(f">{region_id}\n{seq}\n")
