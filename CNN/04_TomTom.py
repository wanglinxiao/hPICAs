#!/usr/bin/env python
# coding: utf-8

"""
Motif Discovery from Salient Genomic Regions Using k-mer Analysis and Tomtom Comparison

This script identifies overrepresented k-mers in high-confidence salient regions,
converts them to MEME motif format, and compares them against known motifs using Tomtom.
"""

import sys
import subprocess
import itertools
import numpy as np
import pandas as pd


# ----------------------------- Utility Functions ----------------------------- #

def extract_kmers(sequence: str, k: int) -> list:
    """
    Extract k-mers from a DNA sequence, excluding ambiguous bases (e.g. 'N').
    """
    sequence = sequence.upper().replace('N', '')
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def count_kmers(sequences: list, k: int) -> dict:
    """
    Count k-mer frequencies across a list of sequences.
    Returns a dictionary mapping each k-mer to its total count.
    """
    bases = 'ATGC'
    kmer_space = {''.join(p): 0 for p in itertools.product(bases, repeat=k)}
    
    for seq in sequences:
        for kmer in extract_kmers(seq, k):
            if kmer in kmer_space:
                kmer_space[kmer] += 1
    return kmer_space


def run_command(command: str, description: str):
    """
    Run a shell command and check for errors.
    """
    result = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Error running {description}:\n{result.stderr.decode('utf-8')}")


# ----------------------------- Main Pipeline ----------------------------- #

def main():
    # -------- Input Arguments -------- #
    if len(sys.argv) != 5:
        print("Usage: python script.py <salient_region.csv> <output_meme.txt> <tomtom_output_dir> <motif_database.meme>")
        sys.exit(1)

    salient_region_file = sys.argv[1]
    output_motif_file = sys.argv[2]
    tomtom_output_dir = sys.argv[3]
    motif_database_file = sys.argv[4]

    # -------- Parameters -------- #
    KMER_SIZE = 7
    TOP_KMER_COUNT = 50
    CONFIDENCE_THRESHOLD = 0.8

    # -------- Load Data -------- #
    df = pd.read_csv(salient_region_file)
    df_high_conf = df[df['Seq_pred'] > CONFIDENCE_THRESHOLD].copy()

    if df_high_conf.empty:
        raise ValueError(f"No sequences found with prediction score > {CONFIDENCE_THRESHOLD}")

    sequences = df_high_conf['SR_seq'].tolist()

    # -------- K-mer Counting -------- #
    kmer_counts = count_kmers(sequences, KMER_SIZE)
    sorted_kmers = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)
    top_kmers = [kmer for kmer, count in sorted_kmers[:TOP_KMER_COUNT]]

    if not top_kmers:
        raise ValueError("No valid k-mers were extracted.")

    # -------- Convert to MEME format -------- #
    kmers_str = ' '.join(top_kmers)
    iupac2meme_cmd = f"iupac2meme -dna {kmers_str} > {output_motif_file}"
    run_command(iupac2meme_cmd, "iupac2meme")

    # -------- Run Tomtom -------- #
    tomtom_cmd = (
        f"tomtom -no-ssc -oc {tomtom_output_dir} -verbosity 1 "
        f"-min-overlap 5 -dist pearson -evalue -thresh 10 -time 300 "
        f"{output_motif_file} {motif_database_file}"
    )
    run_command(tomtom_cmd, "tomtom")

    print(f"Top {TOP_KMER_COUNT} k-mers written to {output_motif_file}")
    print(f"Tomtom results saved to {tomtom_output_dir}")


if __name__ == "__main__":
    main()
