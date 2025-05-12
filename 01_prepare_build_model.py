#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import random
import numpy as np
import pandas as pd
import portion as P
import pysam
import datetime

#specify the input files, including sequencing peak files and genome files, as well as the output files, including positive set and negative set.
input_peak_file=sys.argv[1]
input_genome_file=sys.argv[2]
repeat_masker_file=sys.argv[3]
pos_output_file=sys.argv[4]
neg_output_file=sys.argv[5]

##Construct the positive set

#Loading sequencing files with pandas
df_ref=pd.read_table(input_peak_file,header=None)
df_ref.rename(columns={0:'chrom',1:'start',2:'end',9:'summit'},inplace=True)
df_ref['peak_name']=df_ref['chrom'].map(str)+':'+df_ref['start'].map(str)+'-'+df_ref['end'].map(str)

#specify the chromosome ID
chromsome_list=['chr'+str(i) for i in range(1,23)]
chromsome_list.extend(['chrX','chr2A','chr2B','chr2a','chr2b'])
df_ref=df_ref[df_ref['chrom'].apply(lambda x:x in chromsome_list)]

#Extract sequences based on genomic coordinates and save the results in FASTA format.
fasta_open=pysam.Fastafile(input_genome_file)
with open(pos_output_file,'w')as f:
    for index,row in df_ref.iterrows():
        chrom=row['chrom']
        start=row['start']
        end=row['end']
        peak_name=row['peak_name']
        try:
            seq=fasta_open.fetch(chrom,start,end).upper()
            f.write(f'>{peak_name}\n')
            f.write(seq+'\n')
        except:
            print(peak_name)
            continue
fasta_open.close()

##Construct the negative set

#Load UCSC RepeatMasker annotations to extract low-complexity regions.
df_repeat=pd.read_csv(repeat_masker_file)
df_repeat_region=df_repeat[df_repeat['repClass'] == 'Low_complexity'][['genoName','genoStart','genoEnd']]
df_repeat_region.columns=['chrom','start','end']
df_repeat_region=df_repeat_region[df_repeat_region['chrom'].apply(lambda x:x in chromsome_list)]

#Merge low-complexity regions with open chromatin regions.
df_chromatin=df_ref[['chrom','start','end']]
df_chromatin.columns=['chrom','start','end']
df_merge=pd.concat([df_chromatin,df_repeat_region],axis=0)

df_merge=df_merge.set_axis(df_merge['chrom'],axis=0)
merge_peak_dict=df_merge.drop('chrom',axis=1).groupby('chrom').apply(lambda x : x.to_numpy().tolist()).to_dict()

#Generate downsampled regions required for the construction of the negative set.
gap_dict={}
for chrom in list(merge_peak_dict.keys()):
    if len(merge_peak_dict[chrom]) < 2:
        continue
    else:
        intersect_region=P.empty()
        for value in merge_peak_dict[chrom]:
            region=P.closed(value[0],value[1])              
            intersect_region=intersect_region.union(region) 
        
        chrom_lower=intersect_region.lower
        chrom_upper=intersect_region.upper
        chrom_all=P.closed(chrom_lower,chrom_upper)             
        gap_dict[chrom]=chrom_all.difference(intersect_region)  

gap_list=[]
for key,values in gap_dict.items():
    for value in values:
        gap=[key,value.lower,value.upper]
        gap_list.append(gap)
        
df_gap=pd.DataFrame(gap_list,columns=['Chrom','GapStart','GapEnd'])
df_gap=df_gap.set_axis(df_gap['Chrom'],axis=0)
gap_dict=df_gap.drop('Chrom',axis=1).groupby('Chrom').apply(lambda x : x.to_numpy().tolist()).to_dict()

#Calculate the G-C content
def GC_content(seq):
    seq=seq.upper()
    g=seq.count('G')
    c=seq.count('C')
    seq_length=len(seq)
    gc_content=(g+c)/seq_length
    
    return gc_content

#Generate negative set by random sampling.
fasta_open=pysam.Fastafile(input_genome_file)
GC_content_diff=0.05
pos_seq_dict={}
neg_seq_dict={}

for line in open(pos_output_file):
    line=line.strip()
    if line.startswith('>'):
        peak=line[1:]
    else:
        seq=line
        pos_seq_dict[peak]=seq

for pos_peak,pos_seq in pos_seq_dict.items():
    pos_seq_gc=GC_content(pos_seq)            
    pos_seq_length=len(pos_seq)
    
    #Match chromosomes
    pos_chrom=re.findall('(.+):(.+)-(.+)',pos_peak)[0][0]
    neg_chrom=pos_chrom
    gap_candidate_region=gap_dict[neg_chrom]

    #Match sequence length
    time_limit = datetime.timedelta(seconds=180)
    start_time=datetime.datetime.now()
    while True:
        current_time=datetime.datetime.now()
        if current_time - start_time > time_limit:
            break
        
        gap_region=random.choice(gap_candidate_region)
        gap_start=gap_region[0]
        gap_end=gap_region[1]
        neg_region_start=random.randint(gap_start,gap_end)
        neg_region_end=neg_region_start+pos_seq_length
        if neg_region_end > gap_end:
            continue
            
        #Match G-C content
        neg_seq=fasta_open.fetch(neg_chrom,neg_region_start,neg_region_end).upper()
        neg_seq_gc=GC_content(neg_seq)           
        if (neg_seq_gc < pos_seq_gc-GC_content_diff) or (neg_seq_gc > pos_seq_gc+GC_content_diff):
            continue
        else:
            neg_region=f'>{neg_chrom}:{neg_region_start}-{neg_region_end}'
            neg_seq_dict[neg_region]=neg_seq
            break
    
fasta_open.close()

with open(neg_output_file,'w')as f1:
    for key,value in neg_seq_dict.items():
        f1.write(key+'\n')
        f1.write(value+'\n')

