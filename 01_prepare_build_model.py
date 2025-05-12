#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import sys
import random
import numpy as np
import pandas as pd
import portion as P
import pysam
import datetime


# In[5]:


#specify the input files, including sequencing peak files and genome files, as well as the output files, including positive set and negative set.
input_peak_file=sys.argv[1]
input_genome_file=sys.argv[2]
pos_output_file=sys.argv[3]
neg_output_file=sys.argv[4]


# #Construct the positive set

# In[6]:


#Loading sequencing files with pandas
df_ref=pd.read_table(input_peak_file,header=None)
df_ref.rename(columns={0:'chrom',1:'start',2:'end',9:'summit'},inplace=True)

#specify the chromosome ID
chromsome_list=['chr'+str(i) for i in range(1,23)]    #常染色体
chromsome_list.extend(['chrX','chr2A','chr2B','chr2a','chr2b'])                 #性染色体和2号染色体
df_ref=df_ref[df_ref['chrom'].apply(lambda x:x in chromsome_list)]


# In[ ]:


#Extract sequences based on genomic coordinates and save the results in FASTA format.
df_ref['pos_peak']=df_ref['chrom'].map(str)+':'+df_ref['start'].map(str)+'-'+df_ref['end'].map(str)
fasta_open=pysam.Fastafile(input_genome_file)
with open(pos_output_file,'w')as f:
    for index,row in df_ref.iterrows():
        chrom=row['chrom']
        start=row['start']
        end=row['end']
        peak_name=row['pos_peak']
        try:
            seq=fasta_open.fetch(chrom,start,end).upper()
            f.write(f'>{peak_name}\n')
            f.write(seq+'\n')
        except:
            print(peak_name)
            continue
fasta_open.close()


# In[ ]:





# #Construct the negative set

# In[ ]:


#加载hg38 repeatmasker信息（这些区域含有重复序列和卫星DNA，考虑阴性集前应首先排除这些区域）
repeatmasker_file=f'/home/wanglinxiao/wangzhen/genome_annotation/{genome_version}/repeat_masker/repeat_masker.csv'
df_repeat=pd.read_csv(repeatmasker_file)

#只保留低复杂度区域
df_repeat_region=df_repeat[df_repeat['repClass'] == 'Low_complexity'][['genoName','genoStart','genoEnd']]
df_repeat_region.columns=['chrom','start','end']
df_repeat_region=df_repeat_region[df_repeat_region['chrom'].apply(lambda x:x in chromsome_list)]

