#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from torchsummary import summary
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

input_fasta_file=sys.argv[1]
output_file=sys.argv[2]
model_file=sys.argv[3]

window_size=20
device = torch.device('cuda:0')

seq_dict = {rec.id : str(rec.seq) for rec in SeqIO.parse(input_fasta_file, "fasta")
df_seq=pd.DataFrame({'Peak name':list(seq_dict.keys()),'Seq':list(seq_dict.values())})
df_seq['Seq_length']=df_seq['Seq'].apply(len)

cfg={'VGG16':[280,280,'M',180,180,'M',120,120,'M']}

class VGG(nn.Module):
    def __init__(self,vgg_name):
        super(VGG,self).__init__()
        self.feature=self.make_layers(cfg[vgg_name])
        self.fc1=nn.Linear(360,64)
        self.fc2=nn.Linear(64,1)
        
    def forward(self,x):
        out=self.feature(x)          
        out=out.view(out.size(0),-1) 
        out1=F.relu(self.fc1(out))   
        out2=torch.sigmoid(self.fc2(out1))
        out2=out2.view(-1) 
        return out2
    
    def make_layers(self,cfg):
        layers=[]
        in_channels=4
        for x in cfg:
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

VGG16=VGG('VGG16')
VGG16=VGG16.to(device)
model_dict=torch.load(model_file)
VGG16.load_state_dict(model_dict['model_state_dict'])
VGG16.eval()

#one-hot coding
def seq_to_hot(seq):
    seq=seq.replace('a','A')
    seq=seq.replace('c','C')
    seq=seq.replace('g','G')
    seq=seq.replace('t','T')
    seq=seq.replace('n','N')

    Aseq=seq
    Aseq=Aseq.replace('A','1')
    Aseq=Aseq.replace('C','0')
    Aseq=Aseq.replace('G','0')
    Aseq=Aseq.replace('T','0')
    Aseq=Aseq.replace('N','0')
    Aseq=np.asarray(list(Aseq),dtype='float32')

    Cseq=seq
    Cseq=Cseq.replace('A','0')
    Cseq=Cseq.replace('C','1')
    Cseq=Cseq.replace('G','0')
    Cseq=Cseq.replace('T','0')
    Cseq=Cseq.replace('N','0')
    Cseq=np.asarray(list(Cseq),dtype='float32')

    Gseq=seq
    Gseq=Gseq.replace('A','0')
    Gseq=Gseq.replace('C','0')
    Gseq=Gseq.replace('G','1')
    Gseq=Gseq.replace('T','0')
    Gseq=Gseq.replace('N','0')
    Gseq=np.asarray(list(Gseq),dtype='float32')

    Tseq=seq
    Tseq=Tseq.replace('A','0')
    Tseq=Tseq.replace('C','0')
    Tseq=Tseq.replace('G','0')
    Tseq=Tseq.replace('T','1')
    Tseq=Tseq.replace('N','0')
    Tseq=np.asarray(list(Tseq),dtype='float32')
    hot=np.vstack((Aseq,Cseq,Gseq,Tseq))
    
    hot=torch.from_numpy(hot)
    hot=torch.unsqueeze(hot,0)
    
    return hot

#Integrated gradient
def Integrated_gradient(ser):
    ref_seq=ser['Seq']
    if len(ref_seq) < 300:
        ref_seq=ref_seq.rjust(300,'N')
    
    peak_chrom=re.findall('(.+):(.+)-(.+)',ser['Peak name'])[0][0]
    peak_start=re.findall('(.+):(.+)-(.+)',ser['Peak name'])[0][1]
    peak_end=re.findall('(.+):(.+)-(.+)',ser['Peak name'])[0][2]
    
    ref_seq_onehot=seq_to_hot(ref_seq).to(device)
    ref_seq_pred=VGG16(ref_seq_onehot).item()
    baseline = torch.zeros_like(ref_seq_onehot).to(device)

    ig = IntegratedGradients(VGG16)
    attributions, delta = ig.attribute(ref_seq_onehot, baseline, n_steps=1000, return_convergence_delta=True)
    
    df_ISM=pd.DataFrame(attributions.cpu().detach().numpy().squeeze(0))
    df_ISM.index=['A','C','G','T']
    df_ISM.columns=[i for i in range(0,len(ref_seq))]

    df_ISM.loc['max']=df_ISM.max()
    df_ISM.loc['rolling']=df_ISM.loc['max'].rolling(window=window_size).sum()
    
    max_number=df_ISM.loc['rolling'].max()
    SR_end=df_ISM.columns[df_ISM.loc['rolling'].isin([max_number])][0]
    SR_start=SR_end-(window_size-1)

    SR_seq=ref_seq[SR_start:SR_end+1] 
    SR_genomic_start=eval(peak_start)+SR_start
    SR_genomic_end=eval(peak_start)+SR_end

    return peak_chrom,SR_genomic_start,SR_genomic_end,SR_seq,ref_seq_pred

df_seq[['SR_chrom','SR_start','SR_end','SR_seq','Seq pred_value']]=df_seq.apply(Integrated_gradient,result_type='expand',axis=1)

#save salient region to file
df_seq.to_csv(output_file,index=False)
