#!/usr/bin/env python
# coding: utf-8

import os
import sys
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from torchsummary import summary

pos_file=sys.argv[1]
neg_file=sys.argv[2]
model_save_file=sys.argv[3]

#ont-hot coding
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
    
    return hot

def generate_one_hot_seq(seq_list):
    onehot_seq_list=[]
    for seq in seq_list:
        onehot_seq=seq_to_hot(seq)
        onehot_seq_list.append(onehot_seq)
    
    onehot_seq_array=np.array(onehot_seq_list)
    return onehot_seq_array

#padding sequence
def padding_sequence(seq):
    ref_seq_length = len(seq)
    if ref_seq_length < 1000:
        padding_seq=seq.center(1000,'N')
    else:
        center=int(ref_seq_length/2)
        padding_seq=seq[center-500:center+500]
        
    return padding_seq

#Split into training, validation, and test sets.
def seq_label(pos_file,neg_file,sign):
    pos_seq_dict={}
    neg_seq_dict={}

    for line in open(pos_file,'r'):
        line=line.strip()
        if line.startswith('>'):
            pos_peak=line[1:]
        else:
            pos_seq=line.upper()
            pos_padding_seq=padding_sequence(pos_seq)
            pos_seq_dict[pos_peak] = pos_padding_seq
            
    for line in open(neg_file,'r'):
        line=line.strip()
        if line.startswith('>'):
            neg_peak=line[1:]
        else:
            neg_seq=line.upper()
            neg_padding_seq=padding_sequence(neg_seq)
            neg_seq_dict[neg_peak] = neg_padding_seq

    if sign == 'train':
        pos_train_seq=[]
        neg_train_seq=[]
        for pos_peak in list(pos_seq_dict.keys()):
            peak_chrom=re.findall('(.+):(.+)-(.+)',pos_peak)[0][0]
            if (peak_chrom != 'chr1') and (peak_chrom != 'chr2'):
                pos_train_seq.append(pos_seq_dict[pos_peak])
        for neg_peak in list(neg_seq_dict.keys()):
            peak_chrom=re.findall('(.+):(.+)-(.+)',neg_peak)[0][0]
            if (peak_chrom != 'chr1') and (peak_chrom != 'chr2'):
                neg_train_seq.append(neg_seq_dict[neg_peak])
                
        pos_train_label=[1 for x in range(len(pos_train_seq))]
        neg_train_label=[0 for x in range(len(neg_train_seq))]
        
        total_seq=pos_train_seq+neg_train_seq
        total_label=pos_train_label+neg_train_label
        
    elif sign == 'val':
        pos_val_seq=[]
        neg_val_seq=[]
        for pos_peak in list(pos_seq_dict.keys()):
            peak_chrom=re.findall('(.+):(.+)-(.+)',pos_peak)[0][0]
            if peak_chrom == 'chr1' :
                pos_val_seq.append(pos_seq_dict[pos_peak])
        for neg_peak in list(neg_seq_dict.keys()):
            peak_chrom=re.findall('(.+):(.+)-(.+)',neg_peak)[0][0]
            if peak_chrom == 'chr1' :
                neg_val_seq.append(neg_seq_dict[neg_peak])
                
        pos_val_label=[1 for x in range(len(pos_val_seq))]
        neg_val_label=[0 for x in range(len(neg_val_seq))]
        
        total_seq=pos_val_seq+neg_val_seq
        total_label=pos_val_label+neg_val_label
        
    elif sign == 'test':
        pos_test_seq=[]
        neg_test_seq=[]
        for pos_peak in list(pos_seq_dict.keys()):
            peak_chrom=re.findall('(.+):(.+)-(.+)',pos_peak)[0][0]
            if peak_chrom == 'chr2':
                pos_test_seq.append(pos_seq_dict[pos_peak])
        for neg_peak in list(neg_seq_dict.keys()):
            peak_chrom=re.findall('(.+):(.+)-(.+)',neg_peak)[0][0]
            if peak_chrom == 'chr2':
                neg_test_seq.append(neg_seq_dict[neg_peak])
                
        pos_test_label=[1 for x in range(len(pos_test_seq))]
        neg_test_label=[0 for x in range(len(neg_test_seq))]
        
        total_seq=pos_test_seq+neg_test_seq
        total_label=pos_test_label+neg_test_label
        
    return total_seq,total_label

train_seq,train_label=seq_label(pos_file=pos_file,neg_file=neg_file,sign='train')
val_seq,val_label=seq_label(pos_file=pos_file,neg_file=neg_file,sign='val')
test_seq,test_label=seq_label(pos_file=pos_file,neg_file=neg_file,sign='test')

train_label=np.array(train_label)
val_label=np.array(val_label)
test_label=np.array(test_label)

train_data=generate_one_hot_seq(train_seq)
val_data=generate_one_hot_seq(val_seq)
test_data=generate_one_hot_seq(test_seq)

#create pytorch dataset
class TrainDataset(data.Dataset):
    def __init__(self):
        self.Data=train_data
        self.Label=train_label
        
    def __getitem__(self,index):
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label
    
    def __len__(self):
        return len(self.Data)

class ValDataset(data.Dataset):
    def __init__(self):
        self.Data=val_data
        self.Label=val_label
        
    def __getitem__(self,index):
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label
    
    def __len__(self):
        return len(self.Data)
    
class TestDataset(data.Dataset):
    def __init__(self):
        self.Data=test_data
        self.Label=test_label
        
    def __getitem__(self,index):
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label
    
    def __len__(self):
        return len(self.Data)

Train=TrainDataset()
Val=ValDataset()
Test=TestDataset()

train_loader=data.DataLoader(Train,batch_size=64,shuffle=True,num_workers=4)
val_loader=data.DataLoader(Val,batch_size=64,shuffle=False,num_workers=4)
test_loader=data.DataLoader(Test,batch_size=64,shuffle=False,num_workers=4)

#model architecture
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
        return out1,out2
    
    def make_layers(self,cfg):
        layers=[]
        in_channels=4
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool1d(kernel_size=2,stride=2))
            elif x == 280:
                layers += [nn.Conv1d(in_channels,x,kernel_size=13),nn.BatchNorm1d(x),nn.Threshold(0,1e-6)]
                in_channels=x
            elif x == 180:
                layers += [nn.Conv1d(in_channels,x,kernel_size=11),nn.BatchNorm1d(x),nn.Threshold(0,1e-6)]
                in_channels=x
            elif x == 120:
                layers += [nn.Conv1d(in_channels,x,kernel_size=9),nn.BatchNorm1d(x),nn.Threshold(0,1e-6)]
                in_channels=x                
        layers += [nn.AdaptiveAvgPool1d(3)]
        return nn.Sequential(*layers)

device = torch.device('cuda:0')

criterion=nn.BCELoss()
criterion=criterion.to(device)

optimizer=optim.Adam(VGG16.parameters(),lr=0.0001)

VGG16=VGG('VGG16')
VGG16=VGG16.to(device)

def train(model,dataloader,optimizer,criterion):
    print('Start train')
    model.train()
    train_running_loss=0
    for train_seq,train_labels in dataloader:   
        train_seq=train_seq.to(device)   
        train_labels=train_labels.to(device)

        _,train_pred=model(train_seq)
        train_pred=train_pred.to(torch.float32)     
        train_labels=train_labels.to(torch.float32) 
        loss=criterion(train_pred,train_labels) 

        optimizer.zero_grad() 
        loss.backward()       
        optimizer.step()     
        train_running_loss += loss.item()   
        
    train_loss=train_running_loss/len(dataloader)

    return train_loss

def validate(model,dataloader,criterion):
    print('Start validation')
    model.eval()
    val_running_loss=0
    val_all_preds,val_all_labels=[],[]
    for val_seq,val_labels in dataloader:
        val_seq=val_seq.to(device)
        val_labels=val_labels.to(device)

        _,val_pred=model(val_seq)
        val_pred=val_pred.to(torch.float32)     
        val_labels=val_labels.to(torch.float32) 
        loss=criterion(val_pred,val_labels)
        val_running_loss += loss.item()

        val_all_preds.extend(val_pred.detach().cpu().numpy())
        val_all_labels.extend(val_labels.cpu().numpy())

    val_auc=roc_auc_score(val_all_labels,val_all_preds)
    val_loss=val_running_loss/len(dataloader)
   
    return val_loss,val_auc

train_loss=[]
val_loss,val_auc=[],[]
max_val_auc=None
epochs=30
counter=0
patience=7

#Start training
for epoch in range(1,epochs+1):  
    print(f'Epoch:{epoch}')
    train_epoch_loss=train(model=VGG16,dataloader=train_loader,criterion=criterion,optimizer=optimizer)
    train_loss.append(train_epoch_loss)
    print(f'Train Loss = {train_epoch_loss}')

#Start validation
    val_epoch_loss,val_epoch_auc=validate(model=VGG16,dataloader=val_loader,criterion=criterion)
    val_loss.append(val_epoch_loss)
    val_auc.append(val_epoch_auc)
    print(f'Validation Loss = {val_epoch_loss}')
    print(f'Validation Auc = {val_epoch_auc}')
     
#Early stopping
    if max_val_auc is None:         
        max_val_auc=val_epoch_auc
        torch.save({'epoch':epoch,
            'model_state_dict':VGG16.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'Val_auc':val_epoch_auc},model_save_path)
    elif max_val_auc > val_epoch_auc: 
        counter += 1
        if counter >= patience:
            break
    else:                              
        max_val_auc = val_epoch_auc
        counter = 0
        torch.save({'epoch':epoch,
            'model_state_dict':VGG16.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'Val_auc':val_epoch_auc},model_save_path)