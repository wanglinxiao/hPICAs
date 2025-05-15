## **Description**
This section of the code is used to evaluate the cross-species predictive capability of the CNN model. It includes the construction of positive and negative datasets, CNN architecture, the training process, and the saliency detection on the input sequences.

## **Usage**
### **1. Prepare for training CNN**
`python 01_prepare_build_model.py input_OCR.bed hg38.fa repeatmasker_file output_pos.fa output_neg.fa`
#### **Input files**
`input_OCR.bed`: ATAC-seq-derived chromatin accessible regions (narrowPeak format).

`hg38.fa`: Reference genome file for the corresponding species

`repeatmasker_file`: Repeating element file for the corrsponding species, downloaded from the UCSC RepeatMasker track.

`output_pos.fa`: The positive set required for training the CNN

`output_neg.fa`: The negative set required for training the CNN

### **2. CNN architecture and training**
`python 02_build_model.py input_pos.fa input_neg.fa model_result_file`
#### **Input files**
`input_pos.fa input_neg.fa`: The positive and negative sets required for training the CNN.

`model_result_file`: Trained model weights file (best validation AUROC) 

### **3. Identify the salient regions from the input sequences**
Apply Captum's Integrated Gradients (https://captum.ai/docs/extension/integrated_gradients) to detect feature sequences (length = 20bp) in each chromatin accessible region.

`python 03_identify_salient_region.py input_OCR.fa output_result_file input_model`
#### **Input files**
`input_OCR.fa`: ATAC-seq-derived chromatin accessible regions (FASTA format).

`output_result_file`: The output file contains the salient regions for each open chromatin region, along with the corresponding genomic coordinates.

`input_model`: Trained model weights file (*.pth.tar)

### **4. Motif discovery from salient regions**
`python 04_TomTom.py salient_region.bed output_motif_file tomtom_output_dir motif_database`
#### **Input files**
`salient_region.bed`: Feature sequences identified in each input sequence using Integrated Gradients.

`output_motif_file`: Convert each feature sequence to MEME format (see [https://meme-suite.org/meme/doc/meme-format.html]).

`tomtom_output_dir`: TOMTOM output directory.

`motif_database`: Motif database file

## **Dependencies**
Numpy == 1.26.4, pandas == 2.2.2, pysam == 0.19.0, torch == 1.13.1+cu117
