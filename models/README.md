## **Description**
This section of the code is used to evaluate the cross-species predictive capability of the CNN model. It includes the construction of positive and negative datasets, the model architecture, the training process, and the saliency detection on the input sequences.

## **Usage**
### **1. Prepare datasets for training CNN**
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
`input_pos.fa input_neg.fa`: The positive and negative set files required for training the CNN.

`model_result_file`: Optimal model parameters (best validation AUROC) 

### **3. Identify the salient regions from the input sequences**
Apply Captum's Integrated Gradients (https://captum.ai/docs/extension/integrated_gradients) to detect feature sequences (length = 20bp) in each chromatin accessible region.

`python 03_identify_salient_region.py input_OCR.fa output_result_file input_model`
#### **Input files**
`input_OCR.fa`: ATAC-seq-derived chromatin accessible regions (FASTA format).

`output_result_file`: The output file contains the salient region for each open chromatin region, along with the corresponding genomic coordinates.

`input_model`: CNN model file (*.pth)

### **4. Identify the salient regions from the input sequences**



## **Dependencies**
