## **Description**
This section of the code is used to evaluate the cross-species predictive capability of the CNN model. It includes the construction of positive and negative datasets, the model architecture, the training process, and the saliency detection on the input sequences.

## **Usage**
### **Prepare datasets for training CNN**
`python 01_prepare_build_model.py input_OCR.bed hg38.fa repeatmasker_file output_pos.fa output_neg.fa`
#### **Input files**
`input_OCR.bed`: ATAC-seq-derived chromatin accessible regions (narrowPeak format).

`hg38.fa`: Reference genome file for the corresponding species

`output_pos.fa`: The positive set required for training the CNN

`output_neg.fa`: The negative set required for training the CNN

`repeatmasker_file`: Repeating element file for the corrsponing species, downloaded from the UCSC RepeatMasker track.

### **CNN architecture and training**



## **Dependencies**
