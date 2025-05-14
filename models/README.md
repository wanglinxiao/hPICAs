## **Description**
This section of the code is used to evaluate the cross-species predictive capability of the CNN model. It includes the construction of positive and negative datasets, the model architecture, the training process, and the saliency detection on the input sequences.

## **Dependencies**


## **Usage**
### **prepare datasets for training CNN**
`python 01_prepare_build_model.py input_OCR.bed hg38.fa output_pos.fa output_neg.fa repeatmasker_file`
### **Input files**
input_OCR.bed:

#构建模型前，准备阳性集和阴性集
python 01_prepare_build_model.py {peak_file} {genome_file} {output_pos_file} {output_neg_file} {UCSC_repeatmasker_file}
peak_file: 输入的OCR文件，narrow peak format
genome file: 输入的基因组文件，fasta format
输出文件：fasta format

ATAC-seq
