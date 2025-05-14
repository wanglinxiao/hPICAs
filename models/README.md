## **Description**
这部分代码应用于评估CNN的跨物种预测能力，包括阳性集和阴性集的构建，模型架构以及训练过程，在输入序列中进行显著性检测。
This section of the code is used to evaluate the cross-species predictive capability of the CNN model. It includes the construction of positive and negative datasets, the model architecture, the training process, and the saliency detection on the input sequences.

`python 01_prepare_build_model.py {peak_file} {genome_file} {output_pos_file} {output_neg_file} {UCSC_repeatmasker_file}`

#构建模型前，准备阳性集和阴性集
python 01_prepare_build_model.py {peak_file} {genome_file} {output_pos_file} {output_neg_file} {UCSC_repeatmasker_file}
peak_file: 输入的OCR文件，narrow peak format
genome file: 输入的基因组文件，fasta format
输出文件：fasta format

