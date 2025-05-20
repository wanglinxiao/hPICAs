# **Description**
这部分代码描述了如何推测human-chimpanzee的共同祖先序列以及如何通过模拟DNA序列进化过程确定与祖先序列的染色质开放性存在显著差异的区域。我们首先确定human序列在chimpanzee和gorilla基因上的同源序列（见materials and methods），利用muscle5完成多序列比对并利用phylofit基于比对结果拟合进化树。Fastml是一款基于物种进化关系来重构祖先序列的软件，可以使用多种算法重构祖先序列，重点在于对Indel和字符的精确重建。对于FastML，我们采取的碱基替代模型是GTR，选择最大似然法推测祖先序列可能存在的indel。
为了确定与human-chimpanzee共同祖先染色质开放性显著差异的OCR区域，我们利用Alisim模拟序列进化过程。我们将模拟过程整理成可执行的程序(03_Alisim.py)，允许使用者选择指定细胞类型的预训练模型，输入单个human序列以及对应的祖先序列即可模拟序列进化过程并计算p值。
This section describes the inference of the human-chimpanzee common ancestral sequences and the simulation-based approach used to identify regions with significantly increased chromatin accessibility compared to the ancestor.

# **Usage**
## **1. Multiple sequence alignment and fit the tree model**
We first identified orthologous sequences of the human open chromatin regions (OCRs) in chimpanzee and gorilla genomes (see Materials and Methods). Multiple sequence alignments were performed using MUSCLE5, and evolutionary trees were fitted from these alignments using phyloFit.

`python 01_muscle5_and_phyloFit.py msa_output_dir phylofit_output_dir hc_file hg_file`
### **Input files**
`msa_output_dir phylofit_output_dir`: Specify the output directory for multiple sequence alignment and phyloFit results.

`hc_file hg_file`: The orthologous sequences of all human OCRs in a single cell type within the chimpanzee (hc_file) and gorilla (hg_file) genomes.

## **2. Reconstruct ancestral sequences**
FastML, a tool designed for reconstructing ancestral sequences based on phylogenetic relations. FastML implements several algorithms that reconstruct the ancestral sequences with emphasis on an accurate reconstruction of both indels and characters. For ancestral inference, we used the General Time Reversible (GTR) model of nucleotide substitution and applied the maximum likelihood method to estimate potential indels.

``
