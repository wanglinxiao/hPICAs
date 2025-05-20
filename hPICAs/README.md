# **Description**
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

`python 02_FastML.py msa_dir tree_dir output_dir`
### **Input files**
`msa_dir tree_dir`: Containing the multiple sequence alignments and phylogenetic trees of all human OCRs in a single cell type.

`output_dir`: The output directory of FastML.

## **3. Simulate DNA sequence evolution**
We utilized Alisim to simulate the sequence evolutionary process and packaged the pipeline into an executable script (03_Alisim.py). This implementation allows users to select a pre-trained model corresponding to a specific cell type and, by inputting a single human sequence along with its corresponding ancestral sequence, simulate the evolutionary process and compute the associated p-value.

`python 03_Alisim.py model_file human_seq ancestral_seq tree_file output_file`
### **Input files**
`model_file`: A pre-trained CNN from 111 cell types
`human_seq ancestral_seq`: Input DNA sequence
`tree_file`: Fitted evolutionary tree file
`output_file`: output file

# **Help**
# **Author**
