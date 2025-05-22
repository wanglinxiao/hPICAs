# **Description**
This code demonstrates how to use a pre-trained CNN model to predict the effects of genetic variations that occurred in chromatin accessibility regions during the evolutionary transition from ancestral to modern humans. 
By introducing mutations individually into the ancestral sequence and computing the change in prediction values before and after each mutation, the code enables an independent assessment of the impact of each variant.

# **Usage**
`python cal_variation_effect.py human_seq anc_seq model_path output_file`
### **Input files**
`human_seq anc_seq`: Human chromatin accessibility regions and their corresponding ancestral sequences.

`model path`: A pre-trained CNN from 111 cell types.

`output_file`: output result file.

# **Help**
# **Author**
