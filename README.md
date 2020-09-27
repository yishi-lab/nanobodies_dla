# Nanobodies Analysis using deep learning models

The following library contains notebooks for a step by step analysis of nanobodies datasets using deep neural networks and a complete algorithm to explore and analyze the trained models.
The algorithm for analysis is still in optimization stage and adaptation for different models.  
They were all developed by Lirane Bitton under the supervision of Dina Schneidman at the Hebrew University of Jerusalem.

### Requirements
- python >= 3.7.3 (a [requirements.txt](https://github.com/yishi-lab/nanobodies_dla/blob/master/requirements.txt) is provided)
- [cd-hit](http://weizhongli-lab.org/cd-hit/) installed on your system
- [mafft](https://mafft.cbrc.jp/alignment/software/) installed on your system 

### Inputs
In data folder you can find two input dataset of nanobodies binding gst and has proteins. 

### Workflow
Start with the [dataset_explore](https://github.com/yishi-lab/nanobodies_dla/blob/master/dataset_explore.ipynb) notebook then continue with the [analysis one](https://github.com/yishi-lab/nanobodies_dla/blob/master/analysis.ipynb). For more statistics about cdr amino acids distribution go through the [cdr_posistion_stats](https://github.com/yishi-lab/nanobodies_dla/blob/master/cdr_position_stats.ipynb) notebook.

### Contacts
Lirane Bitton: lirane.bitton@mail.huji.ac.il
Dina Schneidman: dina.schneidman@mail.huji.ac.il