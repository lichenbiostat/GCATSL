# GCATSL: Graph Contextualized Attention Network for Predicting Synthetic Lethality in Human Cancers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4522679.svg)](https://zenodo.org/record/4522679#.YCKCi-gzaUk) 

Our Software and test data have been archived at the dedicated repository of Zenodo. The archived Software version is available [here](https://zenodo.org/record/4522679#.YCKCi-gzaUk)



GCATSL is a deep learning model that can be used for SL prediction. As shown in the following flowchart (a), GCATSL first learns representations for nodes based on different feature graphs together with a known SL interaction graph, and then uses the learned node representations to reconstruct SL interaction matrix for SL prediction. 

![Image text](https://github.com/longyahui/GCATSL/blob/master/flowchart.jpg)
# Installation
GCATSL is implemented with Tensorflow library. For the detail instruction of installing Tensorflow, see the [guidence](https://www.tensorflow.org/install) on official website of Tensorflow.

## Requirements
You'll need to install the following packages in order to run the codes.
* Python 3.7
* Tensorflow 1.13.1
* numpy 1.16.2
* scipy 1.4.1
* sklearn

# Inputs
This Python script is designed to implement the GCATSL model. GCATSL model needs two kinds of data files in TXT format as inputs, i.e., `adj.txt` and `feature_x.txt`. `adj.txt` represents the file of known SL pairs, where the first and second columns denote the numbers of gene1 and gene2 (e.g., 1,2,3...) respectively, and the third column deontes the labels of SL pairs (i.e., 1 or 0). A toy example for `adj.txt` is available below, where the second row means that the first gene is confirmed to be associated with the fifth gene. 

A toy example of `adj.txt`

gene1|gene2|label
----|----|----|
1|5|1
3|9|1
5|30|1

Note that for convenient instruction, here we add names of genes and their label in the first line in the above example. You should remove these information when inputting them to the model. 

GCATSL can use multiple feature graphs of genes for SL prediction. `feature_x.txt` is a matrix that represents the 'x'-th feature graph for genes.  It should be noted that the dimensions of all feature graphs should be completely consistent. To reproduce the results for SL prediction, here we provided a set of input data (http://synlethdb.sist.shanghaitech.edu.cn/downloadPage.php) as examples, which are available in `./data/toy_examples`. Please see the [readme](https://github.com/longyahui/GCATSL/blob/master/data/readme.md) for the detailed explanations about the example data.

# Outputs
The outputs of GCATSL model include two files, i.e., `test_result.txt` and `log.txt`.  `test_result.txt` records the predicted scores for test samples. The first, second, third and fourth columns in the `test_result.txt` denote the number of gene1, the number of gene2, the predicted score for gene1-gene2 SL pair and their label, respectively. A toy example of `test_result.txt` is available below.

A toy example of `test_result.txt`

gene1|gene2|score|label
----|----|----|----
1|5|0.924|1
3|9|0.865|1
5|30|0.532|1

In addition, we used metrics AUC and AUPR to evaluate our proposed GCATSL model. `log.txt` records the values of metrics AUC and AUPR for each fold CV (Cross Validation) respectively. 

# Usage
First, you need to clone the repository or download source codes and data files. 

    $ git clone https://github.com/longyahui/GCATSL.git
 
You can see the optional arguments for GCATSL by "help" option:

    $ python source/main.py --help

You can easily run GCATSL with two steps, and then obtain the predicted results.
1) Unzip the file `./data/toy_examples/global interaction matrix.rar` in directory `./data/toy_examples/`.
2) Run `main.py` to obtain the predicted results as follows:

        python source/main.py --n_epoch 600 \
                              --n_head 2 \
                              --n_fold 5 \
                              --n_node 6375 \
                              --n_feature 3 \
                              --learning_rate 0.005 \
                              --weight_decay 0.0001 \
                              --dropout 0.7 \
                              --input_dir ../data/toy_examples/    \
                              --output_dir ../output/              \
                              --log_dir ../output/                 \


 

# Contact

    yahuilong@hnu.edu.cn 

