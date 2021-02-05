# GCATSL: Graph Contextualized Attention Network for Predicting Synthetic Lethality in Human Cancers
GCATSL is a deep learning model that can be used for SL prediction. GCATSL first learns representations for nodes based on different feature graphs together with a known SL graph, and then uses the learned node representations to reconstruct SL interaction matrix for SL prediction. 

# Installation
GCATSL is implemented in with Tensorflow library. For detail instruction of installing Tensorflow, see the [guidence on official website](https://www.tensorflow.org/install) of Tensorflow.

# Requirements
* Python 3.7
* Tensorflow 1.13.1
* numpy 1.16.2
* scipy 1.4.1
* sklearn

# Inputs
This Python script is designed to implement the GCATSL model. It needs two categories of input data in TXT format, i.e., `adj.txt` and `feature_x.txt`, for model training and testing. `adj.txt` represents the file of known SL pairs, where the first and second columns denote the serial numbers of gene1 and gene2 (e.g., 1,2,3...) respectively, and the third column deontes the labels (i.e., 1 or 0). `feature_x.txt` represents the feature graphs for SLs. 'x' is the number of feature graph and can be replaced by 1, 2, 3, etc. 

# Outputs

# Usage
First, you need to clone the repository or download source codes and data files. 

    $ git clone https://github.com/longyahui/GCATSL.git
 
You can see the valid parameters for GCATSL by help option:

    $ python source/main.py --help

You can easily run GCATSL with two steps, and then obtain the prediction results.
1) Unzip the file `./data/toy_examples/global interaction matrix.rar` to the same level directory
2) Run `main.py` to obtain the prediction results like:

        python source/main.py --n_epoch 600 \
                              --n_heads 2 \
                              --n_folds 5 \
                              --n_nodes 6375 \
                              --n_feature 3 \
                              --learning_rate 0.005 \
                              --weight_decay 0.0001 \
                              --dropout 0.7 \
                              --input_dir ../data/toy_examples/    \
                              --output_dir ../output/              \
                              --log_dir ../output/                 \


# Data description
* `/data/adj`: known SL interaction pairs.
* `/data/interaction`: adjacent matrix for SL pairs.
* `/data/feature_ppi_sparse`: features obtained based on PPI network.
* `/data/feature_Human_GOsim_BP`: features obtained based on GO terms (i.e., BP).
* `/data/feature_Human_GOsim_CC`: features obtained based on GO terms (i.e., CC).

# Example input data for 5-fold CV
In this work, we conducted 5-fold CV to evaluate the performance of our proposed GCATSL model. For reproducing the results of our model on database SynLethDB in 5-fold CV, we provided a set of input exmple data from database SynLethDB. The details of input exmple data are introduced as follows:
* `/data/demo/test_arr_0.txt`, `/data/demo/test_arr_1.txt`, `/data/demo/test_arr_2.txt`, `/data/demo/test_arr_3.txt`, `/data/demo/test_arr_4.txt` are input example data for model testing. With each of them as inputs, `main.py` would automatically generate corresponding training example data.
* `/data/demo/global_interaction_matrix.rar` contains global interaction/adjacency matrices, which were obtained from training example data using functions `random_walk_with_restart` and `extract_global_neighs` in `inits.py`. 

# Run steps
* Download the GitHub repository locally. 
* Decompress `/data/demo/global_interaction_matrix.rar` to obtain global interaction matrices.
* Change the data directories in `main.py` and `inits.py`.
* Run `main.py` to train the model and then obtain the average values of AUC and AUPR in 5-fold CV, as well as ROC and PR curves.

Note: Here we take SynLethDB dataset as an example. You can also update the code with the dataset and the network that you would like to use. 

