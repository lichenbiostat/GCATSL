# GCATSL: Graph Contextualized Attention Network for Predicting Synthetic Lethality in Human Cancers
GCATSL is a deep learning model that can be used for SL prediction. GCATSL first learns representations for nodes based on different feature graphs together with a known SL graph, and then uses the learned node representations to reconstruct SL interaction matrix for SL prediction. 

# Installation
GCATSL is implemented in with Tensorflow library. For detail instruction of installing Tensorflow, see the guidence on official website of Tensorflow.

# Requirements
* Python 3.7.
* Tensorflow 1.13.1
* numpy 1.16.2
* scipy 1.4.1
* sklearn

# Usage
First, you need to clone the repository or download source code and data files. 
    $ git clone https://github.com/longyahui/GCATSL.git


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

