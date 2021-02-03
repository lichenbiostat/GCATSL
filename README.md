# GCATSL
**Graph Contextualized Attention Network for Predicting Synthetic Lethality in Human Cancers.** 

# Data description
* adj: known SL interaction pairs.
* interaction: adjacent matrix for SL pairs.
* feature_ppi_sparse: features obtained based on PPI network.
* feature_Human_GOsim_BP: features obtained based on GO terms (i.e., BP).
* feature_Human_GOsim_CC: features obtained based on GO terms (i.e., CC).
* demo: a file including training data for 5-fold CV.

# Run steps
Run main.py to train the model and obtain the predicted scores for SL interactions.

# Requirements
* GCATSL is implemented to work under Python 3.7.
* Tensorflow
* numpy
* scipy
* sklearn
