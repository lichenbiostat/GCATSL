To reproduce the results of GCATSL reported in the paper, here we provide a set of input example data. 

# Data description
* `./toy_examples/adj.txt`: known SL interaction pairs obtained from dataset [SynLethDB](http://synlethdb.sist.shanghaitech.edu.cn/downloadPage.php).
* `./toy_examples/interaction.txt`: adjacent matrix for SL pairs, constructed from `./toy_examples/adj.txt`.
* `./toy_examples/feature_1.txt`: features obtained based on PPI network. The PPI network can be constructed from dataset [BioGrid](https://thebiogrid.org/)
* `./toy_examples/feature_2.txt`: features obtained based on GO terms (i.e., BP). The sub-ontologies and annotation ﬁles can be downloaded from [here](http://geneontology.org/).
* `./toy_examples/feature_3.txt`: features obtained based on GO terms (i.e., CC). The sub-ontologies and annotation ﬁles can be downloaded from [here](http://geneontology.org/).

We randomly divide all known SL pairs into five groups as test samples. 
* `./toy_examples/test_arr_0.txt`: the first set of test samples.
* `./toy_examples/test_arr_1.txt`: the second set of test samples.
* `./toy_examples/test_arr_2.txt`: the third set of test samples.
* `./toy_examples/test_arr_3.txt`: the fourth set of test samples.
* `./toy_examples/test_arr_4.txt`: the fifth set of test samples.


* `/data/demo/global_interaction_matrix.rar` contains global interaction/adjacency matrices, which were obtained from training example data using functions `random_walk_with_restart` and `extract_global_neighs` in `inits.py`.
