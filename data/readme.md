To reproduce the results of GCATSL reported in the paper, here we provided a set of input example data. 

# Data description
* `/data/toy_examples/adj.txt`: known SL interaction pairs obtained from dataset [SynLethDB](http://synlethdb.sist.shanghaitech.edu.cn/downloadPage.php).
* `/data/toy_examples/interaction.txt`: adjacent matrix for SL pairs, constructed from `/data/toy_examples/adj.txt`.
* `/data/toy_examples/feature_1.txt`: features obtained based on PPI network. The PPI network can be constructed from dataset [BioGrid](https://thebiogrid.org/)
* `/data/toy_examples/feature_2.txt`: features obtained based on GO terms (i.e., BP). The sub-ontologies and annotation ﬁles can be downloaded from [here](http://geneontology.org/).
* `/data/toy_examples/feature_3.txt`: features obtained based on GO terms (i.e., CC). The sub-ontologies and annotation ﬁles can be downloaded from [here](http://geneontology.org/).


* `/data/test_arr_0.txt`, `/data/demo/test_arr_1.txt`, `/data/demo/test_arr_2.txt`, `/data/demo/test_arr_3.txt`, `/data/demo/test_arr_4.txt` are input example data for model testing. With each of them as inputs, `main.py` would automatically generate corresponding training example data.
* `/data/demo/global_interaction_matrix.rar` contains global interaction/adjacency matrices, which were obtained from training example data using functions `random_walk_with_restart` and `extract_global_neighs` in `inits.py`.
