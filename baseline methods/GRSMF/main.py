#!/usr/bin/python
# -*- coding:utf-8 -*-
import time
import numpy as np
#from lmf import LMF
from grsmf import GRSMF
from grsmf_functions import load_ppi_data, load_ppi_data_long
from grsmf_functions import generate_test_neg_data
from grsmf_functions import evaluation_bal
from grsmf_functions import evaluation_two
from grsmf_functions import mean_confidence_interval
from collections import defaultdict
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
flag = 1

inter_pairs, inter_scores, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat, pro_id_mapping = load_ppi_data_long(flag)
print(len(inter_pairs))
print("Data downloaded")

if flag==0:
   num_nodes = 6375
elif flag==1:
   num_nodes = 10218
   
x, y = np.triu_indices(num_nodes, k=1)
c_set = set(zip(x, y)) - set(zip(inter_pairs[:, 0], inter_pairs[:, 1])) - set(zip(inter_pairs[:, 1], inter_pairs[:, 0]))
noninter_pairs = np.array(list(c_set))

reorder = np.arange(len(inter_pairs))
np.random.shuffle(reorder)
inter_pairs = inter_pairs[reorder]

reorder_neg = np.arange(len(noninter_pairs))
np.random.shuffle(reorder_neg)
noninter_pairs = noninter_pairs[reorder_neg[0:len(inter_pairs)]]

if flag==0:
   go_sim_mat = go_sim_mat.toarray()
   go_sim_mat = go_sim_mat+go_sim_mat.T

prng = np.random.RandomState(123)

t1 = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=0)
num = len(pro_id_mapping.keys())

pos_edge_kf = kf.split(inter_pairs)
neg_edge_kf = kf.split(noninter_pairs)
# num = 300
k = 0
i,j = -7, -5
for t in range(1):       
    auc_vec, aupr_vec = [], []
    t = time.time()
    for train, test in pos_edge_kf:
        print("cross_validation:", '%01d' % (k))
        neg_train, neg_test = next(neg_edge_kf)
        
        model = GRSMF(lambda_d=2 ** (i), beta=2 ** (j), max_iter=10)
        cmd = str(model)
        x, y = inter_pairs[train, 0], inter_pairs[train, 1]
        x_test, y_test = inter_pairs[test, 0], inter_pairs[test, 1]
        IntMat = np.zeros((num, num))
        W = np.zeros((num, num))
        IntMat[x, y] = 1
        IntMat[y, x] = 1  
        
        x_neg, y_neg = noninter_pairs[neg_train, 0], noninter_pairs[neg_train, 1]
        
        W[x, y] = 1
        W[y, x] = 1
        W[x_neg, y_neg] = 1
        W[y_neg, x_neg] = 1
        
        
        model.fix_model(W, IntMat, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat)
            
        test_edges_pos = inter_pairs[test, :]  
        test_edges_neg = noninter_pairs[neg_test[0:len(test)], :]  
        print(type(test_edges_pos))
        auc_val, aupr_val = evaluation_bal(model.predictR, test_edges_pos, test_edges_neg)
            
        #auc_val, aupr_val = evaluation_two(inter_pairs[train, :], inter_pairs[test, :], model.predictR)
        auc_vec.append(auc_val)
        aupr_vec.append(aupr_val)
        k = k+1
        print ("auc:%.6f, aupr:%.6f, time: %f" % (auc_val, aupr_val, time.time()-t))
    aupr_mean=np.mean(aupr_vec)
    aupr_std=np.std(aupr_vec) 
    auc_mean=np.mean(auc_vec)
    auc_std=np.std(auc_vec)
    print ("Average metrics over pairs: auc_mean:%s, auc_std:%s, aupr_mean:%s, aupr_std:%s,Time:%.6f\n" % (auc_mean, auc_std, aupr_mean, aupr_std, time.time() - t1))

