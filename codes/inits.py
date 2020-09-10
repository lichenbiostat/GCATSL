import os
import numpy as np
import scipy.sparse as sp
import random
import tensorflow as tf
from sklearn.decomposition import PCA         

def load_data(train_arr, test_arr, cv):
    features_list = []
    labels = np.loadtxt("data/adj.txt")
    n = 6375
    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(n,n)).toarray() 
    logits_test = logits_test.reshape([-1,1])  

    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(n,n)).toarray()
    logits_train = logits_train + logits_train.T
    logits_train = logits_train.reshape([-1,1])
      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])  
       
    #feature pre-processing
    features1 = np.loadtxt("data/feature_ppi_sparse.txt")
    features2 = np.loadtxt("data/feature_Human_GOsim_BP.txt")    
    features3 = np.loadtxt("data/feature_Human_GOsim_CC.txt")   
    features1 = normalize_features(features1)
    features2 = normalize_features(features2)
    features3 = normalize_features(features3)
    features_list.append(features1)
    features_list.append(features2)
    features_list.append(features3)  
    
    #interaction for lobal neighbors
    interaction_local = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(n,n)).toarray()
    interaction_local = interaction_local + interaction_local.T
    interaction_local = interaction_local + np.eye(interaction_local.shape[0])
    interaction_local = sp.csr_matrix(interaction_local)
    
    #interaction for global neighbors
    #walk_matrix = random_walk_with_restart(interaction_local)
    #interaction_mask = extract_global_neighs(interaction_local, walk_matrix)
    data_path = "data/demo/"
    path_global = os.path.normpath(data_path + 'interaction_global_' + str(cv) + '.txt')
    interaction_global = np.loadtxt(path_global)
    interaction_global = interaction_global + np.eye(interaction_global.shape[0])
    interaction_global = sp.csr_matrix(interaction_global)
    
    interaction_local_list = [interaction_local, interaction_local, interaction_local]
    interaction_global_list = [interaction_global, interaction_global, interaction_global]
    
    return interaction_local_list, features_list, logits_train, logits_test, train_mask, test_mask, labels, interaction_global_list 

def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()
    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)
    return feat_norm

def random_walk_with_restart(interaction):
    p = 0.9  
    iter_max = 1000
    origi_matrix = np.identity(interaction.shape[0])
    sum_col = interaction.sum(axis=0)  
    sum_col[sum_col == 0.] = 2
    interaction = np.divide(interaction,sum_col)
    pre_t = origi_matrix
    
    for i in range(iter_max):
        t = (1-p) * (np.dot(interaction, pre_t)) + p * origi_matrix
        pre_t = t
    return t 

def extract_global_neighs(interaction, walk_matrix):
    interaction = interaction.astype(int)
    interaction_mask = np.zeros_like(interaction)
    neigh_index = np.argsort(-walk_matrix, axis=0)    
    
    for j in range(interaction.shape[1]):      
        for i in range(np.sum(interaction[j,:])):
            interaction_mask[neigh_index[i,j],j]=1
    return interaction_mask.T       

def generate_mask(labels,N):  
    num = 0
    n = 6375
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(n,n)).toarray()
    A = A + A.T
    mask = np.zeros(A.shape)
    while(num<2*N):
        a = random.randint(0,n-1) 
        b = random.randint(0,n-1) 
        if  A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            num += 1           
    mask = np.reshape(mask,[-1,1])  
    return mask

def test_negative_sample(labels,N,negative_mask):  
    num = 0
    (nd,nm)=negative_mask.shape
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(nd,nm)).toarray()  
    A = A + A.T
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,nd-1) 
        b = random.randint(0,nm-1) 
        if A[a,b] != 1 and mask[a,b] != 1 and negative_mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1    
    return test_neg

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)        

def normalization(data):   
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def row_normalization(feature):
    feature_normalized = np.zeros_like(feature)
    for i in range(feature.shape[0]):
        feature_normalized[i,:] = normalization(feature[i,:])
    return feature_normalized

def feature_reduction(feature, dim=64):
    pca = PCA(n_components=dim)
    feature_reduced = pca.fit_transform(feature)
    return feature_reduced

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
