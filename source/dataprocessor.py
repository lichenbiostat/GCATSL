import numpy as np
import tensorflow as tf
import os
import scipy.sparse as sp

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def random_walk_with_restart(interaction):
    p = 0.9  #0.9
    iter_max = 1000
    origi_matrix = np.identity(interaction.shape[0])
    sum_col = interaction.sum(axis=0)  
    sum_col[sum_col == 0.] = 2
    interaction = np.divide(interaction,sum_col)
    pre_t = origi_matrix
    
    for i in range(iter_max):
        print("i:",i)
        t = (1-p) * (np.dot(interaction, pre_t)) + p * origi_matrix
        pre_t = t
    return t

def extract_global_neighbors(interaction, walk_matrix):
    interaction = interaction.astype(int)
    interaction_mask = np.zeros_like(interaction)
    neigh_index = np.argsort(-walk_matrix, axis=0)    
    
    for j in range(interaction.shape[1]):      
        for i in range(np.sum(interaction[j,:])):
            interaction_mask[neigh_index[i,j],j]=1
    return interaction_mask.T

def generate_test_data(args):
    # Set random seed
    seed = 456  #123
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    data_path = args.input_dir 
    path_adj = os.path.normpath(data_path + 'adj.txt')
    labels = np.loadtxt(path_adj)  
    
    reorder = np.arange(labels.shape[0])
    np.random.shuffle(reorder)

    cv_num = args.n_folds
    order = div_list(reorder.tolist(),cv_num)
    for i in range(cv_num):
        print("cross_validation:", '%01d' % (i))
        test_arr = order[i]
        path_arr = os.path.normpath(data_path + 'test_arr_' + str(i) + '.txt')
        np.savetxt(path_arr, np.array(test_arr))
    print("Finish the generation of test data")    
            
def generate_global_interaction_matrix(args, train_arr, cv):
    data_path = args.input_dir
    path_adj = os.path.normpath(data_path + 'adj.txt')
    labels = np.loadtxt(path_adj)
    num_nodes = args.n_nodes
    np.random.shuffle(train_arr)
       
    M = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(num_nodes, num_nodes)).toarray()
    M = M + M.T
        
    print("Start to implement random walk.")
    walk_matrix = random_walk_with_restart(M)
    print("Finish random walk.")
    interaction_global = extract_global_neighbors(M, walk_matrix)
    path_global = os.path.normpath(data_path + 'interaction_global_' + str(cv) + '.txt')
    np.savetxt(path_global, interaction_global)
