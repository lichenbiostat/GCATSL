import pdb
import time
import numpy as np
from lmf import LMF
from functions import mean_confidence_interval
from functions import load_ppi_data, load_ppi_data_long
from collections import defaultdict
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from functions import evalution_bal


def pair2dict(pairs, scores):
    pro_dict = defaultdict(lambda: defaultdict(float))
    for i, inx in enumerate(pairs): 
        pro_dict[inx[0]][inx[1]] = scores[i]   
        pro_dict[inx[1]][inx[0]] = scores[i]
    return pro_dict

def evaluation(train_dict, test_dict, U):
    num = U.shape[0]
    set_all = set(range(num))
    auc_val, aupr_val = 0.0, 0.0
    for p in test_dict:
        val = np.exp(np.dot(U[p, :], U.T))
        val = val/(1.0+val)
        inx = np.array(list(set_all - set(train_dict[p].keys()) - set([p])))
        label = np.zeros(num)
        ii = np.array(test_dict[p].keys())
        ii = ii.astype(int)
        label[ii] = 1
        try:
            auc_val += roc_auc_score(label[inx], val[inx])
        except:
            pdb.set_trace()
        prec, rec, thr = precision_recall_curve(label[inx], val[inx])
        aupr_val += auc(rec, prec)
    return auc_val/len(test_dict), aupr_val/len(test_dict)

def evaluation_two(train_pairs, test_pairs, U):
    num = U.shape[0]
    x, y = np.triu_indices(num, k=1)
    c_set = set(zip(x, y)) - set(zip(train_pairs[:, 0], train_pairs[:, 1])) - set(zip(train_pairs[:, 1], train_pairs[:, 0]))
    inx = np.array(list(c_set))
    Y = np.zeros((num, num))
    Y[test_pairs[:, 0], test_pairs[:, 1]] = 1
    Y[test_pairs[:, 1], test_pairs[:, 0]] = 1
    labels = Y[inx[:, 0], inx[:, 1]]
    Y = np.dot(U, U.T)
    val = Y[inx[:, 0], inx[:, 1]]
    val = np.exp(val)
    val = val/(1+val)
    # pdb.set_trace()
    auc_val = roc_auc_score(labels, val)
    prec, rec, thr = precision_recall_curve(labels, val)
    aupr_val = auc(rec, prec)
    return auc_val, aupr_val

def evaluation_two_long(train_pairs, test_pairs, U):
    num = U.shape[0]
    x, y = np.triu_indices(num, k=1)
    zero_matrix = np.zeros([6375,6375])
    x1, y1 = np.where(zero_matrix==0)
    c_set = set(zip(x1, y1)) - set(zip(train_pairs[:, 0], train_pairs[:, 1])) - set(zip(train_pairs[:, 1], train_pairs[:, 0])) - set(zip(test_pairs[:, 0], test_pairs[:, 1])) - set(zip(test_pairs[:, 1], test_pairs[:, 0]))
    inx = np.array(list(c_set))  #all negative samples set
    num_neg = inx.shape[0]
    
    reorder = np.arange(num_neg)
    np.random.shuffle(reorder)
    
    test_neg_set = inx[reorder[0:len(test_pairs)]]
    test_sample_set = np.vstack((test_pairs,test_neg_set))
    
    Y = np.zeros((num, num))
    Y[test_pairs[:, 0], test_pairs[:, 1]] = 1
    Y[test_pairs[:, 1], test_pairs[:, 0]] = 1
    labels = Y[test_sample_set[:, 0], test_sample_set[:, 1]]
    Y = np.dot(U, U.T)
    val = Y[test_sample_set[:, 0], test_sample_set[:, 1]]
    val = np.exp(val)
    val = val/(1+val)
    # pdb.set_trace()
    auc_val = roc_auc_score(labels, val)
    prec, rec, thr = precision_recall_curve(labels, val)
    aupr_val = auc(rec, prec)
    return auc_val, aupr_val

def predict_top_pairs(train_pairs, U, num, pro_id_mapping):
    pass

def wighting_fun():
    pass
    
flag = 1 #0:SL; 1:SL_extension

inter_pairs, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat, pro_id_mapping = load_ppi_data_long(flag)
print(len(inter_pairs))

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
noninter_pairs = noninter_pairs[reorder_neg]

print("Data downloaded!")

if flag==0:
   go_sim_mat = go_sim_mat.toarray()   
   go_sim_mat = go_sim_mat+go_sim_mat.T

#pdb.set_trace()

cv_flag = True
prng = np.random.RandomState(123)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
#kf = KFold(inter_scores.size, n_folds=5, shuffle=True, random_state=prng)
num = len(pro_id_mapping.keys())
pos_edge_kf = kf.split(inter_pairs)
neg_edge_kf = kf.split(noninter_pairs)

# index = prng.permutation(inter_scores.size)
# train_num = int(np.floor(inter_scores.size*0.8))
# train, test = index[:train_num], index[train_num:]
for nn_size in [45]:
        auc_pro, aupr_pro = [], []
        auc_pair, aupr_pair = [], []
        t = time.time()
        for train, test in pos_edge_kf:
            
            neg_train, neg_test = next(neg_edge_kf)
            
            model = LMF(num_factors=50, nn_size=nn_size, theta=2.0**(-5), reg=10.0**(-2), alpha=1*10.0**(0), beta=1*10.0**(0), beta1=1*10.0**(0), beta2=1*10.0**(0), max_iter=100)
            print(str(model))
            x, y = inter_pairs[train, 0], inter_pairs[train, 1]
            IntMat = np.zeros((num, num))
            W = np.ones((num, num))
            IntMat[x, y] = 1
            IntMat[y, x] = 1
            W[x, y] = 50
            W[y, x] = W[x, y]
            
            x_neg, y_neg = noninter_pairs[neg_train[0:len(train)], 0], noninter_pairs[neg_train[0:len(train)], 1]
            mask = np.zeros_like(IntMat)
            mask[x, y] = 1
            mask[y, x] = 1
            mask[x_neg, y_neg] = 1
            mask[y_neg, x_neg] = 1
            
            t1 = time.time()
            #model.fix(IntMat, W, None, tp_sim_mat)
            model.fix(IntMat, W, mask, go_sim_mat, go_sim_cc_mat, ppi_sparse_mat, co_pathway_mat)
            #print("nn_size: %s" % nn_size)
        
            #auc_val, aupr_val = evaluation_two_long(inter_pairs[train, :], inter_pairs[test, :], model.U)
            auc_val, aupr_val = evalution_bal(np.dot(model.U, model.U.T), inter_pairs[test, :], noninter_pairs[neg_test[0:len(test)], :])
            auc_pair.append(auc_val)
            aupr_pair.append(aupr_val)
            print("metrics over protein pairs: auc %f, aupr %f, time: %f\n" % (auc_val, aupr_val, time.time()-t))
        
        m1, sdv1 = mean_confidence_interval(auc_pair)
        m2, sdv2 = mean_confidence_interval(aupr_pair)
        print("Average metrics over pairs: auc_mean:%s, auc_sdv:%s, aupr_mean:%s, aupr_sdv:%s\n" %(m1, sdv1, m2, sdv2))         
