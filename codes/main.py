import os
import numpy as np
import tensorflow as tf
from inits import div_list
from train import train
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def main():
   # Set random seed
   seed = 456  
   np.random.seed(seed)
   tf.compat.v1.set_random_seed(seed)
   data_path = "data/demo/"
   labels = np.loadtxt("data/adj.txt")  
   reorder = np.arange(labels.shape[0])
   np.random.shuffle(reorder)
   cv_num = 5
   
   auc_vec, aupr_vec = [], []
   #order = div_list(reorder.tolist(),cv_num)
   for cv in range(cv_num):
      print("cross_validation:", '%01d' % (cv))
      #test_arr = order[i]
      path_arr = os.path.normpath(data_path + 'test_arr_' + str(cv) + '.txt')
      test_arr = np.loadtxt(path_arr)
      test_arr = test_arr.astype(int).tolist()
      arr = list(set(reorder).difference(set(test_arr)))
      np.random.shuffle(arr)
      train_arr = arr
      test_labels, scores = train(train_arr, test_arr, cv) 
      
      prec, rec, thr = precision_recall_curve(test_labels, scores)
      aupr_val = auc(rec, prec)
      aupr_vec.append(aupr_val)
      fpr, tpr, thr = roc_curve(test_labels,scores)
      auc_val = auc(fpr, tpr)
      auc_vec.append(auc_val)
        
      print ("auc:%.6f, aupr:%.6f" % (auc_val, aupr_val))
        
   aupr_mean=np.mean(aupr_vec)
   aupr_std=np.std(aupr_vec) 
   auc_mean=np.mean(auc_vec)
   auc_std=np.std(auc_vec)

    
   print ("auc_ave:%.6f, auc_std: %.6f, aupr_ave:%.6f, aupr_std:%.6f" % (auc_mean, auc_std, aupr_mean, aupr_std))
   plt.figure
   plt.plot(fpr,tpr)
   plt.show()
   plt.figure
   plt.plot(rec,prec)
   plt.show()
   #train_time = time.time() - h
      
if __name__ == "__main__":
    main()      
        
     
    
       
