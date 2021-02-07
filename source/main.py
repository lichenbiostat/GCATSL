import os
import numpy as np
import tensorflow as tf
from inits import logging
from train import train
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from dataprocessor import generate_test_data

def main(args):
   # Set random seed
   seed = 456  
   np.random.seed(seed)
   tf.compat.v1.set_random_seed(seed)
   data_path = args.input_dir
   output_path = args.output_dir
   path_adj = os.path.normpath(data_path + 'adj.txt')
   labels = np.loadtxt(path_adj)
   reorder = np.arange(labels.shape[0])
   np.random.shuffle(reorder)
   cv_num = args.n_fold
   
   auc_vec, aupr_vec = [], []
   logging("-----Results-----", args)
   for cv in range(cv_num):
      print("cross_validation:", '%01d' % (cv))
      path_arr = os.path.normpath(data_path + 'test_arr_' + str(cv) + '.txt')
      
      if not os.path.exists(path_arr):
          print("Test data do not exist. Please generate test data!")
          generate_test_data(args)
          
      test_arr = np.loadtxt(path_arr)
      test_arr = test_arr.astype(int).tolist()
      arr = list(set(reorder).difference(set(test_arr)))
      np.random.shuffle(arr)
      train_arr = arr
      test_labels, scores, test_samples = train(train_arr, test_arr, cv, args, labels) 
      
      prec, rec, thr = precision_recall_curve(test_labels, scores)
      aupr_val = auc(rec, prec)
      aupr_vec.append(aupr_val)
      fpr, tpr, thr = roc_curve(test_labels,scores)
      auc_val = auc(fpr, tpr)
      auc_vec.append(auc_val)
        
      print ("auc:%.6f, aupr:%.6f" % (auc_val, aupr_val))
      logging("The %d-th fold CV: AUC = %.6f,  AUPR = %.6f" % 
                 (cv+1, auc_val, aupr_val), args)
 
      test_result = np.hstack((np.array(test_samples),np.array(scores).reshape([len(scores),-1])))
      test_result = np.hstack((np.array(test_result),np.array(test_labels)))
      
      #path_label = os.path.normpath(output_path + 'test_label_' + str(cv) + '.txt')
      #np.savetxt(path_label, np.array(test_labels))
      
      #path_score = os.path.normpath(output_path + 'test_score_' + str(cv) + '.txt')
      #np.savetxt(path_score, np.array(scores))
       
      path_test_result = os.path.normpath(output_path + 'test_result_' + str(cv) + '.txt')
      np.savetxt(path_test_result, test_result)
      
   aupr_mean=np.mean(aupr_vec)
   aupr_std=np.std(aupr_vec) 
   auc_mean=np.mean(auc_vec)
   auc_std=np.std(auc_vec)

   logging("AUC mean = %.6f, AUC std = %.6f, AUPR mean = %.6f, AUPR std = %.6f" % (auc_mean, auc_std, aupr_mean, aupr_std), args) 
   print ("auc_ave:%.6f, auc_std: %.6f, aupr_ave:%.6f, aupr_std:%.6f" % (auc_mean, auc_std, aupr_mean, aupr_std))
   #plt.figure
   #plt.plot(fpr,tpr)
   #plt.show()
   #plt.figure
   #plt.plot(rec,prec)
   #plt.show()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train and test deep learning model for prediction of SL interaction\n
    Deep learning model will be built by Keras with tensorflow.\n
    You can set almost hyper-parameters as you want, See below parameter description\n
    adj and feature data must be written as txt file format.\n
   
    requirement\n
    ============================\n
    Python >=3.7
    tensorflow >= 1.13.1\n
    numpy\n
    scipy\n
    sklearn\n
    ============================\n
    \n
    contact : yahuilong@hnu.edu.cn\n
    """)
    # train_params
    parser.add_argument("--n_epoch", '-e', help="The number of epochs for training", default=200, type=int)
    parser.add_argument("--n_head", "-n", help="The number of heads", default=8, type=int)
    parser.add_argument("--n_fold", "-F", help="The number of cross validation", default=5, type=int)
    parser.add_argument("--n_node", "-N", help="The number of nodes in the network", default=6375, type=int)
    parser.add_argument("--n_feature", "-f", help="The number of features", default=3, type=int)
    parser.add_argument("--learning_rate", '-r', help="Learning late for training", default=0.005, type=float)
    parser.add_argument("--weight_decay", '-w', help="Weight decay for controling the impact of latent factor", default=1e-4, type=float)
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.7, type=float)
    
    parser.add_argument("--input_dir", help="The directory of input data", type=str)
    parser.add_argument("--output_dir", help="The directory of output results", type=str)
    parser.add_argument("--log_dir", help="The directory of logs", type=str)

    args = parser.parse_args()  #解析参数
    print("Starting to train and test model.")
    main(args)       
        
     
    
       
