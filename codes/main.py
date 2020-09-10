import os
import numpy as np
import tensorflow as tf
from inits import div_list
from train import train


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
      
if __name__ == "__main__":
    main()       
        
     
    
       
