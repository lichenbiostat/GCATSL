import tensorflow as tf
import numpy as np


def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)  
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32) 
    error *= mask  
    return tf.sqrt(tf.reduce_mean(error))

def ROC(score_matrix, labels, test_arr, label_neg):
    test_scores=[]
    for i in range(len(test_arr)):
        l = test_arr[i]
        test_scores.append(score_matrix[int(labels[l,0]-1),int(labels[l,1]-1)])
    for i in range(label_neg.shape[0]):
        test_scores.append(score_matrix[int(label_neg[i,0]),int(label_neg[i,1])])
        
    test_labels_pos = np.ones((len(test_arr),1))
    test_labels_neg = np.zeros((label_neg.shape[0],1))
    
    test_labels = np.vstack((test_labels_pos,test_labels_neg))
    test_labels = np.array(test_labels,dtype=np.bool).reshape([-1,1])
    return test_labels, test_scores