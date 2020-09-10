import time
import numpy as np
import tensorflow as tf

from models import GAT
from inits import test_negative_sample
from inits import load_data
from inits import generate_mask
from inits import sparse_to_tuple
from metrics import masked_accuracy
from metrics import ROC

def train(train_arr, test_arr, cv):
    batch_size = 1
    nb_epochs = 600  
    lr = 0.005  
    l2_coef = 0.0005  
    weight_decay = 1e-4  
    hid_units = [8] 
    n_heads = [2, 1] 
    residual = False
    nonlinearity = tf.nn.elu
    model = GAT

    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))

    interaction_local_list, features_list, y_train, y_test, train_mask, test_mask, labels, interaction_global_list = load_data(train_arr, test_arr, cv)
    nb_nodes = features_list[0].shape[0]  
    ft_size = features_list[0].shape[1]  
    
    features_list = [feature[np.newaxis] for feature in features_list]  
    biases_local_list = [sparse_to_tuple(interaction) for interaction in interaction_local_list]  
    biases_global_list = [sparse_to_tuple(interaction) for interaction in interaction_global_list]
    n = 6375
    entry_size = n * n
    with tf.Graph().as_default():
        with tf.name_scope('input'):
              feature_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(features_list))]
              bias_in_local_list = [tf.compat.v1.sparse_placeholder(tf.float32, name='ftr_in_{}'.format(i)) for i in range(len(biases_local_list))]
              bias_in_global_list = [tf.compat.v1.sparse_placeholder(tf.float32, name='ftr_in_{}'.format(i)) for i in range(len(biases_global_list))]
              lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
              msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
              neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size,batch_size))
              attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
              ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
              is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=())
        
        final_embedding = model.encoder(feature_in_list, nb_nodes, is_train,   
                                attn_drop, ffd_drop,
                                bias_mat_local_list = bias_in_local_list,
                                bias_mat_global_list = bias_in_global_list,
                                hid_units=hid_units, n_heads = n_heads,
                                residual=residual, activation=nonlinearity)
        
        pro_matrix = model.decoder_revised(final_embedding)    
        
        loss = model.loss_overall(pro_matrix, lbl_in, msk_in, neg_msk, weight_decay, final_embedding)
        accuracy = masked_accuracy(pro_matrix, lbl_in, msk_in, neg_msk)
        
        train_op = model.training(loss, lr, l2_coef)

        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        neg_mask = generate_mask(labels, len(train_arr))
        
        #start to train
        with tf.compat.v1.Session() as sess:
          sess.run(init_op)
          train_loss_avg = 0
          train_acc_avg = 0

          for epoch in range(nb_epochs):
              t = time.time()
              
              ##########    train     ##############
              tr_step = 0
              tr_size = features_list[0].shape[0] 
              while tr_step * batch_size < tr_size:  
                      fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                             for i, d in zip(feature_in_list, features_list)}       
                      fd2 = {bias_in_local_list[i]: biases_local_list[i] for i in range(len(biases_local_list))}   
                      fd3 = {bias_in_global_list[i]: biases_global_list[i] for i in range(len(biases_global_list))}   
                      fd4 = {lbl_in: y_train,   
                             msk_in: train_mask,       
                             neg_msk: neg_mask,
                             is_train: True,
                             attn_drop: 0.7,
                             ffd_drop: 0.7}
                      fd = fd1
                      fd.update(fd2)
                      fd.update(fd3)
                      fd.update(fd4)                  
                      _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy], feed_dict=fd)
                     
                      train_loss_avg += loss_value_tr
                      train_acc_avg += acc_tr
                      tr_step += 1  
                     
              print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr,acc_tr, time.time()-t))
          
          print("Finish traing.")
          
          ###########     test      ############
          ts_size = features_list[0].shape[0]
          ts_step = 0
          ts_loss = 0.0
          ts_acc = 0.0
          print("Start to test")
          while ts_step * batch_size < ts_size:
              fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                     for i, d in zip(feature_in_list, features_list)}       
              fd2 = {bias_in_local_list[i]: biases_local_list[i] for i in range(len(biases_local_list))}   
              fd3 = {bias_in_global_list[i]: biases_global_list[i] for i in range(len(biases_global_list))}   
              fd4 = {lbl_in: y_test,   
                     msk_in: test_mask,       
                     neg_msk: neg_mask,
                     is_train: False,
                     attn_drop: 0.0,
                     ffd_drop: 0.0}
              fd = fd1
              fd.update(fd2)
              fd.update(fd3)
              fd.update(fd4) 
              score_matrix, loss_value_ts, acc_ts = sess.run([pro_matrix, loss, accuracy], feed_dict=fd)
              ts_loss += loss_value_ts
              ts_acc += acc_ts
              ts_step += 1
          print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
          
          score_matrix = score_matrix.reshape((n,n))
          test_negative_samples = test_negative_sample(labels,len(test_arr),neg_mask.reshape((n,n)))
          test_labels, test_scores = ROC(score_matrix,labels, test_arr,test_negative_samples)  
          return test_labels, test_scores
          sess.close()
