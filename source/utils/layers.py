import tensorflow as tf
from inits import glorot

conv1d = tf.layers.conv1d

def sp_attn_head(seq, out_sz, adj_mat_local, adj_mat_global, activation, in_drop=0.0, coef_drop=0.0, residual=False):      
  with tf.name_scope('my_attn'):
    if in_drop != 0.0:
       seq = tf.nn.dropout(seq, 1.0 - in_drop)
    seq_fts = seq
    
    latent_factor_size = 8  
    nb_nodes = seq_fts.shape[1].value
    
    w_1 = glorot([seq_fts.shape[2].value,latent_factor_size])
    w_2 = glorot([3*seq_fts.shape[2].value,latent_factor_size])
    
    f_1 = tf.layers.conv1d(seq_fts, 1, 1) 
    f_2 = tf.layers.conv1d(seq_fts, 1, 1) 
    
    #local neighbours
    logits = tf.add(f_1[0], tf.transpose(f_2[0]))
    logits_first = adj_mat_local * logits
    lrelu = tf.SparseTensor(indices=logits_first.indices,
                                values=tf.nn.leaky_relu(logits_first.values),
                                dense_shape=logits_first.dense_shape)
    coefs = tf.sparse_softmax(lrelu)
    
    coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])   
    seq_fts = tf.squeeze(seq_fts) 
    neigh_embs = tf.sparse.sparse_dense_matmul(coefs, seq_fts)  
    
    #non-local neighbours
    logits_global = adj_mat_global * logits
    lrelu_global = tf.SparseTensor(indices=logits_global.indices,
                                values=tf.nn.leaky_relu(logits_global.values),
                                dense_shape=logits_global.dense_shape)
    coefs_global = tf.sparse_softmax(lrelu_global)
    
    coefs_global = tf.sparse_reshape(coefs_global, [nb_nodes, nb_nodes])
    neigh_embs_global = tf.sparse.sparse_dense_matmul(coefs_global, seq_fts)  
    
    neigh_embs_sum_1 = tf.matmul(tf.add(tf.add(seq_fts,neigh_embs),neigh_embs_global),w_1)
    neigh_embs_sum_2 = tf.matmul(tf.concat([tf.concat([seq_fts,neigh_embs],axis=-1),neigh_embs_global],axis=-1),w_2)
    
    final_embs = activation(neigh_embs_sum_1) + activation(neigh_embs_sum_2)   
    
    return final_embs

def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):   
    hidden_size = inputs.shape[2].value   

    # Trainable parameters
    w_omega = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))    
    b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))  

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)  
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  
    alphas = tf.nn.softmax(vu, name='alphas')            
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)   
    
    if not return_alphas:
        return output
    else:
        return output, alphas
  

     
