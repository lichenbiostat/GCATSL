import tensorflow as tf
from utils import layers
from models.base_gattn import BaseGAttN
from inits import glorot
from metrics import masked_accuracy

class GAT(BaseGAttN): 
    def encoder(inputs_list, nb_nodes, training, attn_drop, ffd_drop,   
            bias_mat_local_list, bias_mat_global_list, hid_units, n_heads, mp_att_size=16, activation=tf.nn.elu, residual=False):
        embed_list = []
        for inputs, bias_mat_local, bias_mat_global in zip(inputs_list, bias_mat_local_list, bias_mat_global_list):
            attns = []
            for _ in range(n_heads):     
                attn_temp = layers.sp_attn_head(inputs, adj_mat_local=bias_mat_local, adj_mat_global=bias_mat_global,   
                        out_sz=hid_units[0], activation=activation, 
                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
                attns.append(attn_temp)
            h_1 = tf.concat(attns, axis=-1)   
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))  
        multi_embed = tf.concat(embed_list,axis=1)
        final_embed, alpha = layers.SimpleAttLayer(multi_embed, mp_att_size,      
                                                     time_major=False,
                                                     return_alphas=True)
        return final_embed   
         
    def decoder(embed):
        embed_size = embed.shape[1].value
        with tf.compat.v1.variable_scope("deco"):
             weight3 = glorot([embed_size,embed_size])
        U=embed
        V=embed
        logits=tf.matmul(tf.matmul(U,weight3),tf.transpose(V))
        logits=tf.reshape(logits,[-1,1])
        return tf.nn.sigmoid(logits)
        
    def decoder_revised(embed):
        num_nodes = embed.shape[0].value
        embed_size = embed.shape[1].value
        with tf.compat.v1.variable_scope("deco_revised"):
             weight1 = glorot([embed_size,embed_size])
             weight2 = glorot([embed_size,embed_size])
             bias = glorot([num_nodes,embed_size])
        embedding = tf.add(tf.matmul(embed,weight1), bias)
        logits=tf.matmul(tf.matmul(embedding,weight2),tf.transpose(embedding))
        logits=tf.reshape(logits,[-1,1])
        return tf.nn.sigmoid(logits)    
    
    def loss_overall(scores, lbl_in, msk_in, neg_msk, weight_decay, emb):
        loss_basic = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="deco_revised")
        loss_basic +=  weight_decay * tf.nn.l2_loss(para_decode[0])  
        loss_basic +=  weight_decay * tf.nn.l2_loss(para_decode[1])
        loss_basic +=  weight_decay * tf.nn.l2_loss(para_decode[2])
        return loss_basic
    
    
    
    
    
    
    
    
    
    
    
     