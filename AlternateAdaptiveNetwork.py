#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:14:31 2017

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import os
import numpy as np
import json
from Sampler import SampleTypeWise
import datetime
import cPickle as pkl
from Evaluation import Evaluate_MR
from pathos.threading import ThreadPool as Pool
from copy import deepcopy
import time
from numba import jit
# =============================================================================
#  Training Parameters
# =============================================================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 160 #@myself: need to set this
NUM_TYPES_BATCH = BATCH_SIZE 
RHO = 0.05 #Desired average activation value
BETA = 0.05
ALPHA = 1
GAMMA = 1
MARGIN = 1
BATCH_EVAL = 32
NUM_EPOCHS = 1000
Nsamples_Transe_Neg = 4
Nsamples_Transe_Pos = 160
# =============================================================================
#  Network Parameters-1 #First let us solve only for Type loss
# =============================================================================
NUM_HIDDEN_1 = 128 # 1st layer num features
NUM_HIDDEN_2 = 256 # 2nd layer num features (the latent dim)
NUM_INPUT = EMBEDDING_SIZE = 64 #@myself: need to set this
VOCABULARY_SIZE = 14951
RELATIONS_SIZE = 1345
LOG_DIR = 'Logs/'+datetime.datetime.now().strftime("%B %d, %Y, %I.%M%p")
DEVICE = '/cpu:0'
# =============================================================================
#  Import data regarding embedding relations and type information
# =============================================================================
Type2Data = pkl.load(open('Type2Data.pkl'))
ent2type = pkl.load(open('ent2type.pkl'))
relations_dic_h = pkl.load(open('relations_dic_h.pkl'))
relations_dic_t = pkl.load(open('relations_dic_t.pkl'))
#for evaluation during training
evalsubset_relations= np.array(pkl.load(open\
               ('evalsubset_relations_train.pkl','r')))
evalsubset_relations = evalsubset_relations\
     [0:len(evalsubset_relations) - \
     len(evalsubset_relations) % BATCH_EVAL]
relations = json.load(open('relations_hrt.json'))
TOT_RELATIONS = len(json.load(open('relations_hrt.json')))
NUM_TYPES = len(Type2Data)

@jit
def generate_labels(batch):
    out = []
    for x in batch:
        out.append(NUM_TYPES*[0])
        for i in ent2type[x]:
            out[-1][i]=1
    return out
# =============================================================================
# tf Graph input 
# =============================================================================
#Define the entity embedding matrix to be uniform in a unit cube
with tf.device(DEVICE):
    ent_embeddings = tf.get_variable(name='W_Ent',shape = [VOCABULARY_SIZE,\
                       EMBEDDING_SIZE],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True))
    
    
                                               
    #Define the relation embedding matrix to be uniform in a unit cube
    rel_embeddings = tf.get_variable(name='W_Rel',shape = [RELATIONS_SIZE,\
                       EMBEDDING_SIZE],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True))



    #this will contain embedding ids in given a batch
    X = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    #this will contain the type labels for each embedding in the given batch
    Y = tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_TYPES])
    
    #these are the placeholders necessary for the TransE loss Part
    #placeholders for positive samples
    pos_h = tf.placeholder(tf.int32, [None])
    pos_r = tf.placeholder(tf.int32, [None])
    pos_t = tf.placeholder(tf.int32, [None])
    #placeholders for negative samples
    neg_h = tf.placeholder(tf.int32, [None,Nsamples_Transe_Neg])
    neg_r = tf.placeholder(tf.int32, [None,Nsamples_Transe_Neg])
    neg_t = tf.placeholder(tf.int32, [None,Nsamples_Transe_Neg])
    
    #these are the placeholders necessary for evaluating the link prediction
    eval_h = tf.placeholder(tf.int32, [None])
    eval_t = tf.placeholder(tf.int32, [None])#will do h+r and generate rank t 
    eval_r = tf.placeholder(tf.int32, [None])
    eval_to_rank = tf.placeholder(tf.int32, [None])    
    

    
    

#look up the vector for each of the source words in the batch for Jtype part
embed = tf.nn.embedding_lookup(ent_embeddings, X)

#look up the vector for each of the source words in the batch for TransE part
pos_h_e = tf.nn.embedding_lookup(ent_embeddings, pos_h)
pos_t_e = tf.nn.embedding_lookup(ent_embeddings, pos_t)
pos_r_e = tf.nn.embedding_lookup(rel_embeddings, pos_r)
neg_h_e = tf.nn.embedding_lookup(ent_embeddings, neg_h)
neg_t_e = tf.nn.embedding_lookup(ent_embeddings, neg_t)
neg_r_e = tf.nn.embedding_lookup(rel_embeddings, neg_r)


eval_h_e = tf.nn.embedding_lookup(ent_embeddings, eval_h)
eval_t_e = tf.nn.embedding_lookup(ent_embeddings, eval_t)
eval_r_e = tf.nn.embedding_lookup(rel_embeddings, eval_r)
eval_to_rank_e = tf.nn.embedding_lookup(ent_embeddings, eval_to_rank)

normalize_entity_op = tf.assign(ent_embeddings,tf.nn.l2_normalize\
                                            (ent_embeddings,dim=1))
                                                    

normalize_rel_op = tf.assign(rel_embeddings,tf.nn.l2_normalize\
                                             (rel_embeddings,dim=1))



with tf.device(DEVICE):
    
    weights = {
            
        'encoder_h1': tf.get_variable(name='W_encoder_h1',shape = \
                      [NUM_INPUT, NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),
    
        'encoder_h2': tf.get_variable(name='W_encoder_h2',shape = \
                      [NUM_HIDDEN_1, NUM_HIDDEN_2],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),      
                                                                   
        'decoder_h1': tf.get_variable(name='W_decoder_h1',shape = \
                      [NUM_HIDDEN_2, NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),      
        'decoder_h2': tf.get_variable(name='W_decoder_h2',shape = \
                      [NUM_HIDDEN_1, NUM_INPUT],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),      
        'classification_h': tf.get_variable(name='W_classification_h',shape = \
                        [NUM_HIDDEN_2, NUM_TYPES],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),
    }
    
    biases = {
        'encoder_b1': tf.get_variable(name='W_encoder_b1',shape = \
                      [NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),
        'encoder_b2': tf.get_variable(name='W_encoder_b2',shape = \
                      [NUM_HIDDEN_2],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),
        'decoder_b1': tf.get_variable(name='W_decoder_b1',shape = \
                       [NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),      
        'decoder_b2': tf.get_variable(name='W_decoder_b2',shape = \
                      [NUM_INPUT],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),      
        'classification_b': tf.get_variable(name='W_classification_b',shape = \
                        [NUM_TYPES],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True)),
    }

# =============================================================================
#  Building the encoder
# =============================================================================
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    
    RhoJEH1 = tf.reduce_mean(tf.abs(layer_1),0)
    RhoJEH2 = tf.reduce_mean(tf.abs(layer_2),0) 
    return layer_2, RhoJEH1, RhoJEH2


# =============================================================================
#  Building the decoder
# =============================================================================
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    RhoJDH1 = tf.reduce_mean(tf.abs(layer_1),0)
    RhoJDH2 = tf.reduce_mean(tf.abs(layer_2),0)     
    return layer_2, RhoJDH1, RhoJDH2

# =============================================================================
# Building the classification layer
# =============================================================================
def classify(x):
    #classification hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['classification_h']),
                                   biases['classification_b'])
    return layer_1

# =============================================================================
# Building the testing layer which outputs a ranked list of entities
# =============================================================================
def predict_rank(x, y, z, batch_size_eval, z_eval_dataset_size, K):
    z_ = tf.add(x,y)
    z_ = tf.stack(z_eval_dataset_size*[z_],axis=1)
    z = tf.stack(batch_size_eval*[z],axis=0) 
    out = -1*tf.norm(tf.add(z,-1*z_),axis=2)
    #now find the top_k members from this set
    values_indices = tf.nn.top_k(out,k=K)
    return values_indices


# =============================================================================
#  Construct model
# =============================================================================
#training part of network
encoder_op, RhoJEH1, RhoJEH2 = encoder(embed)
decoder_op, RhoJDH1, RhoJDH2 = decoder(encoder_op)
classifier_op = classify(encoder_op)


#evaluation part of TransE
_t, indices_t = predict_rank(eval_h_e, eval_r_e, eval_to_rank_e, \
                    BATCH_EVAL, VOCABULARY_SIZE, VOCABULARY_SIZE)
_h, indices_h = predict_rank(eval_t_e, -1*eval_r_e, eval_to_rank_e, \
                    BATCH_EVAL, VOCABULARY_SIZE, VOCABULARY_SIZE)

#evaluation part of Classification
score_classification = tf.nn.sigmoid(classifier_op)
marker_classification = 2*(Y-0.5)

margin_classification = tf.reduce_mean(tf.reduce_sum(tf.multiply(\
                         score_classification,marker_classification),axis=1))

pos_score_classification = tf.divide(tf.reduce_sum(\
                              tf.multiply(score_classification,Y),axis=1),\
                                tf.reduce_sum(Y,axis=1))
pos_score_classification = tf.reduce_mean(tf.boolean_mask(\
                      pos_score_classification, \
                      tf.logical_not(tf.is_nan(pos_score_classification))))
neg_score_classification = tf.divide(tf.reduce_sum(\
                              tf.multiply(score_classification,1-Y),axis=1),\
                                tf.reduce_sum(1-Y,axis=1))
neg_score_classification = tf.reduce_mean(tf.boolean_mask(\
                          neg_score_classification, \
                          tf.logical_not(tf.is_nan(neg_score_classification))))
delta_classification = pos_score_classification - neg_score_classification

# =============================================================================
#  Prediction
# =============================================================================
 #For autoencoder part
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = embed

 #For classification part
logits = classifier_op
labels = Y

# =============================================================================
#  Define loss and optimizer, minimize the squared error
# =============================================================================
 #For autoencoder part
loss_autoenc = tf.reduce_sum(tf.reduce_mean(tf.pow(y_true - y_pred, 2),axis=1))

 #For classification part
loss_classifier =  tf.reduce_sum(tf.reduce_mean((\
                         tf.nn.sigmoid_cross_entropy_with_logits(\
                             labels=labels,logits=logits)),axis=1))
 #sparsity loss
RhoJ = tf.clip_by_value(tf.concat([RhoJEH1, RhoJEH2, RhoJDH1, RhoJDH2],
                                  axis = 0),1e-10,1-1e-10)
Rho = tf.constant(RHO) #Desired average activation value

loss_sparsity = tf.reduce_mean(tf.add(tf.multiply(Rho,tf.log(tf.div(Rho,RhoJ)))
                    ,tf.multiply((1-Rho),tf.log(tf.div((1-Rho),(1-RhoJ))))))

 #regularization cost
loss_regulariation = tf.reduce_sum(tf.stack(map(lambda x: 
                        tf.nn.l2_loss(x), weights.values()),axis=0))
    
    
 #TransE loss
pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
neg_per_pos = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 2, keep_dims = True)		
neg = tf.reduce_min(neg_per_pos,axis=1)
loss_transe = tf.reduce_sum(tf.maximum(pos - neg + MARGIN, 0))
    
loss_nontranse = ALPHA*loss_autoenc + GAMMA*loss_classifier + loss_sparsity + \
                    BETA*loss_regulariation 


loss = loss_transe + loss_nontranse

stacked_loss = tf.stack([loss_autoenc, loss_classifier, loss_sparsity, \
                        loss_regulariation,loss_transe],axis = 0)
                                
#optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.9,\
                                   beta2=0.999,epsilon=1e-08)

unshared_vars = []
unshared_vars.extend(weights.values())
unshared_vars.extend(biases.values())
unshared_vars.append(rel_embeddings)

grads_vars_unshared_vars = optimizer.compute_gradients(loss,unshared_vars)
grads_vars_1 = optimizer.compute_gradients(loss_nontranse,[ent_embeddings])
grads_vars_2 = optimizer.compute_gradients(loss_transe,[ent_embeddings])


new_grads_vars_1 = []
new_grads_vars_2 = []

for gv1,gv2 in zip(grads_vars_1, grads_vars_2):
    g1,v1=  gv1
    g2,v2 = gv2
    
    assert v1==v2
    print (g1)
    new_grads_vars_1.append((tf.where(tf.greater_equal(tf.multiply(g1,g2),0),g1,tf.multiply(0.0,g1)),v1))
    new_grads_vars_2.append((tf.where(tf.greater_equal(tf.multiply(g1,g2),0),g2,tf.multiply(0.0,g2)),v2))
    
op1 = optimizer.apply_gradients(new_grads_vars_1)
op2 = optimizer.apply_gradients(new_grads_vars_2)
op_unshared = optimizer.apply_gradients(grads_vars_unshared_vars)

saver = tf.train.Saver(max_to_keep = 4)
# =============================================================================
#  Initialize the variables (i.e. assign their default value)
# =============================================================================
init = tf.global_variables_initializer()

# =============================================================================
#  Start Training
# =============================================================================
# Start a new TF session
conf = tf.ConfigProto()
conf.gpu_options.allow_growth=True
conf.log_device_placement=False #@myself: use this for debugging
conf.allow_soft_placement=True
P = Pool()
with tf.Session(config = conf) as sess:

    # Run the initializer
    sess.run(init)
    sess.run(normalize_rel_op)
    # Training
    NOW_DISPLAY = False
    epoch=1
    step=1    
    temp_Type2Data = deepcopy(Type2Data)
    mean_losses = np.zeros([5])
    mean_delta = 0
    while (epoch < NUM_EPOCHS):
        if sum(map(len,temp_Type2Data.values())) < 0.1 * TOT_RELATIONS:
            epoch += 1
            NOW_DISPLAY = True
            temp_Type2Data = deepcopy(Type2Data)            
            
        #prepare the data
        POS_DATA,NEG_DATA,batch_x = SampleTypeWise(temp_Type2Data,ent2type,\
                       Nsamples_Transe_Pos,BATCH_SIZE,\
                       relations_dic_h,relations_dic_t,VOCABULARY_SIZE,\
                       Nsamples_Transe_Neg,NUM_TYPES_BATCH,NUM_TYPES,1)
        POS_DATA = np.array(POS_DATA)
        NEG_DATA = np.array(NEG_DATA)
        # Get the next batch of type labels
        batch_y = generate_labels(batch_x)  

        sess.run(normalize_entity_op)                

        # Run optimization op (backprop) and cost op (to get loss value)
        _1,_2,_3,l_array,margin_cl,delta_cl = sess.run([op_unshared, op1, op2, stacked_loss,
                    margin_classification,delta_classification], feed_dict=\
                        {
                            X: batch_x,
                            Y: batch_y,
                            pos_h:POS_DATA[:,0],
                            pos_r:POS_DATA[:,1],
                            pos_t:POS_DATA[:,2],                        
                            neg_h:NEG_DATA[:,:,0],
                            neg_r:NEG_DATA[:,:,1],
                            neg_t:NEG_DATA[:,:,2],
                        })

        
        
        
        
        
        
        l = np.sum(l_array)  
        mean_losses+=np.array(l_array)
        mean_delta+=delta_cl
        # Display logs per step
        if NOW_DISPLAY or step==1:
            mean_losses/=float(step)
            mean_delta/=float(step)
            print('Epoch %i : Minibatch Loss: %f\n' % (epoch, l))
            l_array = [str(token) for token in mean_losses]
            print('Epoch %i : Mean Loss Array: %s\n' % (epoch,','.join(l_array)))
            print('Epoch %i : Margin Classification: %f\n'% \
                                     (epoch,margin_cl))         
            print('Epoch %i : Mean Delta Classification: %f\n\n'% \
                                     (epoch,mean_delta))
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            with open(LOG_DIR+'/loss.txt','a+') as fp:
                fp.write('Epoch %i : Minibatch Loss: %f\n' % \
                                                     (epoch, l))
                fp.write('Epoch %i : Mean Loss Array: %s\n'% \
                                         (epoch,','.join(l_array)))
                fp.write('Epoch %i : Margin Classification: %f\n'% \
                                         (epoch,margin_cl))
                fp.write('Epoch %i : Mean Delta Classification: %f\n\n'% \
                                         (epoch,mean_delta)) 
            mean_losses = np.zeros([5])
            mean_delta = 0
            step=1

                
        if (NOW_DISPLAY) and epoch%10==1:
            # Evaluation on Training Data
            MRT = []
            MRH = []
            skip_rate = int(evalsubset_relations.shape[0]/BATCH_EVAL)
            for j in xrange(0, skip_rate):
                eval_batch_h = evalsubset_relations[j::skip_rate,0]
                eval_batch_r = evalsubset_relations[j::skip_rate,1] 
                eval_batch_t = evalsubset_relations[j::skip_rate,2] 
                assert eval_batch_h.shape[0]==BATCH_EVAL
                
                indexes_h, indexes_t = sess.run([indices_h,indices_t], \
                                        feed_dict = 
                                 {
                                    eval_h:eval_batch_h,                                
                                    eval_r:eval_batch_r,                                
                                    eval_t:eval_batch_t,
                                    eval_to_rank:xrange(VOCABULARY_SIZE) 
                                 })
                mrt, mrh = map(Evaluate_MR,*[(eval_batch_t.tolist(),\
                              eval_batch_h.tolist()), (indexes_t.tolist(),\
                                                 indexes_h.tolist()), (P,P)])
                MRT.extend(mrt)
                MRH.extend(mrh)            
                
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), epoch)                
            with open(LOG_DIR+'/progress.txt','a+') as fp:        
                fp.write('Epoch %i: Minibatch MRT: %f\n' % (epoch, \
                                                        np.mean(MRT)))
                fp.write('Epoch %i: Minibatch MRH: %f\n\n' % (epoch, \
                                                        np.mean(MRH)))

        NOW_DISPLAY = False
        step += 1
        
P.close()