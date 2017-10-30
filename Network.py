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
from Sampler import SampleTransEData
import itertools
import random
import datetime
import cPickle as pkl
from Evaluation import Evaluate_MR
from pathos.threading import ThreadPool as Pool


# =============================================================================
#  Training Parameters
# =============================================================================
LEARNING_RATE = 0.01
NUM_STEPS = 300000 #@myself: need to set this
BATCH_SIZE = 128 #@myself: need to set this
DISPLAY_STEP = 1000 #@myself: need to set this
EVAL_STEP = 1 * DISPLAY_STEP
NUM_TYPES_PER_BATCH = 64
RHO = 0.005 #Desired average activation value
BETA = 0.5
MARGIN = 1
BATCH_EVAL = 32
# =============================================================================
#  Network Parameters-1 #First let us solve only for Type loss
# =============================================================================
NUM_HIDDEN_1 = 128 # 1st layer num features
NUM_HIDDEN_2 = 256 # 2nd layer num features (the latent dim)
NUM_INPUT = EMBEDDING_SIZE = 64 #@myself: need to set this
VOCABULARY_SIZE = 14951
RELATIONS_SIZE = 1345
LOG_DIR = 'Logs/'+str(datetime.datetime.now())
DEVICE = '/cpu:0'
# =============================================================================
#  Import data regarding embedding relations and type information
# =============================================================================
types = json.load(open('types_fb.json'))
relations_dic_h = pkl.load(open('relations_dic_h.pkl'))
relations_dic_t = pkl.load(open('relations_dic_t.pkl'))

#for evaluation during training
#relations = np.array(json.load(open('relations_hrt.json')))
evalsubset_relations_train = np.array(pkl.load(open\
               ('evalsubset_relations_train.pkl','r')))
evalsubset_relations_train = evalsubset_relations_train \
     [0:len(evalsubset_relations_train) - \
     len(evalsubset_relations_train) % BATCH_EVAL]
NUM_TYPES = len(types)

def generate_labels(data,batch):
    out = []
    for x in batch:
        out.append(len(data)*[0])
        for i in range(len(data)):
            if x in data[i]:
                out[-1][i]=1                    
    return out
# =============================================================================
# tf Graph input 
# =============================================================================
#Define the entity embedding matrix to be uniform in a unit cube
with tf.device(DEVICE):
    ent_embeddings = tf.get_variable(name='W_Ent',shape = [VOCABULARY_SIZE,\
                       EMBEDDING_SIZE],initializer = \
                        tf.contrib.layers.xavier_initializer())
    
    
                                               
    #Define the relation embedding matrix to be uniform in a unit cube
    rel_embeddings = tf.get_variable(name='W_Rel',shape = [RELATIONS_SIZE,\
                       EMBEDDING_SIZE],initializer = \
                        tf.contrib.layers.xavier_initializer())



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
    neg_h = tf.placeholder(tf.int32, [None])
    neg_r = tf.placeholder(tf.int32, [None])
    neg_t = tf.placeholder(tf.int32, [None])
    
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





with tf.device(DEVICE):
    
    weights = {
            
        'encoder_h1': tf.get_variable(name='W_encoder_h1',shape = \
                      [NUM_INPUT, NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer()),
    
        'encoder_h2': tf.get_variable(name='W_encoder_h2',shape = \
                      [NUM_HIDDEN_1, NUM_HIDDEN_2],initializer = \
                        tf.contrib.layers.xavier_initializer()),      
                                                                   
        'decoder_h1': tf.get_variable(name='W_decoder_h1',shape = \
                      [NUM_HIDDEN_2, NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer()),      
        'decoder_h2': tf.get_variable(name='W_decoder_h2',shape = \
                      [NUM_HIDDEN_1, NUM_INPUT],initializer = \
                        tf.contrib.layers.xavier_initializer()),      
        'classification_h': tf.get_variable(name='W_classification_h',shape = \
                            [NUM_HIDDEN_2, NUM_TYPES],initializer = \
                            tf.contrib.layers.xavier_initializer()),
    }
    
    biases = {
        'encoder_b1': tf.get_variable(name='W_encoder_b1',shape = \
                      [NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer()),
        'encoder_b2': tf.get_variable(name='W_encoder_b2',shape = \
                      [NUM_HIDDEN_2],initializer = \
                        tf.contrib.layers.xavier_initializer()),
        'decoder_b1': tf.get_variable(name='W_decoder_b1',shape = \
                       [NUM_HIDDEN_1],initializer = \
                        tf.contrib.layers.xavier_initializer()),      
        'decoder_b2': tf.get_variable(name='W_decoder_b2',shape = \
                      [NUM_INPUT],initializer = \
                        tf.contrib.layers.xavier_initializer()),      
        'classification_b': tf.get_variable(name='W_classification_b',shape = \
                            [NUM_TYPES],initializer = \
                            tf.contrib.layers.xavier_initializer()),
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
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['classification_h']),
                                   biases['classification_b']))
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

#evaluation part of network
_t, indices_t = predict_rank(eval_h_e, eval_r_e, eval_to_rank_e, \
                    BATCH_EVAL, VOCABULARY_SIZE, VOCABULARY_SIZE)
_h, indices_h = predict_rank(eval_t_e, eval_r_e, eval_to_rank_e, \
                    BATCH_EVAL, VOCABULARY_SIZE, VOCABULARY_SIZE)
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
loss_autoenc = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

 #For classification part
loss_classifier =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,logits=logits))
 #sparsity loss
RhoJ = tf.clip_by_value(tf.concat([RhoJEH1, RhoJEH2, RhoJDH1, RhoJDH2],
                                  axis = 0),1e-10,1-1e-10)
Rho = tf.constant(RHO) #Desired average activation value

loss_sparsity = tf.reduce_mean(tf.add(tf.multiply(Rho,tf.log(tf.div(Rho,RhoJ)))
,tf.multiply((1-Rho),tf.log(tf.div((1-Rho),(1-RhoJ))))))
 
 #regularization cost
loss_regulariation = BETA*tf.reduce_mean(tf.stack(map(lambda x: 
                        tf.nn.l2_loss(x), weights.values()),axis=0))
    
 #TransE loss
pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)		
loss_transe = tf.reduce_sum(tf.maximum(pos - neg + MARGIN, 0))
    
loss = loss_autoenc + loss_classifier + loss_sparsity + \
        loss_regulariation + loss_transe

stacked_loss = tf.stack([loss_autoenc, loss_classifier, loss_sparsity, \
                        loss_regulariation, loss_transe],axis = 0)
                                
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
saver = tf.train.Saver()
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

    # Training
    for step in range(1, NUM_STEPS+1): 
        #prepare the data
        #first select the Types that will be in this batch
        indices = random.sample(range(len(types)),NUM_TYPES_PER_BATCH)
        #then randomly sample data from these types
        data = [types[i] for i in indices]
        data_flat = list(itertools.chain(*data))        
        # Get the next batch of input data
        batch_x = random.sample(data_flat,BATCH_SIZE)
        # Get the next batch of type labels
        batch_y = generate_labels(types,batch_x)        
        #for TransE loss part, get positive and negative samples
        posh_batch,posr_batch,post_batch,negh_batch,negr_batch,negt_batch = \
        SampleTransEData(relations_dic_h,relations_dic_t,\
                         batch_x,VOCABULARY_SIZE)
        
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, l_array= sess.run([optimizer, loss, stacked_loss], feed_dict=\
                        {
                            X: batch_x,Y:batch_y,
                            pos_h:posh_batch,
                            pos_r:posr_batch,
                            pos_t:post_batch,                        
                            neg_h:negh_batch,
                            neg_r:negr_batch,
                            neg_t:negt_batch,                        
                        })
        
        # Display logs per step
        if step % DISPLAY_STEP == 0 or step == 1:
            print('Step %i: Minibatch Loss: %f\n' % (step, l))
            l_array = [str(token) for token in l_array]
            print('Step %i: Loss Array: %s\n' % (step,','.join(l_array)))
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)
            with open(LOG_DIR+'/loss.txt','a+') as fp:
                fp.write('Step %i: Minibatch Loss: %f\n' % (step, l))
                fp.write('Step %i: Loss Array: %s\n'% (step,','.join(l_array)))
                
        if step % EVAL_STEP == 0 or step == 1:
            # Evaluation on Training Data
            MRT = []
            MRH = []
            skip_rate = int(evalsubset_relations_train.shape[0]/BATCH_EVAL)
            for j in range(0, skip_rate):
                eval_batch_h = evalsubset_relations_train[j::skip_rate,0]
                eval_batch_r = evalsubset_relations_train[j::skip_rate,1] 
                eval_batch_t = evalsubset_relations_train[j::skip_rate,2] 
                assert eval_batch_h.shape[0]==BATCH_EVAL
                
                indexes_h, indexes_t = sess.run([indices_h,indices_t], feed_dict = \
                                 {
                                    eval_h:eval_batch_h,                                
                                    eval_r:eval_batch_r,                                
                                    eval_t:eval_batch_t,
                                    eval_to_rank:range(VOCABULARY_SIZE) 
                                 })
                mrt, mrh = map(Evaluate_MR,*[(eval_batch_t.tolist(),\
                              eval_batch_h.tolist()), (indexes_t.tolist(),\
                                                 indexes_h.tolist()), (P,P)])
                MRT.extend(mrt)
                MRH.extend(mrh)            
                

            with open(LOG_DIR+'/progress.txt','a+') as fp:        
                fp.write('Step %i: Minibatch MRT: %f\n' % (step, np.mean(MRT)))
                fp.write('Step %i: Minibatch MRH: %f\n' % (step, np.mean(MRH)))
P.close()