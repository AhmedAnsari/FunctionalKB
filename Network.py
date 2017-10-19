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

# =============================================================================
#  Import data regarding embedding relations and type information
# =============================================================================
types = json.load(open('types_fb.json'))
relations_dic_h = json.load(open('relations_dic_h.json'))
relations_dic_t = json.load(open('relations_dic_t.json'))
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
#  Training Parameters
# =============================================================================
LEARNING_RATE = 0.01
NUM_STEPS = 30000 #@myself: need to set this
BATCH_SIZE = 64 #@myself: need to set this
DISPLAY_STEP = 1000 #@myself: need to set this
EXAMPLES_TO_SHOW = 10 #@myself: need to set this
NUM_TYPES_PER_BATCH = 32
RHO = 0.05 #Desired average activation value
BETA = 0.5
MARGIN = 1
# =============================================================================
#  Network Parameters-1 #First let us solve only for Type loss
# =============================================================================
NUM_HIDDEN_1 = 128 # 1st layer num features
NUM_HIDDEN_2 = 256 # 2nd layer num features (the latent dim)
NUM_INPUT = EMBEDDING_SIZE = 64 #@myself: need to set this
VOCABULARY_SIZE = 14951
RELATIONS_SIZE = 1345
LOG_DIR = 'Logs/' 
# =============================================================================
# tf Graph input 
# =============================================================================
#Define the entity embedding matrix to be uniform in a unit cube
ent_embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE],
                                           -1.0, 1.0),name = 'Ent_Embedding')
#Define the relation embedding matrix to be uniform in a unit cube
rel_embeddings = tf.Variable(tf.random_uniform([RELATIONS_SIZE, EMBEDDING_SIZE],
                                           -1.0, 1.0),name = 'Rel_Embedding')#@myself: have to switch to tf.get_variable for xavier stuff

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

#look up the vector for each of the source words in the batch for Jtype part
embed = tf.nn.embedding_lookup(ent_embeddings, X)

#look up the vector for each of the source words in the batch for TransE part
pos_h_e = tf.nn.embedding_lookup(ent_embeddings, pos_h)
pos_t_e = tf.nn.embedding_lookup(ent_embeddings, pos_t)
pos_r_e = tf.nn.embedding_lookup(rel_embeddings, pos_r)
neg_h_e = tf.nn.embedding_lookup(ent_embeddings, neg_h)
neg_t_e = tf.nn.embedding_lookup(ent_embeddings, neg_t)
neg_r_e = tf.nn.embedding_lookup(rel_embeddings, neg_r)






weights = {
    'encoder_h1': tf.Variable(tf.random_normal([NUM_INPUT, NUM_HIDDEN_1])),
    'encoder_h2': tf.Variable(tf.random_normal([NUM_HIDDEN_1, NUM_HIDDEN_2])),
    'decoder_h1': tf.Variable(tf.random_normal([NUM_HIDDEN_2, NUM_HIDDEN_1])),
    'decoder_h2': tf.Variable(tf.random_normal([NUM_HIDDEN_1, NUM_INPUT])),
    'classification_h': tf.Variable(tf.random_normal([NUM_HIDDEN_2,
                                                      NUM_TYPES]))
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([NUM_HIDDEN_1])),
    'encoder_b2': tf.Variable(tf.random_normal([NUM_HIDDEN_2])),
    'decoder_b1': tf.Variable(tf.random_normal([NUM_HIDDEN_1])),
    'decoder_b2': tf.Variable(tf.random_normal([NUM_INPUT])),
    'classification_h': tf.Variable(tf.random_normal([NUM_TYPES]))
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
    
    RhoJEH1 = tf.reduce_mean(layer_1,0)
    RhoJEH2 = tf.reduce_mean(layer_2,0) 
    return layer_2, RhoJEH1, RhoJEH2


# =============================================================================
#  Building the decoder
# =============================================================================
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    RhoJDH1 = tf.reduce_mean(layer_1,0)
    RhoJDH2 = tf.reduce_mean(layer_2,0)     
    return layer_2, RhoJDH1, RhoJDH2

# =============================================================================
# Building the classification layer
# =============================================================================
def classify(x):
    #classification hidden layer with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['classification_h']),
                                   biases['classification_h']))
    return layer_1

# =============================================================================
#  Construct model
# =============================================================================
encoder_op, RhoJEH1, RhoJEH2 = encoder(embed)
decoder_op, RhoJDH1, RhoJDH2 = decoder(encoder_op)
classifier_op = classify(encoder_op)

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
                                  axis = 0),1e-10,1e10)
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
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, NUM_STEPS+1): 
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
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x,Y:batch_y, \
                        pos_h:posh_batch,
                        pos_r:posr_batch,
                        pos_t:post_batch,                        
                        neg_h:negh_batch,
                        neg_r:negr_batch,
                        neg_t:negt_batch,                        
                        })
        
        # Display logs per step
        if i % DISPLAY_STEP == 0 or i == 1:
            print('Step %f: Minibatch Loss: %f' % (i, l))
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)
