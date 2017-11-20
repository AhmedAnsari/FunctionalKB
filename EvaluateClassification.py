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
from os import sys
from skmultilearn.adapt import MLkNN
import random
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# =============================================================================
#  Training Parameters
# =============================================================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 160 #@myself: need to set this
NUM_TYPES_BATCH = BATCH_SIZE 
RHO = 0.005 #Desired average activation value
BETA = 0.5
ALPHA = 10
GAMMA = 5
MARGIN = 1
BATCH_EVAL = 32
NUM_EPOCHS = 1000
Pos2NegRatio_Transe = 4
Nsamples_Transe = 160*Pos2NegRatio_Transe
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
## =============================================================================
## tf Graph input 
## =============================================================================
##Define the entity embedding matrix to be uniform in a unit cube
with tf.device(DEVICE):
    ent_embeddings = tf.get_variable(name='W_Ent',shape = [VOCABULARY_SIZE,\
                       EMBEDDING_SIZE],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True))
    
    
                                               
    #Define the relation embedding matrix to be uniform in a unit cube
    rel_embeddings = tf.get_variable(name='W_Rel',shape = [RELATIONS_SIZE,\
                       EMBEDDING_SIZE],initializer = \
                        tf.contrib.layers.xavier_initializer(uniform = True))



    #this will contain embedding ids in given a batch
    X = tf.placeholder(tf.int32, shape=[None])
    
    #this will contain the type labels for each embedding in the given batch
    Y = tf.placeholder(tf.float32, shape=[None,NUM_TYPES])
    
    

    
    

#look up the vector for each of the source words in the batch for Jtype part
embed = tf.nn.embedding_lookup(ent_embeddings, X)



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


#evaluation part of Classification
score_classification = tf.nn.sigmoid(classifier_op)


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


 #For classification part
loss_classifier =  tf.reduce_sum(tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,logits=logits)),axis=1))


saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
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

with tf.Session(config = conf) as sess:

    # Run the initializer
    sess.run(init)
    # Training   
    temp_Type2Data = deepcopy(Type2Data)
    mean_losses = np.zeros([5])
    mean_delta = 0
#    saver.restore(sess,sys.argv[-1])
    saver.restore(sess,'/Users/ghulam/Documents/Work@IBM/CODE/FunctionalKB/Logs/data/model.ckpt-411')
    
    
    #first get the feature vector for all the entities in Vocabulary
    batch_x = xrange(VOCABULARY_SIZE)
    
    feature_x = sess.run(score_classification,feed_dict = {X:batch_x}).tolist()
    
with open('features_for_classification.pkl','w') as fp:
    pkl.dump(feature_x,fp)
    
classifier = MLkNN(k = 20)

#feature_x = pkl.load(open('features_for_classification.pkl'))
#classifier = BinaryRelevance(GaussianNB())


Keys_Train = random.sample(ent2type.keys(),10000)
Keys_Test = list(ent2type.keys())
[Keys_Test.remove(val) for val in Keys_Train]
X_Train = [feature_x[key] for key in Keys_Train]
X_Test = [feature_x[key] for key in Keys_Test]
Y_Train = generate_labels(Keys_Train)
Y_Test = generate_labels(Keys_Test)
print('HEERE 1')
classifier.fit(np.array(X_Train), np.array(Y_Train))
print('HEERE 2')    
predictions = classifier.predict(np.array(X_Test))

print(accuracy_score(np.array(Y_Test),predictions))

preds = predictions.toarray()

def accuracy(input):
    data = input[0]
    true = input[1]
    size = len(data)
    FP = TP = FN = TN = 0
    for i in xrange(size):
        if true[i] == True:
            if data[i] == True:
                TP += 1
            else:
                FN += 1
        else:
            if data[i] == True:
                FP += 1
            else:
                TN += 1
    return (TP + TN)/float(size)

Y_Test = Y_Test
preds = preds.tolist()

print(np.mean(map(accuracy,zip(preds,Y_Test))))
        
