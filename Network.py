#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:14:31 2017

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

# =============================================================================
#  Import data regarding embedding relations and type information
# =============================================================================
types = json.load(open('types_fb.json'))
relations = json.load(open('relations_hrt.json'))

# =============================================================================
#  Training Parameters
# =============================================================================
learning_rate = 0.01
num_steps = 30000 #@myself: need to set this
batch_size = 256 #@myself: need to set this

display_step = 1000 #@myself: need to set this
examples_to_show = 10 #@myself: need to set this
types_per_batch = 32 #number of types in a given batch
# =============================================================================
#  Sampler Function to generate minibatches for type data
# =============================================================================
def SampleTypeData(data,batch_size,types_per_batch):
    
    return


# =============================================================================
#  Network Parameters-1 #First let us solve only for Type loss
# =============================================================================
num_hidden_1 = 128 # 1st layer num features
num_hidden_2 = 64 # 2nd layer num features (the latent dim)
num_input = embedding_size = 256 #@myself: need to set this
vocabulary_size = 14951
 
# =============================================================================
# tf Graph input 
# =============================================================================
#Define the embedding matrix to be uniform in a unit cube
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

#this will contain embedding ids in given a batch
X = tf.placeholder(tf.int32, shape=[batch_size])

#look up the vector for each of the source words in the batch
embed = tf.nn.embedding_lookup(embeddings, X)


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
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
    return layer_2


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
    return layer_2

# =============================================================================
#  Construct model
# =============================================================================
encoder_op = encoder(embed)
decoder_op = decoder(encoder_op)

# =============================================================================
#  Prediction
# =============================================================================
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = embed

# =============================================================================
#  Define loss and optimizer, minimize the squared error
# =============================================================================
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

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
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of input data
        batch_x = SampleTypeData(types,batch_size) #@myself: need to define a sampler function

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

#    # Testing
#    # Encode and decode images from test set and visualize their reconstruction.
#    n = 4
#    canvas_orig = np.empty((28 * n, 28 * n))
#    canvas_recon = np.empty((28 * n, 28 * n))
#    for i in range(n):
#        # MNIST test set
#        batch_x, _ = mnist.test.next_batch(n)
#        # Encode and decode the digit image
#        g = sess.run(decoder_op, feed_dict={X: batch_x})
#
#        # Display original images
#        for j in range(n):
#            # Draw the original digits
#            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
#                batch_x[j].reshape([28, 28])
#        # Display reconstructed images
#        for j in range(n):
#            # Draw the reconstructed digits
#            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
#                g[j].reshape([28, 28])
#
#    print("Original Images")
#    plt.figure(figsize=(n, n))
#    plt.imshow(canvas_orig, origin="upper", cmap="gray")
#    plt.show()
#
#    print("Reconstructed Images")
#    plt.figure(figsize=(n, n))
#    plt.imshow(canvas_recon, origin="upper", cmap="gray")
#    plt.show()