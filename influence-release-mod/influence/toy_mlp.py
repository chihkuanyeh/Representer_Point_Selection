"""
Model for Toy dataset using raw features
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 

import os.path
import time
import IPython
import tensorflow as tf
import math

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet


class MLP(GenericNeuralNet):

    def __init__(self, input_dim, num_hidden_units, weight_decay, **kwargs):
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.num_hidden_units = num_hidden_units
        super(MLP, self).__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        for layer in ['fc1', 'fc2','fc3']:        
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)      
        return all_params        

    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])

        for step in xrange(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def train(self, num_steps, 
              iter_to_switch_to_sgd=3000,
              save_checkpoints=True, verbose=True):

        if verbose: print('Training for %s steps' % num_steps)

        sess = self.sess            

        for step in xrange(num_steps):

            start_time = time.time()

            if step < iter_to_switch_to_sgd:
                feed_dict = self.all_train_feed_dict          
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

            else: 
                feed_dict = self.all_train_feed_dict          
                _, loss_val = sess.run([self.train_sgd_op, self.total_loss], feed_dict=feed_dict)          

            duration = time.time() - start_time

            if verbose:
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.8f (%.3f sec)' % (step, loss_val, duration))

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 100 == 0 or (step + 1) == num_steps:
                if save_checkpoints: self.saver.save(sess, self.checkpoint_file, global_step=step)
                if verbose: self.print_model_eval()

    def inference(self, input_x):        
        # first fc layer
        with tf.variable_scope('fc1'):
            weights1 = variable(
                'weights', 
                [self.input_dim * self.num_hidden_units],
                tf.contrib.layers.xavier_initializer())            
            biases1 = variable(
                'biases',
                [self.num_hidden_units],
                tf.constant_initializer(0.0))

            hidden = tf.matmul(input_x, tf.reshape(weights1, [self.input_dim, self.num_hidden_units])) + biases1
            hidden = tf.nn.relu(hidden)
        # second fc layer 
        with tf.variable_scope('fc2'):
            weights2 = variable(
                'weights', 
                [self.num_hidden_units * self.num_hidden_units],
                tf.contrib.layers.xavier_initializer())            
            biases2 = variable(
                'biases',
                [self.num_hidden_units],
                tf.constant_initializer(0.0))

            fc2_out = tf.matmul(hidden, tf.reshape(weights2, [self.num_hidden_units, self.num_hidden_units])) + biases2
            self.fc2_out = tf.nn.relu(fc2_out)

        # third fc layer
        with tf.variable_scope('fc3'):
            self.weights3 = variable(
                'weights', 
                [self.num_hidden_units * self.num_classes],
                tf.contrib.layers.xavier_initializer())            
            self.biases3 = variable(
                'biases',
                [self.num_classes],
                tf.constant_initializer(0.0))

            logits = tf.matmul(self.fc2_out, tf.reshape(self.weights3, [self.num_hidden_units, self.num_classes])) + self.biases3

        return logits

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds
