"""
Model for CIFAR using VGG features
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

class CIFAR_MLP(GenericNeuralNet):
    ## The last three layers of the VGG net
    def __init__(self, input_dim, idx, **kwargs):
        self.input_dim = input_dim
        self.idx = idx
        super(CIFAR_MLP, self).__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        if self.idx == 31:
            for layer in ['fc1', 'fc2', 'fc3']:        
                for var_name in ['weights', 'biases']:
                    temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                    all_params.append(temp_tensor)      
        elif self.idx == 34:
            for layer in ['fc']:        
                for var_name in ['weights', 'biases']:
                    temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                    all_params.append(temp_tensor)      
        elif self.idx == 32:
            for layer in ['fc1', 'fc2']:        
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


    def inference(self, input_x):
        if self.idx == 31:
            # first fc layer
            with tf.variable_scope('fc1'):
                weights1 = variable(
                    'weights', 
                    [self.input_dim * 4096],
                    tf.contrib.layers.xavier_initializer())            
                biases1 = variable(
                    'biases',
                    [4096],
                    tf.constant_initializer(0.0))

                hidden = tf.matmul(input_x, tf.reshape(weights1, [self.input_dim, 4096])) + biases1
                hidden = tf.nn.relu(hidden)

            # dropout
            with tf.variable_scope('do1'):
                hidden = tf.layers.dropout(hidden, rate=0.5)

            # second fc layer 
            with tf.variable_scope('fc2'):
                weights2 = variable(
                    'weights', 
                    [4096 * 4096],
                    tf.contrib.layers.xavier_initializer())            
                biases2 = variable(
                    'biases',
                    [4096],
                    tf.constant_initializer(0.0))

                hidden = tf.matmul(hidden, tf.reshape(weights2, [4096, 4096])) + biases2
                hidden = tf.nn.relu(hidden)

            # dropout
            with tf.variable_scope('do2'):
                hidden = tf.layers.dropout(hidden, rate=0.5)

            # last fc layer
            with tf.variable_scope('fc3'):
                weights3 = variable(
                    'weights', 
                    [4096 * 10],
                    tf.contrib.layers.xavier_initializer())            
                biases3 = variable(
                    'biases',
                    [10],
                    tf.constant_initializer(0.0))

                logits = tf.matmul(hidden, tf.reshape(weights3, [4096, 10])) + biases3

        elif self.idx == 32:
            # first fc layer 
            with tf.variable_scope('fc1'):
                weights2 = variable(
                    'weights', 
                    [4096 * 4096],
                    tf.contrib.layers.xavier_initializer())            
                biases2 = variable(
                    'biases',
                    [4096],
                    tf.constant_initializer(0.0))

                hidden = tf.matmul(input_x, tf.reshape(weights2, [4096, 4096])) + biases2
                hidden = tf.nn.relu(hidden)

            # dropout
            with tf.variable_scope('do1'):
                hidden = tf.layers.dropout(hidden, rate=0.5)

            # last fc layer
            with tf.variable_scope('fc2'):
                weights3 = variable(
                    'weights', 
                    [4096 * 10],
                    tf.contrib.layers.xavier_initializer())            
                biases3 = variable(
                    'biases',
                    [10],
                    tf.constant_initializer(0.0))

                logits = tf.matmul(hidden, tf.reshape(weights3, [4096, 10])) + biases3

        elif self.idx == 34:
            # otherwise just the last layer
            with tf.variable_scope('fc'):
                weights3 = variable(
                    'weights', 
                    [4096 * 10],
                    tf.contrib.layers.xavier_initializer())            
                biases3 = variable(
                    'biases',
                    [10],
                    tf.constant_initializer(0.0))

                logits = tf.matmul(input_x, tf.reshape(weights3, [4096, 10])) + biases3

        return logits

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds
