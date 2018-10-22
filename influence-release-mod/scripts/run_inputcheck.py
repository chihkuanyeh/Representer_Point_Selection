"""
Dataset Debugging Experiment code
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import sys
sys.path.append('../')
import math
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
from tensorflow.contrib.learn.python.learn.datasets import base

import scipy
import sklearn

import influence.experiments as experiments
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_mnist import load_small_mnist, load_mnist
from influence.experiments import get_try_check, test_mislabeled_detection_batch

import tensorflow as tf
import pickle

np.random.seed(42)

data_sets = load_mnist('data')

# Filter out two classes
pos_class = 1
neg_class = 7

X_train = data_sets.train.x
Y_train = data_sets.train.labels
X_test = data_sets.test.x
Y_test = data_sets.test.labels

# Comment this part out to work on the whole data for binary.
X_train, Y_train = dataset.sample_random(X_train, Y_train, no_per_class=1000)
X_test, Y_test = dataset.sample_random(X_test, Y_test, no_per_class=200)

X_train, Y_train = dataset.filter_dataset(X_train, Y_train, pos_class, neg_class)
X_test, Y_test = dataset.filter_dataset(X_test, Y_test, pos_class, neg_class)
np.savez('data/mnist_binary_%dvs%d_small.npz'%(pos_class, neg_class),
            x_train = X_train,
            y_train = Y_train,
            x_test = X_test,
            y_test = Y_test)

## If using logistic regression to train
lr_train = DataSet(X_train, np.array((Y_train + 1) / 2, dtype=int))
lr_validation = None
lr_test = DataSet(X_test, np.array((Y_test + 1) / 2, dtype=int))
lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

num_classes = 2
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels
weight_decay = 0.01
batch_size = 100
initial_learning_rate = 0.001
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

num_params = 784

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=lr_data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist-%dvs%d-logreg'%(pos_class, neg_class))

tf_model.train()
#tf_model.load_checkpoint(0)

print('Training complete')
##############################################
### Flipping experiment
##############################################
X_train = np.copy(tf_model.data_sets.train.x)
Y_train = np.copy(tf_model.data_sets.train.labels)
X_test = np.copy(tf_model.data_sets.test.x)
Y_test = np.copy(tf_model.data_sets.test.labels) 

## what percentage of data to corrupt
perc = 0.4
num_train_examples = Y_train.shape[0]

# Random seed for each experiments
num_random_seeds = 10

# number of points to flip the label
num_to_flip = int(num_train_examples * perc)
checkpoint = 9

# Save the experiment results here 
dims = (checkpoint, num_random_seeds, 3)
fixed_influence_loo_results = np.zeros(dims)
fixed_loss_results = np.zeros(dims)
fixed_random_results = np.zeros(dims)
fixed_ours_results = np.zeros(dims)
fixed_ours_train_results = np.zeros(dims)
fixed_ours_avg_results = np.zeros(dims) 
flipped_results = np.zeros((num_random_seeds, 3))

# Save the original result without flipping
orig_results = tf_model.sess.run(
    [tf_model.loss_no_reg, tf_model.accuracy_op], 
    feed_dict=tf_model.all_test_feed_dict)

print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))

num_train_examples = Y_train.shape[0] 

exp_results = dict()
# repeat experiments 10 times
for exp_no in range(num_random_seeds):
    print('--Experiment %d'%exp_no)

    # select idx to corrupt
    np.random.seed(exp_no)
    idx_to_flip = np.random.choice(num_train_examples, num_to_flip, replace=False)

    # binary
    Y_train_flipped = np.copy(Y_train)
    Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip]

    #assert(np.sum(Y_train_flipped[idx_to_flip] + Y_train[idx_to_flip]) == 0)
    ## Binary LR takes labels 0 and 1 only...

    # save the corrupted data
    np.savez('data/mnist_%dvs%d_corrupt_%d'%(pos_class, neg_class, exp_no), 
             x_train = X_train,
             y_train = Y_train_flipped * 2 - 1,
             x_test = X_test,
             y_test = Y_test * 2 -1)

    # Save the performance after corrupting the data
    tf_model.update_train_x_y(X_train, Y_train_flipped)
    tf_model.train()

    flipped_results[exp_no, 1:] = tf_model.sess.run(
        [tf_model.loss_no_reg, tf_model.accuracy_op], 
        feed_dict=tf_model.all_test_feed_dict)
    print('--Corrupted %d points'%num_to_flip)
    print('--Corrupted loss: %.5f. Accuracy: %.3f' % (
            flipped_results[exp_no, 1], flipped_results[exp_no, 2]))

    # get the influence and loss values
    train_losses = tf_model.sess.run(tf_model.indiv_loss_no_reg, feed_dict=tf_model.all_train_feed_dict)
    train_loo_influences = tf_model.get_loo_influences()
    #pickle.dump(train_losses, open('output/losses_%d_%d'%(exp_no, c), 'wb'))
    #pickle.dump(train_loo_influences, open('output/inf_%d_%d'%(exp_no, c), 'wb'))

    # NOTE we need representer values here
    weight_fname = '../../output/weight_matrix_mnist_binary_%d_27.pkl'%exp_no
    alpha = pickle.load(open(weight_fname, 'rb'))

    #  just alphas for one class
    ours_influences = np.sum(alpha[0] * alpha[1], axis=1)

    # alpha times input
    ours_influences_trains = ours_influences * np.sum(X_train * X_train, axis=1)

    # average
    ours_influences_avg = np.mean(alpha[0], axis=1)

    for c in range(checkpoint):
        # c is the porportion of the data to check 
        num_checks = int(num_train_examples / 20) * (c + 1)
        print('---- checking %d points (%f of training examples)'%(num_checks, (c+1) / 20.))

        """
        try_check = get_try_check(tf_model, X_train, Y_train, Y_train_flipped, X_test, Y_test, retrain_no='%s-%s'%(exp_no, c))

        # Pick by top loss to fix
        idx_to_check = np.argsort(np.abs(train_losses))[-num_checks:]    
        fixed_loss_results[c, exp_no, :] = try_check(idx_to_check, 'Loss')

        # Randomly pick stuff to fix
        idx_to_check = np.random.choice(num_train_examples, size=num_checks, replace=False)    
        fixed_random_results[c, exp_no, :] = try_check(idx_to_check, 'Random')

        # Pick by the representer value absolute
        idx_to_check = np.argsort(np.abs(ours_influences))[-num_checks:]
        fixed_ours_results[c, exp_no, :] = try_check(idx_to_check, 'Ours')

        # Pick by LOO influence    
        idx_to_check = np.argsort(train_loo_influences)[-num_checks:]
        fixed_influence_loo_results[c, exp_no, :] = try_check(idx_to_check, 'Influence (LOO)')

        """
        fixed_influence_loo_results[c, exp_no, :], \
        fixed_loss_results[c, exp_no, :],\
        fixed_random_results[c, exp_no, :],\
        fixed_ours_results[c, exp_no, :] = test_mislabeled_detection_batch(
                tf_model, 
                X_train,
                Y_train,
                Y_train_flipped,
                X_test, Y_test,
                train_losses, train_loo_influences, ours_influences,
                2, num_checks)

        # Pick by our influence alpha times input
        #idx_to_check = np.argsort(np.abs(ours_influences_trains))[-num_checks:]
        #fixed_ours_train_results[c, exp_no, :] = try_check(idx_to_check, 'Ours Abs with Train')

        # Pick by our influencee alhpa times input
        #idx_to_check = np.argsort(np.abs(ours_influences_avg))[-num_checks:]
        #fixed_ours_avg_results[c, exp_no, :] = try_check(idx_to_check, 'Ours Abs with Avg')

print('Done. Saving...')
file_name = 'mnist_%dvs%d_inputcheck_results.npz'%(pos_class, neg_class)

np.savez(
    'output/%s'%(file_name), 
    orig_results=orig_results,
    flipped_results=flipped_results,
    fixed_influence_loo_results=fixed_influence_loo_results,
    fixed_loss_results=fixed_loss_results,
    fixed_random_results=fixed_random_results,
    fixed_ours_results=fixed_ours_results,
    #fixed_ours_train_results=fixed_ours_train_results,
    #fixed_ours_avg_results=fixed_ours_avg_results,
)

