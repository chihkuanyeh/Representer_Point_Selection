"""
Compute influence function values for CIFAR10 using VGG-16 feautures.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np

import tensorflow as tf

import os, sys
sys.path.append('../')

import influence.experiments as experiments
from influence.cifar_mlp import CIFAR_MLP
import pickle

from load_mnist import load_small_mnist, load_mnist
from load_cifar import *
from gen_vgg_features import *
from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
import h5py
import time

# First create the dataset object from the VGG features
print("Loading Data...")

## Use 34th layer

idx = 34
hf = h5py.File('data/vgg_features_cifar_%d.h5'%idx, 'r')
x_train = np.array(hf.get('train'))
x_test = np.array(hf.get('test'))
hf.close()

y_train, y_test = load_cifar_labels()
train = DataSet(x_train, y_train.flatten())
test = DataSet(x_test, y_test.flatten())
data_sets = base.Datasets(train=train, validation=None, test=test)

num_classes = 10
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.00001 
decay_epochs = [10000, 20000]
input_dim = x_train.shape[1]

model = CIFAR_MLP(
    input_dim=input_dim,
    idx = idx,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output', 
    log_dir='log',
    model_name='cifar_mlp_%d'%idx)

if idx == 31:
    num_steps = 500000
else:
    num_steps = 300000

model.train(
    num_steps=num_steps, 
    iter_to_switch_to_batch=10000000,
    iter_to_switch_to_sgd=10000000)
iter_to_load = num_steps - 1

print('Training done')

# Load the trained model
model.load_checkpoint(iter_to_load)

# Compute influence values for the set of test points

## NOTE specify the test point indices to compute the influence values of.
# Here, just compute for the first 1000 test points
num_test_points = 1000
test_indices = range(num_test_points)

num_train = len(model.data_sets.train.labels)
influences = None

# Compute influence function values.
time_lst = []
for test_idx in test_indices:
    start = time.time()
    influence = model.get_influence_on_test_loss(
                 [test_idx],
                 np.arange(num_train),
                 force_refresh=True)
    end = time.time()
    elapsed = end - start
    time_lst.append(elapsed)
    influence = np.transpose(np.array([influence]))
    if influences is not None:
        influences = np.append(influences, influence, 1)
    else:
        influences = influence                                                                                                                                                                                                
    print('============time: %f'%elapsed)

# Save the computation time info.
pickle.dump(time_lst, open('output/time_inf_cifar.pkl', 'wb'))

# Save the influence values computed
np.savez('output/cifar_inf_test_%d.npz'%(t), influences=influences)
