"""
Compute influence function values for Animals with Attributes dataset using ResNet features.
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
from influence.awa_mlp import AWA_MLP
import pickle

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
import h5py
import time

# First create the dataset object from the VGG features
print("Loading Data...")
x_train = np.squeeze(np.load('data/train_feature_awa.npy'))
y_train = np.squeeze(np.load('data/train_output_awa.npy'))
x_test = np.squeeze(np.load('data/val_feature_awa.npy'))
y_test = np.squeeze(np.load('data/val_output_awa.npy'))

# Get labels
train_labels = np.argmax(y_train, axis=1)
test_labels = np.argmax(y_test, axis=1)
train = DataSet(x_train, train_labels)
test = DataSet(x_test, test_labels)
data_sets = base.Datasets(train=train, validation=None, test=test)

num_classes = 50
weight_decay = 0.001
batch_size = 290

initial_learning_rate = 0.00001 
decay_epochs = [10000, 20000]
input_dim = x_train.shape[1]

model = AWA_MLP(
    input_dim=input_dim,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output', 
    log_dir='log',
    model_name='awa_mlp')

num_steps = 300000
model.train(
    num_steps=num_steps, 
    iter_to_switch_to_batch=10000000,
    iter_to_switch_to_sgd=10000000)
iter_to_load = num_steps - 1

print('Training done')

# Load the trained model
model.load_checkpoint(iter_to_load)


# For AWA, randomly select 1000 test points to compute the influence function values.
num_train = len(model.data_sets.train.labels)
num_test = len(model.data_sets.test.labels)

num_select = 1000
np.random.seed(42)
test_indices = np.random.choice(num_test, num_select, replace=False)

# save the indices selected for future references
np.savez('output/idx_inf_awa_%d.npz'%num_select, idx=test_indices)

# Compute influence function values.
influences = None
time_lst = []
for test_idx in test_indices:
    start = time.time()
    influence = model.get_influence_on_test_loss(
                 [test_idx],
                 np.arange(num_train),
                 force_refresh=True)
    elapsed = time.time() - start
    time_lst.append(elapsed)
    influence = np.transpose(np.array([influence]))
    if influences is not None:
        influences = np.append(influences, influence, 1)
    else:
        influences = influence                                                                                                                                                                                                
    print('============time %f'%elapsed)

# save computation time info
pickle.dump(time_lst, open('output/time_inf_awa.pkl', 'wb'))

# Save the influence values
np.savez('output/awa_inf_test_%d.npz'%(num_select), influences=influences)
