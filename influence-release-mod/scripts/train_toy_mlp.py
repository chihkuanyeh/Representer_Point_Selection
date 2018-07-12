"""
Compute influence function values for a synthetic toy dataset using multi-layer perceptron.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf
import sys
sys.path.append('../')

import influence.experiments as experiments
from influence.toy_mlp import MLP

from load_toy import load_toy
import os

# Load toy2d data
data_sets = load_toy(from_file=True)
x_train = data_sets.train.x
x_test = data_sets.test.x
print(x_train.shape, x_test.shape)

# Define the model
num_classes = 2
input_dim = 2
weight_decay = 0.001
batch_size = 10
num_hidden_units = 256

initial_learning_rate = 0.001 
decay_epochs = [1e10, 1e10]
model = MLP(
    input_dim=input_dim,
    num_hidden_units=num_hidden_units,
    weight_decay=weight_decay,
    decay_epochs=decay_epochs,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    mini_batch=True,
    train_dir='output', 
    log_dir='log',
    model_name='toy2d_mlp')

# Train the model
num_steps = 100000
model.train(
    num_steps=num_steps, 
    iter_to_switch_to_sgd=10000000)
iter_to_load = num_steps - 1

# Or load the existing model
model.load_checkpoint(iter_to_load)

# Obtain the network outputs to compute representer values
w, b = model.sess.run([model.weights3, model.biases3])
w = np.reshape(w, (num_hidden_units, num_classes))

train_pred= model.sess.run(model.preds, feed_dict=model.all_train_feed_dict)
test_pred = model.sess.run(model.preds, feed_dict=model.all_test_feed_dict)

intermediate_train_output = model.sess.run(model.fc2_out, feed_dict=model.all_train_feed_dict)
intermediate_test_output = model.sess.run(model.fc2_out, feed_dict=model.all_test_feed_dict)

# save relevant information
np.savez('output/toy2d_outputs.npz',
        intermediate_train_output=intermediate_train_output,
        intermediate_test_output=intermediate_test_output,
        last_train_output=train_pred,
        last_test_output=train_pred,
        weight=w,
        bias=b)

# Now compute influence values
num_train = len(model.data_sets.train.labels)
num_test = len(model.data_sets.test.labels)

influences = None

for test_idx in range(num_test):
    influence = model.get_influence_on_test_loss(
                 [test_idx],
                 np.arange(num_train),
                 force_refresh=True)
    influence = np.transpose(np.array([influence]))
    if influences is not None:
        influences = np.append(influences, influence, 1)
    else:
        influences = influence                                                                                                                                                                                                

np.savez('output/toy2d_mlp_inf.npz', influences=influences)
