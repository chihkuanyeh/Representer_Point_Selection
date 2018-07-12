"""
Code for loading CIFAR-10 data.
"""


import gzip
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
from keras.datasets import cifar10


def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.transpose(0,2,3,1)
    x_test = x_test.transpose(0,2,3,1)
    train = DataSet(x_train, y_train.flatten())
    test = DataSet(x_test, y_test.flatten())
    validation = None

    return base.Datasets(train=train, validation=validation, test=test)

def load_cifar_labels():
    (_, y_train), (_, y_test) = cifar10.load_data()
    return y_train, y_test
