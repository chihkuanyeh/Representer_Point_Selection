# -*- coding: utf-8 -*-
"""
Generate VGG-16 features for CIFAR10 data from a trained model.
"""

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model

from utils import *
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import tensorflow as tf
import pickle, os
import sys
sys.path.append('../')

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
from utils import *
import h5py

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras

    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    #model.load_weights('imagenet_models/vgg16_weights_th_dim_ordering_th_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generate_vgg_features(idx):
    # idx = 31 for post conv layers
    # idx 32 for after first fc layer
    # idx 34 for after second fc layer

    num_classes = 10
    batch_size = 50
    base = 0
    nb_epoch = 10
    num_test = 10000
    num_train = 50000
    channel = 3
    img_rows = 224
    img_cols = 224

    # Load our model
    K.set_learning_phase(1)
    model = vgg16_model(img_rows, img_cols, channel, num_classes)
    model.summary()
    ## NOTE where do we load the weights from?
    model.load_weights('data/fine_tune_relu_cifar.h5')
    X_train, Y_train, X_valid, Y_valid = \
            load_cifar10_data(img_rows, img_cols)
    print('Load done')
    gen_feature_model = Model(inputs=model.input, outputs=model.layers[idx].output)
    print('model_defined')

    hf = h5py.File('data/vgg_features_cifar_%d.h5'%idx, 'w')
    train_features = gen_feature_model.predict(X_train)
    print('prediction done')
    hf.create_dataset('train', data=train_features)
    print('saving done')
    test_features = gen_feature_model.predict(X_valid)
    print('prediction done')
    hf.create_dataset('test', data=test_features)
    hf.close()
    print('saving done')

if __name__ == '__main__':
    generate_vgg_features(34)
