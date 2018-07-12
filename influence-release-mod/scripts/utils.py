
import cv2
import numpy as np

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

nb_train_samples = 50000 # 3000 training samples
nb_valid_samples = 10000 # 100 validation samples
num_classes = 10

def load_cifar10_data(img_rows, img_cols, start=None, end=None, what_data=None):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    #print(X_train.shape)
    #print(X_valid.shape)

    if start == None or end == None and what_data == None:
        # Resize trainging images (all of them)
        if K.image_dim_ordering() == 'th':
            X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
            X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
        else:
            X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
            X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

        # Transform targets to keras compatible format
        Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
        Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    else:
        # Resize and load part of them
        if K.image_dim_ordering() == 'th':
            if what_data == 'train':
                X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[start:end,:,:,:]])
                X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
            elif what_data == 'test':
                X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
                X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[start:end,:,:,:]])
        else:
            if what_data == 'train':
                X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[start:end,:,:,:]])
                X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])
            elif what_data == 'test':
                X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
                X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[start:end,:,:,:]])

        # Transform targets to keras compatible format
        if what_data == 'train':
            Y_train = np_utils.to_categorical(Y_train[start:end], num_classes)
            Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)
        elif what_data == 'test':
            Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
            Y_valid = np_utils.to_categorical(Y_valid[start:end], num_classes)

    return X_train, Y_train, X_valid, Y_valid

def reshape2original(img, img_rows, img_cols):
    return np.array([cv2.resize(img[i].transpose(1,2,0), (img_rows, img_cols)).transpose(2,0,1) for i in range(img.shape[0])])


