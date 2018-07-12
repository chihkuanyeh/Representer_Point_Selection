import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet

def generate_toy_2d(scale=100.):
    """
    Generate 2D toy dataset with large margin
    """
    X, y = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=1, cluster_std=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    X = StandardScaler().fit_transform(X)
    X[y==1]  += 6
    X[y==0] -= 6
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    X_train = X_train * scale
    X_test = X_test * scale
    np.savez('data/toy_2d.npz',
            x_train = X_train,
            x_test = X_test,
            y_train = y_train,
            y_test = y_test)
    return X_train, X_test, y_train, y_test

def load_toy(from_file=False):
    """
    Create a dataset object that could be loaded to the training scripts.
    If from_file == True, load from already saved data.
    """
    if from_file:
        data = np.load('data/toy_2d.npz')
        x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    else:
        x_train, x_test, y_train, y_test = generate_toy_2d()
    train = DataSet(x_train, y_train)
    test = DataSet(x_test, y_test)
    validation = None
    return base.Datasets(train=train, validation=validation, test=test)

