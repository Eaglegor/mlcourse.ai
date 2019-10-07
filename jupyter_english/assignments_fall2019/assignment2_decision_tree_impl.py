#%%

import numpy as np

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression, load_digits, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

RANDOM_STATE = 17

#%%
def entropy(y):
    elements, counts = np.unique(y, return_counts = True)
    counts = - (counts/y.size) * np.log2(counts/y.size)
    return np.sum(counts)


def gini(y):
    elements, counts = np.unique(y, return_counts = True)
    counts = (counts/y.size) * (counts/y.size)
    return 1 - np.sum(counts)

def variance(y):
    return np.std(y);

def mad_median(y):
    med = np.median(y)
    return np.mean(np.abs(y - med))

#%%
class Node():
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right

#%%
class DecisionTree(BaseEstimator):
    def __init__(self, max_depth = np.inf, min_samples_split=2, criterion='gini'):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


#%%

mad_median(np.array([1,1,1,1,1,1]))
mad_median(np.array([1,1,1,1,1,2]))
mad_median(np.array([1,1,1,1,2,2]))
mad_median(np.array([1,1,1,2,2,2]))
