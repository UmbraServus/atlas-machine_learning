#!/usr/bin/env python3
"""module containing method of K-means on a dataset"""
import sklearn.cluster


def kmeans(X, k):
    """ that perfrms K-means on a dataset:

X: a numpy.ndarray of shape (n, d) containing the dataset
k: the number of clusters
    The only you are allowed to use is sklearn.cluster
Returns: C, clss
    C: a numpy.ndarray of shape (k, d) containing the centroid means
    each cluster
    clss: a numpy.ndarray of shape (n,) containing the index of the
    cluster in C that each data point belongs to"""
