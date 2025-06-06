#!/usr/bin/env python3
""""module of agglomerative clustering"""
import numpy as np
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """that perfrms agglomerative clustering on a dataset:

    X: a numpy.ndarray of shape (n, d) containing the dataset
    dist: the maximum cophenetic distance all clusters
    
    Perfrms agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color
    The only imports you are allowed to use are:
    
    import scipy.cluster.hierarchy
    import matplotlib.pyplot as plt
Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
indices each data point"""
