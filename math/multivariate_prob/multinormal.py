#!/usr/bin/env python3
""" module for the class multinormal"""
import numpy as np
mean_cov = __import__('0-mean_cov').mean_cov


class MultiNormal():
    """that represents a Multivariate Normal distribution:

class constructor def __init__(self, data):
    data: numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
    
    If data is not a 2D numpy.ndarray, raise a TypeError with the 
    message data must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError with the 
    message data must contain multiple data points

    Set the public instance variables:
        mean - np.ndarr shape (d, 1) containing the mean of data
        cov - a np.ndarr shape (d, d) containing the covar mat data
You are not allowed to use the function numpy.cov"""

    def __init__(self, data):

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.data = data
        dataT = data.T
        self.mean, self.cov = mean_cov(dataT)
