#!/usr/bin/env python3
""" module for one hot encoding """
import numpy as np
import pandas as pd


def one_hot_encode(Y, classes):
    """ method for one hot encoding - converts a numeric label
    vector into a one-hot matrix
        args:
            Y: np array w/ shape (m,) containing numerical class labels
                m: number of examples
            classes: maximum num of classes in Y
            """