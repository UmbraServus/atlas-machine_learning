#!/usr/bin/env python3
""" module for using matplotlib to plot a line graph """
import numpy as np
import matplotlib.pyplot as plt


def line():
    """ method for plotting a simple line graph """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, len(y) - 1)
    plt.plot(y, 'r-')
    plt.show
