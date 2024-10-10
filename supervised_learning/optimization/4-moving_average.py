#!/usr/bin/env python3
""" module for calc the moving average """


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set
        Your moving average calculation should use bias correction
    args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average
    Returns: a list containing the moving averages of data """
    moving_average = []
    w_avg = 0
    for i, x in enumerate(data):
        w_avg = beta * w_avg + (1 - beta) * x
        moving_average.append(w_avg / (1 - beta ** (i + 1)))
    return moving_average
