#!/usr/bin/env python3
""" module for normal distribution class """


class Normal():
    """ class for normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ args:
                data: list of data to be used
                mean: mean of distribution
                stddev: is the standard deviation of distribution
                """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            var = 0
            for x in range(len(data)):
                var += (data[x] - self.mean) ** 2
            var = var / len(data)
            self.stddev = var ** .5