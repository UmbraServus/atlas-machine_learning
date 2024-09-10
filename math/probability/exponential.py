#!/usr/bin/env python3
""" module for exponential distribution """


class Exponential():
    """ class for exponential distribution """

    def __init__(self, data=None, lambtha=1.):
        """ args:
            data: list of data used for the distribution
            lambtha: expected num of occur. in a time frame.
            """
        if lambtha <= 0:
            raise ValueError(f"lambtha must be a positive value")
        if data is None:
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))
