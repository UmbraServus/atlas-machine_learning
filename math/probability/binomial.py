#!/usr/bin/env python3
""" module for binomial distribution """


class Binomial():
    """ class for binomial """

    def __init__(self, data=None, n=1, p=0.5):
        """ args:
                data: list of data to be used for distribution
                n: num of bernoulli trials
                p: probability of success
                """
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            total_trials = len(data) * max(data)
            successes = sum(data)
            self.p = successes / total_trials
            self.n = round(max(data) / self.p)
            self.p = successes / self.n