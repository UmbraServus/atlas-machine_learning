#!/usr/bin/env python3
""" module containing class for poisson distribution """


class Poisson():
    def __init__(self, data=None, lambtha=1.):
        """ args:
                data: list data used to estimate the distribution
                lambtha: expected number of occur. in a given time frame
            """
        if data is None:
            self.data = lambtha
            self.lambtha = lambtha
        if lambtha <= 0:
            raise ValueError(f"lambtha must be a positive value")
        if data:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
