#!/usr/bin/env python3
""" module containing class for poisson distribution """


class Poisson():
    """ Poisson class """

    def __init__(self, data=None, lambtha=1.):
        """ args:
                data: list data used to estimate the distribution
                lambtha: expected number of occur. in a given time frame
            """
        if lambtha <= 0:
            raise ValueError(f"lambtha must be a positive value")
        if data is None:
            self.lambtha = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

            # Instance Methods
    def pmf(self, k):
        """ method for calc. p.m.f. (pmf = e**-lambtha * lambtha**k \ k!) 
            
            args:
                k: number of successes """
        if k < 0:
            return 0
        if not isinstance(k, int):
            int(k)

        e = 2.7182818285
        exp_term = e ** -self.lambtha
        lambtha_term = self.lambtha ** k
        factorial_term = 1
        for i in range(1, k + 1):
            factorial_term *= i
        return exp_term * lambtha_term / factorial_term
