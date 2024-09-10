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
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    # Instance Methods

    def factorial(self, n):
        """Compute the factorial of a non-negative integer n."""
        if n < 0:
            raise ValueError("n must be a non-negative integer")
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def pmf(self, k):
        """ method for calc. p.m.f. (pmf = e**-lambtha * lambtha**k / k!)

            args:
                k: number of successes """

        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        exp_term = e ** -self.lambtha
        lambtha_term = self.lambtha ** k
        factorial_term = self.factorial(k)
        return exp_term * lambtha_term / factorial_term

    def cdf(self, k):
        """ method for calc. c.d.f. (cdf = P(X<=x) = pmf + pmf ...)

            args:
                k: number of successes """

        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        result = 0
        for i in range(0, k + 1):
            result += self.pmf(i)
        return result
