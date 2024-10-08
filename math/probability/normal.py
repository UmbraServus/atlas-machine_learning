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

        self.pi = 3.1415926536
        self.e = 2.7182818285
    # instance methods

    def z_score(self, x):
        """ method for calc z score
            args:
                x: the x value """
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """ method for calc x value
            args:
                z: the z score """
        x = (z * self.stddev) + self.mean
        return x

    def pdf(self, x):
        """ method for calc normal dist. probability density function 
            1 / (stdd * (2 * pi) ** .5) * e ** (-.5 * z ** 2)
            """
        z = self.z_score(x)

        pdf_normal_term1 = 1 / ( self.stddev * (2 * self.pi) ** .5)
        pdf_normal_term2 = self.e ** (-.5 * z ** 2)
        result = pdf_normal_term1 * pdf_normal_term2
        return result
    
    def erf(self, x):
        """erf calculations method"""
        term1 = x ** 3 / 3
        term2 = x ** 5 / 10
        term3 = x ** 7 / 42
        term4 = x ** 9 / 216
        result = 2 / self.pi ** .5 * (x - term1 + term2 - term3 + term4)
        return result

    def cdf(self, x):
        """method or calc norm dist culmulative dist. function
           .5 * (1 + e * (z / 2 ** .5)) """

        result = .5 * (1 + self.erf(x) * (self.z_score(x) / (2 ** .5)))
        return result
