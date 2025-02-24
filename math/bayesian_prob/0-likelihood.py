#!/usr/bin/env python3
"""module for likelihood"""
import numpy as np


def likelihood(x, n, P):
    """method calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects:

x: number of patients that develop severe side effects
n: total number of patients observed
P: 1D np.ndarr containing the various hypothetical prob. of developing severe
side effects

If n is not a positive integer, raise a ValueError with the message n must be
a positive integer

If x is not an integer that is greater than or equal to 0, raise a ValueError
with the message x must be an integer that is greater than or equal to 0

If x is greater than n, raise a ValueError with the message x cannot be
greater than n

If P is not a 1D numpy.ndarray, raise a TypeError with the message P must
be a 1D numpy.ndarray

If any value in P is not in the range [0, 1], raise a ValueError with the
message All values in P must be in the range [0, 1]

Returns: a 1D numpy.ndarray containing the likelihood of obtaining the data,
x and n, for each probability in P, respectively"""

    # Pseudo code:
    # 1. Check if n is a positive integer
    if not isinstance(n, int) or not n > 0:
        raise ValueError("n must be a positive integer")
    # 2. Check if x is an integer and greater than or equal to 0
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    # 3. Check if x is not greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")
    # 4. Check if P is a 1D numpy.ndarray
    if isinstance(P, np.ndarray) is False or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    # 5. Check if all values in P are in the range [0, 1]
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    # 6. Calculate the likelihood for each probability in P
    binomial_coeff = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )
    likelihood = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))
    # 7. Return the likelihoods as a 1D numpy.ndarray
    return likelihood
