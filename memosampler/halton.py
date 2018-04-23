"""
@author: Stephan Balduin
@version: 1.0

"""
import numpy as np
import math


def halton(num_inputs, num_samples, start_value=0):
    primes = []
    p = 1
    for _ in range(num_inputs):
        p = next_prime(p)
        primes.append(p)

    primes = np.array(primes)

    x = np.empty(shape=(num_samples, num_inputs))
    for i in range(num_samples):
        for j in range(num_inputs):
            x[i][j] = next_halton((i + 1 + start_value), primes[j])

    return x


def next_prime(p):
    """
    Find next prime number
    Do not use for large p's
    (brute force)
    """
    p += 1

    for q in range(p, 2 * p):
        for i in range(2, q):
            if q % i == 0:
                break
        else:
            return q
    return None


def next_halton(index, base):
    """ Calculate next halton number """
    result = 0.0
    f = 1.0
    i = index

    while i > 0:
        f = f / base
        result += + f * (i % base)
        i = math.floor(i / base)

    return result
