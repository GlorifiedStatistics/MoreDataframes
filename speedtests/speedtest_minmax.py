"""
Tests speeds of different functions that simultaneously return the min and max of a numpy array.

Copied from: https://stackoverflow.com/questions/12200580/numpy-function-for-simultaneous-max-and-min

Results show that we can just use normal numpy np.min() and np.max() and it's not too much slower
"""
import numpy as np
from moredataframes.mdfutils import check_for_numba
from speedtester import speedtest


def _numba_while(arr):
    n = arr.size
    odd = n % 2
    if not odd:
        n -= 1
    max_val = min_val = arr[0]
    i = 1
    while i < n:
        x = arr[i]
        y = arr[i + 1]
        if x > y:
            x, y = y, x
        min_val = min(x, min_val)
        max_val = max(y, max_val)
        i += 2
    if not odd:
        x = arr[n]
        min_val = min(x, min_val)
        max_val = max(x, max_val)
    return min_val, max_val


def _numba_loop(arr):
    n = arr.size
    max_val = min_val = arr[0]
    for i in range(1, n):
        item = arr[i]
        if item > max_val:
            max_val = item
        elif item < min_val:
            min_val = item
    return min_val, max_val


def numpy_min_max(arr):
    return np.min(arr), np.max(arr)


def speedtest_min_max():

    _nb_while = check_for_numba()(_numba_while)
    _nb_failed_while = check_for_numba(test_no_numba=True)(_numba_while)
    _nb_loop = check_for_numba()(_numba_loop)
    _nb_failed_loop = check_for_numba(test_no_numba=True)(_nb_loop)

    def numba_while(a):
        return _nb_while(a)
    
    def numba_failed_while(a):
        return _nb_failed_while(a)
    
    def numba_loop(a):
        return _nb_loop(a)
    
    def numba_failed_loop(a):
        return _nb_failed_loop(a)

    speed_inputs = [
        [(np.random.rand(100_000),), {}],
        [(np.random.rand(10_000_000),), {}],
        [(np.random.rand(100_000_000),), {}],
        [(np.random.rand(1_000_000_000),), {}],
    ]

    speed_input_labels = [
        '100k',
        '10m',
        '100m',
        '1b'
    ]

    funcs = [
        numpy_min_max,
        numba_while,
        #numba_failed_while,  # This one takes forever
        numba_loop,
        numba_failed_loop,
    ]

    speedtest(speed_inputs, speed_input_labels, funcs)
