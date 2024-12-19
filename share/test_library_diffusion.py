#!/usr/bin/env python3
#usage pytest

#import sys
import numpy as np
import library_diffusion as libdiff

def cartesian(arrays, out=None):
    """
    Obs. Found on internet
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

# To be used in all comparisons
threshold = 0.001
# loading the sequence used to generate data
sequence = np.genfromtxt('share/sequence.dat')
# generate the vector used as input
input = cartesian([sequence, sequence, sequence, sequence])


## comparing each function with respective perl results
def test_Dxsin():
    """Function to test the Dxsin function from library_diffusion"""
    # loading the data from DDsin generated with perl
    DDsin_data = np.genfromtxt('share/DDsin_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Dxsin(a, b, c, d)])
    results = np.asarray(results)
    test = np.less_equal(np.absolute(np.subtract(results, DDsin_data)), \
                         threshold).all()
    assert test
