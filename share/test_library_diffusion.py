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

# # To be used in all comparisons
# threshold = 0.001
# # loading the sequence used to generate data
# sequence = np.genfromtxt('share/sequence.dat')
# # generate the vector used as input
# input = cartesian([sequence, sequence, sequence, sequence])


## comparing each function with respective perl results
def test_Vx():
    """Function to test the Vx function from library_diffusion"""
    ## To be used in all comparisons
    threshold = 0.001
    ## loading the sequence used to generate data
    sequence = np.genfromtxt('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from grad24 generated with perl
    grad24_data = np.genfromtxt('share/grad24_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Vx(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_vx", results, fmt="%10.6f")
    results_data = np.genfromtxt('results_vx') ## add test
    
    test = np.less_equal(np.absolute(np.subtract(results_data, grad24_data)), \
                         threshold).all()
    
    assert test

def test_Fx():
    """Function to test the Fx function from library_diffusion"""
    ## To be used in all comparisons
    threshold = 0.001 # init in 0.001
    ## loading the sequence used to generate data
    sequence = np.genfromtxt('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from E24 generated with perl
    E24_data = np.genfromtxt('share/E24_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Fx(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_Fx", results, fmt="%10.6f")
    results_data = np.genfromtxt('results_Fx') ## add test
    
    test = np.less_equal(np.absolute(np.subtract(results_data, E24_data)), \
                         threshold).all()
    # sub = np.absolute(np.subtract(results, E24_data))                   
    # test = np.less_equal(sub, threshold).all()

    assert test

def test_VG():
    """Function to test the VG function from library_diffusion"""
    ## To be used in all comparisons
    threshold = 0.001
    ## loading the sequence used to generate data
    sequence = np.genfromtxt('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from gradG generated with perl
    gradG_data = np.genfromtxt('share/gradG_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.VG(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_VG", results, fmt="%10.6f")
    results_data = np.genfromtxt('results_VG') ## add test
    
    test = np.less_equal(np.absolute(np.subtract(results_data, gradG_data)), \
                         threshold).all()

    assert test

def test_FG():
    """Function to test the FG function from library_diffusion"""
    ## To be used in all comparisons
    threshold = 0.001
    ## loading the sequence used to generate data
    sequence = np.genfromtxt('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from EG generated with perl
    EG_data = np.genfromtxt('share/EG_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.FG(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_FG", results, fmt="%10.6f")
    results_data = np.genfromtxt('results_FG') ## add test
    
    test = np.less_equal(np.absolute(np.subtract(results_data, EG_data)), \
                         threshold).all()

    assert test

def test_Dxsin():
    """Function to test the Dxsin function from library_diffusion"""
    ## To be used in all comparisons
    threshold = 0.001
    ## loading the sequence used to generate data
    sequence = np.genfromtxt('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from DDsin generated with perl
    DDsin_data = np.genfromtxt('share/DDsin_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Dxsin(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_Dxsin", results, fmt="%10.6f")
    results_data = np.genfromtxt('results_Dxsin') ## add test
    
    test = np.less_equal(np.absolute(np.subtract(results_data, DDsin_data)), \
                         threshold).all()

    assert test

def test_Dxsinpartial():
    """Function to test the Dxsinpartial function from library_diffusion"""
    ## To be used in all comparisons
    threshold = 0.001
    ## loading the sequence used to generate data
    sequence = np.genfromtxt('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from DDsinslope generated with perl
    DDsinslope_data = np.genfromtxt('share/DDsinslope_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Dxsinpartial(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_Dxpartial", results, fmt="%10.6f")
    results_data = np.genfromtxt('results_Dxpartial') ## add test
    
    test = np.less_equal(np.absolute(np.subtract(results_data, DDsinslope_data)), \
                         threshold).all()

    assert test
