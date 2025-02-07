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


def comparison(value, reference):
    """
    Function to compare two vectors given a threshold and a percentage limit
    
    Parameters
    ----------
    value, reference : 1-D arrays.
    
    Returns
    -------
    out : boolean
        comparison between value and reference arrays.
    """
    # making sure both are numpy arrays
    value = np.asarray(value)
    reference = np.asarray(reference)
    # # To be used in the comparisons
    threshold = 0.0001 # (direct)
    threshold_per = 0.01 # (percentage)
    # absolute difference between value and reference
    difference = np.absolute(np.subtract(np.absolute(value), \
                                         np.absolute(reference)))
    # direct comparison between the difference and threshold
    test_direct = np.less_equal(difference, threshold).all()
    if test_direct:
        print("Passed direct comparison")
        test = test_direct
    else:
        # percent comparison
        percentage = np.divide(difference, np.absolute(reference), \
                               out=np.zeros_like(np.absolute(reference)), \
                                where=np.absolute(reference)!=0)
        # testing if direct difference when value or reference is zero is 
        # below threshold
        if np.less_equal(difference[value==0], threshold).all() and \
            np.less_equal(difference[reference==0], threshold).all():
            # excluding percentage comparisons when value is zero
            percentage = percentage[value!=0]
            test_percentage = np.less_equal(percentage, threshold_per).all()
            if test_percentage:
                print("Some values have passed only by percentage comparison")
            else:
                print("Failed in percentage comparison")
            test = test_percentage
        else:
            print("Difference on zero values are above threshold")
            test = False
    return test


## comparing each function with respective perl results
def test_Vx():
    """Function to test the Vx function from library_diffusion"""
    ## To be used in all comparisons
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
#     np.savetxt("results_vx", results, fmt="%10.6f")
#     results_data = np.genfromtxt('results_vx') ## add test
    test = comparison(results, grad24_data)    
    assert test

def test_Fx():
    """Function to test the Fx function from library_diffusion"""
    ## To be used in all comparisons
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
#     np.savetxt("results_Fx", results, fmt="%10.6f")
#     results_data = np.genfromtxt('results_Fx') ## add test
    test = comparison(results, E24_data)
    assert test

def test_VG():
    """Function to test the VG function from library_diffusion"""
    ## To be used in all comparisons
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
#     np.savetxt("results_VG", results, fmt="%10.6f")
#     results_data = np.genfromtxt('results_VG') ## add test
    test = comparison(results, gradG_data)
    assert test

def test_FG():
    """Function to test the FG function from library_diffusion"""
    ## To be used in all comparisons
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
#     np.savetxt("results_FG", results, fmt="%10.6f")
#     results_data = np.genfromtxt('results_FG') ## add test
    test = comparison(results, EG_data)
    assert test

def test_Dxsin():
    """Function to test the Dxsin function from library_diffusion"""
    ## To be used in all comparisons
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
#     np.savetxt("results_Dxsin", results, fmt="%10.6f")
#     results_data = np.genfromtxt('results_Dxsin') ## add test
    test = comparison(results, DDsin_data)
    assert test

def test_Dxsinpartial():
    """Function to test the Dxsinpartial function from library_diffusion"""
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
#     np.savetxt("results_Dxpartial", results, fmt="%10.6f")
#     results_data = np.genfromtxt('results_Dxpartial') ## add test
    test = comparison(results, DDsinslope_data)
    assert test
