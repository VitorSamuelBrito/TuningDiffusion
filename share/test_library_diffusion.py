#!/usr/bin/env python3
#usage pytest

#import sys
import numpy as np
import library_diffusion as libdiff

def import_file(filepath):
    """Function to easily change precision on data being imported"""
    data_type = np.longdouble
    return np.genfromtxt(filepath, dtype=data_type)


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
    threshold = 1e-18 # (direct)
    threshold_per = 1e-9 # (percentage)
    # Will ignore percentage comparison if value or reference is below 1e-100
    threshold_zero = 1e-100 
    # absolute difference between value and reference
    difference = np.absolute(np.subtract(value, reference))
    # direct comparison between the difference and threshold
    test_direct = np.less_equal(difference, threshold).all()
    # which values failed direct comparison (if any)
    idx_failed_direct = \
        np.where(np.less_equal(difference[:, 4], threshold)==False)[0]
    if test_direct:
        print("Passed direct comparison.")
        test = test_direct
    else:
        # Show where direct comparison is failing (if that happens)
        print("{} failed direct comparison at lines {}.".format(idx_failed_direct.shape[0], idx_failed_direct))
        # If none of the following tests return True, must return False
        test = False
        # Percent comparison. 
        # First, evaluate the percentage difference on values where reference is not zero. 
        percentage = np.divide(difference, np.absolute(reference), \
                               out=np.zeros_like(np.absolute(reference)), \
                                where=np.absolute(reference)!=0)
        idx_per = np.where(percentage[:, 4]>=threshold_per)[0]
        print("There are {} values above percentage threshold.".format(idx_per.shape[0]))
        # Then tests if direct difference is below threshold when either
        # correspondent value or reference is zero. Must be True to continue.
        if np.less_equal(difference[value==0], threshold).all() and \
            np.less_equal(difference[reference==0], threshold).all():
            # Excluding percentage comparisons when value or reference is close
            # to zero.
            idx_value_is_far_from_zero = \
                np.where(np.greater_equal(np.absolute(value[:,4]), \
                                       threshold_zero))[0]
            idx_reference_is_far_from_zero = \
                np.where(np.greater_equal(np.absolute(reference[:,4]), \
                                       threshold_zero))[0]
            idx_is_far_from_zero = \
                np.unique(np.append(idx_value_is_far_from_zero, \
                                            idx_reference_is_far_from_zero))
            percentage = percentage[idx_is_far_from_zero]
            value = value[idx_is_far_from_zero]
            reference = reference[idx_is_far_from_zero]
            # In this test, we have to assure comparison only in the functions' results
            test_percentage = np.less_equal(percentage[:, 4], \
                                            threshold_per).all()
            if test_percentage:
                idx_passed = np.where(np.less_equal(percentage[:, 4], \
                                                    threshold_per)==True)[0]
                print("Some values have passed only by percentage comparison in {} lines and {} values passed in the percentage comparison.".format(idx_failed_direct.shape[0], idx_passed.shape[0]))
            else:
                idx_percentage_fail = np.where(np.greater(percentage[:, 4], \
                                                          threshold_per))
                print("Failed percentage comparison which values are {}.".format(percentage[:, 4][idx_percentage_fail]))
                print("Correspondent reference in the same places are {}".format(reference[:, 4][idx_percentage_fail]))
            test = test_percentage
        else:
            print("Failed because direct difference where one of the values is zero is above threshold.")
            test = False
    return test


## comparing each function with respective perl results
def test_Vx():
    """Function to test the Vx function from library_diffusion"""
    ## To be used in all comparisons
    sequence = import_file('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from grad24 generated with perl
    grad24_data = import_file('share/grad24_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Vx(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_vx", results, fmt="%.18e ")
#     results_data = import_file('results_vx') ## add test
    test = comparison(results, grad24_data)    
    assert test


def test_Fx():
    """Function to test the Fx function from library_diffusion"""
    ## To be used in all comparisons
    sequence = import_file('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from E24 generated with perl
    E24_data = import_file('share/E24_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Fx(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_Fx", results, fmt="%.18e ")
#     results_data = import_file('results_Fx') ## add test
    test = comparison(results, E24_data)
    assert test

def test_VG():
    """Function to test the VG function from library_diffusion"""
    ## To be used in all comparisons
    sequence = import_file('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from gradG generated with perl
    gradG_data = import_file('share/gradG_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.VG(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_VG", results, fmt="%.18e ")
#     results_data = import_file('results_VG') ## add test
    test = comparison(results, gradG_data)
    assert test

def test_FG():
    """Function to test the FG function from library_diffusion"""
    ## To be used in all comparisons
    sequence = import_file('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from EG generated with perl
    EG_data = import_file('share/EG_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.FG(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_FG", results, fmt="%.18e ")
#     results_data = import_file('results_FG') ## add test
    test = comparison(results, EG_data)
    assert test

def test_Dxsin():
    """Function to test the Dxsin function from library_diffusion"""
    ## To be used in all comparisons
    sequence = import_file('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from DDsin generated with perl
    DDsin_data = import_file('share/DDsin_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Dxsin(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_Dxsin", results, fmt="%.18e ")
#     results_data = import_file('results_Dxsin') ## add test
    test = comparison(results, DDsin_data)
    assert test

def test_Dxsinpartial():
    """Function to test the Dxsinpartial function from library_diffusion"""
    ## loading the sequence used to generate data
    sequence = import_file('share/sequence.dat')
    ## generate the vector used as input
    input = cartesian([sequence, sequence, sequence, sequence])

    # loading the data from DDsinslope generated with perl
    DDsinslope_data = import_file('share/DDsinslope_test.dat')
    # calculating the results
    results = []
    for a, b, c, d in input:
        results.append([a, b, c, d, libdiff.Dxsinpartial(a, b, c, d)])
    results = np.asarray(results)
    np.savetxt("results_Dxpartial", results, fmt="%.18e ")
#     results_data = import_file('results_Dxpartial') ## add test
    test = comparison(results, DDsinslope_data)
    assert test
