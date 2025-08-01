import numpy as np

perl = np.genfromtxt('gaussian_D_1.dat', dtype= float)
python = np.genfromtxt('Gaussian_D_var_1.txt', dtype= float)

def test_gaussian(A, B):
    threshold = 1e-1 # (direct) to >= 1e-2 false
    threshold_per = 1e-1 # (percentage) to >= 1e-2 false
    
    sdt_A = np.std(A[:,1])
    sdt_B = np.std(B[:,1])
    
    max_A = max(A[:,1])
    max_B = max(B[:,1])
    
    diff_max = np.absolute(np.subtract(max_A, max_B))
    difference = np.absolute(np.subtract(sdt_A, sdt_B))
    
    test_max = np.less_equal(diff_max, threshold).all()
    test_direct = np.less_equal(difference, threshold).all()
    
    if test_direct and test_max:
        print("Passed direct comparison.")
        test = test_direct
    else:
        test = False
        media = (sum(A[:,1])+sum(B[:,1]))/2
        if max_A > max_B:
            pdc = (max_A-max_B)/max_A
        else:
            pdc = (max_B-max_A)/max_B
        
        percentage = np.divide(difference, np.absolute(media))
        
        test_pdc = np.less_equal(pdc, threshold_per).all()
        test_percentage = np.less_equal(percentage, threshold_per).all()
        
        if test_percentage and test_pdc:
            print("Passed percentage comparison.")
            test = test_percentage
        else:
            print("Don't passed in the tests to standard deviation.")
            test = False
            
    return test

final_test = test_gaussian(perl, python)

print(final_test)