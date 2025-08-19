'This script fiting curve'

from scipy.optimize import curve_fit

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_model(x, a, b):
    return a*np.exp**(b*x)

# def gaussian_model(x, a, b):
#     return a*np.exp**(b*x)

# def fourier_model(x, a, b):
#     return a*np.exp**(b*x)