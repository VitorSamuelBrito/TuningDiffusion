'This script fiting curve to Free Energy'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c ## Não

def quartic_model(x, a, b, c, d, e):
    return a*x**4+b*x**3+c*x**2+d*x+e ## Sim

def polynomial_model(x, a, b, c):
    return a*x**6+b*x**4-c*x**2 ## Não

def exponential_model(x, a, b):
    return a*np.exp(b*x) ## Não

def sum_gaussian_model(x, A, mu, sigma, B, mu2, sigma2, C):
    return (A*np.exp(-(x-mu)**2/(2*sigma**2)) +
            B*np.exp(-(x-mu2)**2/(2*sigma2**2)) + C) ## Não

def sin_model(x, a, b):
    return np.sin(a*x)+np.cos(b*x)+x ## Não

# def gaussian_model(x, a, b):
#     return a*np.exp**(b*x)

# def fourier_model(x, a, b):
#     return a*np.exp**(b*x)

x = np.genfromtxt('Free_energy_teste-Q-run-f50-T140.dat.dat')
xdata = x[:,0]
ydata = x[:,1]

popt, pcov = curve_fit(quartic_model, xdata, ydata) # p0=[50,1,1,1,275]

np.savetxt("curve_fit.dat", popt)

# plt.plot(xdata, quartic_model(xdata, *popt), color='red')
# plt.scatter(xdata, ydata, s=0.7, color='black')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()