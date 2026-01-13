# coding: utf8

__author__ = "Vitor Samuel Alves de Brito"
__version__ = "0.0.6" # version three there is the curves fiting
__email__ = "vitorsamuelbr@gmail.com"

import numpy as np

## Script functions

def Vx(C, W, HEIGHT, x): 
    Vx = (-2*HEIGHT*2*(x-C)/W**2 +4*HEIGHT*(x-C)**3/W**4)
    return Vx

def Fx(C, W, HEIGHT, x): 
    Fx = (-HEIGHT*2*(x-C)**2/W**2 +HEIGHT*(x-C)**4/W**4)
    return Fx
    
def VG(v, u, HEIGHT, x):
    VG = HEIGHT*np.exp(-(x-v)**2/u**2)*2*(v-x)/u**2 
    return VG
    
def FG(v, u, HEIGHT, x):
    FG = HEIGHT*np.exp(-(x-v)**2/u**2)
    return FG

def gaussian(D, dt):
    # sd is the rms value of the distribution.
    # sd = 2*D*dt
    sd = np.sqrt(2*D*dt)
    RR = 0 
    while True:
        M1 = np.random.random()
        M2 = np.random.random()
        M1 = 2*(M1-0.5)
        M2 = 2*(M2-0.5)
        tmp1 = M1**2 + M2**2
        if tmp1 <= 1.0 and tmp1 >= 0.0:
            tmp2 = sd*np.sqrt( -2*np.log(tmp1)/tmp1 )
            RR = M1*tmp2
            # print('my RR used is: ', RR)
            break
    return RR

# Diffusion functions for the sinusidal case
def Dxsin(D, A, x, lamb):
    Dxsin = D+A*np.sin(x/lamb)
    return Dxsin

def Dxsinpartial(D, A, x, lamb):
    partial = A/lamb*np.cos(x/lamb)
    return partial

# probability of distribution functions 
def cond_probV2(x, x0, x1, dx):
    "this function calculate the probability of distribution, the P(TP|Q) and calculate the histogram in determined dx"

    # to guarantee x0 < x1
    if x0 > x1:
        x0, x1 = x1, x0

    # Pré-calcular bins e histogramas
    xmin, xmax = np.min(x), np.max(x)
    bins = np.arange(xmin, xmax + dx, dx)
    bin_centers = bins[:-1] + dx / 2
    n_bins = len(bins) - 1

    # Frequência total por bin
    hist, _ = np.histogram(x, bins=bins)
    dist_den = hist / dx  # como dt = 1

    # Contador de transições
    trans_count = np.zeros(n_bins, dtype=float)

    s = 2  # init state (indefinido)
    tpx = [] # List to salve states

    for val in x:
        tpx.append(val)

        if s == 2:
            if val <= x0:
                s = 0
            elif val >= x1:
                s = 1

        elif val <= x0:
            if s == 1:
                # If before in state 1 salve transition
                idxs = np.digitize(tpx, bins) - 1
                valid = (idxs >= 0) & (idxs < n_bins)
                np.add.at(trans_count, idxs[valid], 1)
            s = 0
            tpx = [] # Reset list

        elif val >= x1:
            if s == 0:
                # If before in state 0 salve transition
                idxs = np.digitize(tpx, bins) - 1
                valid = (idxs >= 0) & (idxs < n_bins)
                np.add.at(trans_count, idxs[valid], 1)
            s = 1
            tpx = [] # Reset list

    # Calcula P(TP|Q) = número de transições / ocorrência total
    ptpx = np.divide(trans_count, hist, out=np.zeros_like(trans_count), where=hist > 0)

    return ptpx, bin_centers, dist_den

## curves to fiting 
def third_model(x, a, b, c, d):
    'Diffusion Coeficient'
    return a*x**3+b*x**2+c*x+d 
    
def make_polynomial_model(x0, s):
    def model(x, a, b, c, d, e, f, g):
        x_s = (x - x0) / s
        return (
            a*x_s**6
            + b*x_s**5
            + c*x_s**4
            + d*x_s**3
            + e*x_s**2
            + f*x_s
            + g
        )
    return model

## Derivatives (analytical forms)
def make_deriv(x0, s):
    def model(x, a, b, c, d, e, f, g):
        x_s = (x - x0) / s
        k = 1/s
        return (1/s*(
            6*a*x_s**5
            + 5*b*x_s**4
            + 4*c*x_s**3
            + 3*d*x_s**2
            + 2*e*x_s
            + f
        ))
    return model

def third_deriv(x, a, b, c, d):
    return 3*a*x**2+2*b*x+c 

