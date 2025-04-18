# coding: utf8

__author__ = "Vitor Samuel Alves de Brito"
__version__ = "0.0.2"
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
    sd = 2*D*dt
    sd = np.sqrt(sd)
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

def hist_dx(x, dx, dt=1):
    "this function calculate the histogram in determined dx"

    hist, bins = np.histogram(x, bins=np.arange(min(x), max(x) + dx, dx)) # generate the bins of acording my dx
    dist_den = hist / (dt * dx)
    bin_centers = bins[:-1] + dx / 2  # Center bins
    return hist, bin_centers, dist_den

def cond_prob(x, x0, x1, dx):
    "this function calculate the probability of distribution, the P(TP|Q)"

    hist_zh, bin_centers, dist_den = hist_dx(x, dx, dt=1)
    trans_count = np.zeros_like(hist_zh, dtype=float) #zeros_like criate a matriz with zeros of acording reference
    
    s = 2  # Init state
    tpx = []  # List to salve states
    
    # To guarantee x0 < x1
    if x0 > x1:
        x0, x1 = x1, x0
    
    for val in x:
        tpx.append(val)

        if s == 2:  # Init state
            if val <= x0:
                s = 0
            elif val >= x1:
                s = 1

        if val <= x0:
            if s == 1:  # If before in state 1 salve transition
                hist_vals, _ = np.histogram(tpx, bins=np.arange(min(x), max(x) + dx, dx))
                trans_count += hist_vals
            s = 0
            tpx = []  # Reset list

        elif val >= x1:
            if s == 0:  # If before in state 0 salve transition
                hist_vals, _ = np.histogram(tpx, bins=np.arange(min(x), max(x) + dx, dx))
                trans_count += hist_vals
            s = 1
            tpx = []  # Reset list

    # Calculate P(TP|Q) = transitions / frequency total
    ptpx = np.divide(trans_count, hist_zh, out=np.zeros_like(hist_zh, dtype=float), where=hist_zh > 0)

    return ptpx, bin_centers, dist_den
