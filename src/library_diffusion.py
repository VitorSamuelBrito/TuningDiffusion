# coding: utf8

__author__ = "Vitor Samuel Alves de Brito"
__version__ = "0.0.1"
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


