## Importing libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#-----------------------
## Instructions: This code generates diffusive trajectories that represent a folding of a protein. 

## Documentation: https://numpy.org/doc/stable/reference/index.html

#-----------------------

# Importing specific functions from library diffusion

import library_diffusion as libdiff

#-----------------------
## Loading the input data
energy = np.genfromtxt('Free_energy_teste-Q-run-f50-T140.dat.dat')
energy_x = energy[:,0]
energy_y = energy[:,1]

data = np.genfromtxt('DQteste-Q-run-f50-T140.dat.2.6.1.dat')
data_x = data[:,0]
data_y = data[:,1]

## others parametres 

coef_init = np.polyfit(energy_x, energy_y, 4) 
A0 = (data_y.max()-data_y.min()) * (data_x.max()-data_x.min())
loc0 = data_x[np.argmax(data_y)]
beta0 = (data_x.max()-data_x.min())/8
C0 = data_y.min()

STEPS = 1000000
dt = 0.01
bounds = 100
grids = 100000
width = bounds*1.000000/grids

p0_pol = coef_init[::-1]  
p0_g = [A0, loc0, beta0, C0]

popp, pcop = curve_fit(libdiff.quartic_model, energy_x, energy_y, p0=p0_pol, maxfev=10000)

popg, pcog = curve_fit(libdiff.gumbel_model, data_x, data_y, p0=p0_g, maxfev=10000, bounds=([0, data.min(), 1e-6, -np.inf],
                                                     [np.inf, data.max(), np.inf, np.inf]))                                                  

#-----------------------
## Surface calculation 
fit_free = libdiff.quartic_model(energy_x, *popp)
fit_free_partial = libdiff.partial_qm(energy_x, popp[0], popp[1], popp[2], popp[3])

fit_DQ = libdiff.gumbel_model(data_x, *popg)  
fit_DQ_partial = libdiff.partial_gb(data_x, popg[0], popg[1], popg[2])

DQ = np.asarray(fit_DQ)
DQpartial = np.asarray(fit_DQ_partial)

VQ = []
dim = len(DQ)

for i in range(int(dim)):
    VQ.append((DQpartial[0+i]-DQ[0+i]*fit_free_partial[1+i]))

VQ = np.asarray(VQ)

# total =  np.stack((x, FX, VX, DX), axis=-1)
# np.savetxt("SURFACE_SIN", VQ, fmt="%10.6f")

#-----------------------
## Trajectory calculation

xgrid = np.linspace(min(energy_x.min(), data_x.min()),
                    max(energy_x.max(), data_x.max()), grids)

X = xgrid[len(xgrid)//2]

Q = []
T = []

for i in range(1, STEPS + 1):
    # find index for current position
    J = int((X - xgrid[0]) / width)
    if J < 0 or J >= len(xgrid):
        # if particle leaves domain, reflect it back
        J = np.clip(J, 0, len(xgrid)-1)
        X = xgrid[J]

    v = VQ[J]
    D = DQ[J]
    DP = fit_DQ_partial[J]

    # Langevin update
    X += (DP - D * v) * dt + libdiff.gaussian(D, dt)

    if i % 100 == 0:  # save every 100 steps
        Q.append(X)
        T.append(i * dt)

Q = np.asarray(Q)
T = np.asarray(T)

total = np.stack((T, Q), axis=-1)
np.savetxt("TRAJECTORY_SIN", total, fmt="%10.6f")


# invt =  np.stack((Q, T), axis=-1)
# np.savetxt("TRAJECTORY_SIN", invt, fmt="%5.2f")

#-----------------------
