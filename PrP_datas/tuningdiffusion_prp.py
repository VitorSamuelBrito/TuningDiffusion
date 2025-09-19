## Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

#-----------------------
# Importing specific functions from library diffusion
import library_diffusion as libdiff

#-----------------------
## Loading the input data
energy = np.genfromtxt('Free_energy_teste-Q-run-f50-T140.dat')
energy_x = energy[:,0]
energy_y = energy[:,1]

data = np.genfromtxt('DQteste-Q-run-f50-T140.dat.2.6.1.dat')
data_x = data[:,0]
data_y = data[:,1]

#-----------------------
## Initial parameters
coef_init = np.polyfit(energy_x, energy_y, 4) 
A0 = (data_y.max()-data_y.min()) * (data_x.max()-data_x.min())
loc0 = data_x[np.argmax(data_y)]
beta0 = (data_x.max()-data_x.min())/8
C0 = data_y.min()

STEPS = 1000000
dt = 0.01

p0_pol = coef_init[::-1]  
p0_g = [A0, loc0, beta0, C0]                                      

#-----------------------
## Curve fitting
popp, pcop = curve_fit(libdiff.quartic_model, energy_x, energy_y, p0=p0_pol, maxfev=10000)
popg, pcog = curve_fit(
    libdiff.gumbel_model, data_x, data_y, p0=p0_g, maxfev=10000,
    bounds=([0, data_x.min(), 1e-6, -np.inf],
            [np.inf, data_x.max(), np.inf, np.inf])
)

fit_free = libdiff.quartic_model(energy_x, *popp)
fit_free_partial = libdiff.quartic_deriv(energy_x, *popp)

fit_DQ = libdiff.gumbel_model(data_x, *popg)
fit_DQ_partial = libdiff.gumbel_deriv(data_x, *popg)

#-----------------------
## Interpolation

F_interp  = interp1d(energy_x, fit_free, kind='cubic', fill_value="extrapolate")
Fp_interp = interp1d(energy_x, fit_free_partial, kind='cubic', fill_value="extrapolate")
D_interp  = interp1d(data_x, fit_DQ, kind='cubic', fill_value="extrapolate")
Dp_interp = interp1d(data_x, fit_DQ_partial, kind='cubic', fill_value="extrapolate")

#-----------------------
## Trajectory calculation
Q_min = min(energy_x.min(), data_x.min())
Q_max = max(energy_x.max(), data_x.max())
X = 0.5*(Q_min + Q_max)  # start in middle

# print(float(D_interp(X)), fit_DQ[0])

Q = []
T = []

for i in range(1, STEPS + 1):
    D  = float(D_interp(X))
    Dp = float(Dp_interp(X))
    Fp = float(Fp_interp(X))

    v = Dp - D*Fp   # drift

    X += v*dt + libdiff.gaussian(D, dt)

    if i % 100 == 0:  # save every 100 steps
        Q.append(X)
        T.append(i * dt)

Q = np.asarray(Q)
T = np.asarray(T)

# Save datas 
traj = np.stack((T, Q), axis=-1)
DQ =  np.stack((data_x, fit_DQ), axis=-1)
FQ =  np.stack((energy_x, fit_free), axis=-1)

np.savetxt("TRAJECTORY_PrP.dat", traj, fmt="%12.6f")
np.savetxt("DQ_PrP.dat", DQ, fmt="%12.6f")
np.savetxt("Free_Energy_PrP.dat", FQ, fmt="%12.6f")