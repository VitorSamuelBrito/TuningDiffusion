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
energy = np.genfromtxt('../input_data/Free_energy_teste-Q-run-f50-T140.dat')
energy_x = energy[:,0]
energy_y = energy[:,1]

data = np.genfromtxt('../input_data/DQteste-Q-run-f50-T140.dat.2.6.1.dat')
data_x = data[:,0]
data_y = data[:,1]

#-----------------------
## Initial parameters
coef_init_dq = np.polyfit(data_x, data_y, 3) 

x0 = 0.5 * (energy_x.max() + energy_x.min())
s  = 0.5 * (energy_x.max() - energy_x.min()) # Cálculo do centro e escala

STEPS = 10000000 ## recomended min is 10**8
dt = 0.01
  
p0_pol_dq = coef_init_dq[::-1] 
p0_make_free = np.zeros(7)                                   

#-----------------------
## Curve fitting
make_model = libdiff.make_polynomial_model(x0, s) # Cria o modelo já com x0 e s internamente
make_deriv = libdiff.make_deriv(x0, s) # Cria o modelo já com x0 e s internamente

popp, pcop = curve_fit(make_model, energy_x, energy_y, p0=p0_make_free, maxfev=1000000)
popt, pcot = curve_fit(libdiff.third_model, data_x, data_y, p0=p0_pol_dq, maxfev=1000000)

fit_free = make_model(energy_x, *popp)
fit_free_partial = make_deriv(energy_x, *popp)

fit_DQ = libdiff.third_model(data_x, *popt)
fit_DQ_partial = libdiff.third_deriv(data_x, *popt)

#-----------------------
## Interpolation
F_interp  = interp1d(energy_x, fit_free, kind='cubic', fill_value="extrapolate")
Fp_interp = interp1d(energy_x, fit_free_partial, kind='cubic', fill_value="extrapolate")
D_interp  = interp1d(data_x, fit_DQ, kind='cubic', fill_value="extrapolate")
Dp_interp = interp1d(data_x, fit_DQ_partial, kind='cubic', fill_value="extrapolate")

F_itp  = F_interp(energy_x) 
Fp_itp = Fp_interp(data_x) 
D_itp  = D_interp(data_x) 
Dp_itp = Dp_interp(data_x) 

V_itp = Dp_itp - D_itp*Fp_itp

#-----------------------
## Trajectory calculation
Q_min = min(energy_x.min(), data_x.min())
Q_max = max(energy_x.max(), data_x.max())
X = 0.5*(Q_min + Q_max)  # start in middle

Q = []
T = []

for i in range(1, STEPS + 1):
    
    D  = float(D_interp(X))
    Dp = float(Dp_interp(X))
    Fp = float(Fp_interp(X))

    v = Dp - D*Fp   # drift
    
    X += v*dt + libdiff.gaussian(D, dt)

    #if i % 100 == 0:  # save every 100 steps
        #Q.append(X)
        #T.append(i * dt)
    Q.append(X)
    T.append(i*dt)

Q = np.asarray(Q)
T = np.asarray(T)

# Save datas 
traj = np.stack((T, Q), axis=-1)
# DQ =  np.stack((data_x, D_itp), axis=-1)
# FQ =  np.stack((energy_x, F_itp), axis=-1)
# VQ =  np.stack((data_x, V_itp), axis=-1)

np.savetxt("TRAJECTORY_PrP.dat", traj, fmt="%12.6f")
