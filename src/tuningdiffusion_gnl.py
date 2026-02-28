## Importing libraries
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

#-----------------------
# Importing specific functions from library diffusion
import library_diffusion as libdiff

#-----------------------
## Loading the input data
energy = np.genfromtxt('../input_data/Free_energy_contacts-extended-short.dat.dat')
energy_x = energy[:,0]
energy_y = energy[:,1]

coef = np.genfromtxt('../input_data/DQcontacts-extended-short.dat.2.6.1.dat')
coef_x = coef[:,0]
coef_y = coef[:,1]

STEPS = 10000000 ## recomended min is 10**8
dt = 0.01

#-----------------------
## Curve fitting
y_energy = savgol_filter(energy_y, 11, 3)
y_coef = savgol_filter(coef_y, 11, 3)

results_energy = libdiff.select_best_model(energy_x, y_energy)
results_coef = libdiff.select_best_model(coef_x, y_coef)

# print(results_energy)
# print(results_coef)

best_energy = results_energy[0]
best_coef = results_coef[0]

y_fit_energy = best_energy["func"](energy_x, *best_energy["params"])
y_fit_d_energy = best_energy["dfunc"](energy_x, *best_energy["params"])

y_fit_coef = best_coef["func"](coef_x, *best_coef["params"])
y_fit_d_coef = best_coef["dfunc"](coef_x, *best_coef["params"])

#-----------------------
## Interpolation
F_interp  = interp1d(energy_x, y_fit_energy, kind='cubic', fill_value="extrapolate")
Fp_interp = interp1d(energy_x, y_fit_d_energy, kind='cubic', fill_value="extrapolate")
D_interp  = interp1d(coef_x, y_fit_coef, kind='cubic', fill_value="extrapolate")
Dp_interp = interp1d(coef_x, y_fit_d_coef, kind='cubic', fill_value="extrapolate")

F_itp  = F_interp(energy_x) 
Fp_itp = Fp_interp(coef_x) 
D_itp  = D_interp(coef_x) 
Dp_itp = Dp_interp(coef_x) 

V_itp = Dp_itp - D_itp*Fp_itp

#-----------------------
## Trajectory calculation
Q_min = min(energy_x.min(), coef_x.min())
Q_max = max(energy_x.max(), coef_x.max())
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
DQ =  np.stack((coef_x, D_itp), axis=-1)
FQ =  np.stack((energy_x, F_itp), axis=-1)
VQ =  np.stack((coef_x, V_itp), axis=-1)

np.savetxt("TRAJECTORY.dat", traj, fmt="%12.6f")
np.savetxt("DQ.dat", DQ, fmt="%12.6f")
np.savetxt("FQ.dat", FQ, fmt="%12.6f")
np.savetxt("VQ.dat", VQ, fmt="%12.6f")
