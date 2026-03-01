## Importing libraries
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

#-----------------------
# Importing specific functions from library diffusion
import library_diffusion as libdiff

#-----------------------
## Loading the input data
energy = np.genfromtxt('../input_data/energy_bprotein')
energy_x = energy[:,0]
energy_y = energy[:,1]

STEPS = 10000000 ## recomended min is 10**8
dt = 0.01
D = 1

#-----------------------
## Curve fitting
y_energy = savgol_filter(energy_y, 11, 3)

results_energy = libdiff.select_best_model(energy_x, y_energy)

best_energy = results_energy[0]

y_fit_energy = best_energy["func"](energy_x, *best_energy["params"])
y_fit_d_energy = best_energy["dfunc"](energy_x, *best_energy["params"])

#-----------------------
## Interpolation
F_interp  = interp1d(energy_x, y_fit_energy, kind='cubic', fill_value="extrapolate")
Fp_interp = interp1d(energy_x, y_fit_d_energy, kind='cubic', fill_value="extrapolate")

F_itp  = F_interp(energy_x) 
Fp_itp = Fp_interp(energy_x) 

V_itp = - D*Fp_itp

#-----------------------
## Trajectory calculation
Q_min = min(energy_x)
Q_max = max(energy_x)
X = 0.5*(Q_min + Q_max)  # start in middle

Q = []
T = []

for i in range(1, STEPS + 1):
    Fp = float(Fp_interp(X))

    v = - D*Fp   # drift
    
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
FQ =  np.stack((energy_x, F_itp), axis=-1)
VQ =  np.stack((energy_x, V_itp), axis=-1)

np.savetxt("TRAJECTORY.dat", traj, fmt="%12.6f")
np.savetxt("FQ.dat", FQ, fmt="%12.6f")
np.savetxt("VQ.dat", VQ, fmt="%12.6f")
