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
coef_init_new = np.polyfit(data_x, data_y, 3) 
# A0 = (data_y.max()-data_y.min()) * (data_x.max()-data_x.min())
# loc0 = data_x[np.argmax(data_y)]
# beta0 = (data_x.max()-data_x.min())/8
# C0 = data_y.min() ## using was in test one, but my DQ is bad visualy

STEPS = 1000000
dt = 0.01

p0_pol = coef_init[::-1]  
p0_pol_new = coef_init_new[::-1] 
# p0_g = [A0, loc0, beta0, C0]                                      

#-----------------------
## Curve fitting
popp, pcop = curve_fit(libdiff.quartic_model, energy_x, energy_y, p0=p0_pol, maxfev=10000)
popt, pcot = curve_fit(libdiff.third_model, data_x, data_y, p0=p0_pol_new, maxfev=10000)
# popg, pcog = curve_fit(
#     libdiff.gumbel_model, data_x, data_y, p0=p0_g, maxfev=10000,
#     bounds=([0, data_x.min(), 1e-6, -np.inf],
#             [np.inf, data_x.max(), np.inf, np.inf]))

fit_free = libdiff.quartic_model(energy_x, *popp)
fit_free_partial = libdiff.quartic_deriv(energy_x, *popp)

fit_DQ = libdiff.third_model(data_x, *popt)
fit_DQ_partial = libdiff.third_deriv(data_x, *popt)

#-----------------------
## Interpolation

F_interp  = interp1d(energy_x, fit_free, kind='cubic', fill_value="extrapolate")
Fp_interp = interp1d(energy_x, fit_free_partial, kind='cubic', fill_value="extrapolate")
D_interp  = interp1d(data_x, fit_DQ, kind='cubic', fill_value="extrapolate")
Dp_interp = interp1d(data_x, fit_DQ_partial, kind='cubic', fill_value="extrapolate")

V_interp = []

for k in range(1, 258):
    drift =  float(Dp_interp(k)) - float(D_interp(k))*float(F_interp(k))
    V_interp.append(drift)

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
V_interp = np.asarray(V_interp)

# Save datas 
traj = np.stack((T, Q), axis=-1)
DQ =  np.stack((data_x, fit_DQ), axis=-1)
FQ =  np.stack((energy_x, fit_free), axis=-1)
VQ =  np.stack((data_x, V_interp), axis=-1)

np.savetxt("TRAJECTORY_PrP_new_fit.dat", traj, fmt="%12.6f")
np.savetxt("DQ_PrP_new_fit.dat", DQ, fmt="%12.6f")
np.savetxt("Free_Energy_PrP_new_fit.dat", FQ, fmt="%12.6f")
np.savetxt("Drift_PrP_new_fit.dat", VQ, fmt="%12.6f")

# -------------- 
## Generanting ptpx and Hist 

# Q = np.genfromtxt('TRAJECTORY_PrP_new_fit.dat')
# x0 = 125
# x1 = 200
# dx = 10

# ptpx, bin_centers, dist_den = libdiff.cond_probV2(Q, x0, x1, dx)

# # -------------- 
# ## Ajust the curves 
# A = 1

# log_hist = -A*np.log(dist_den)

# log_hist_min = log_hist.min()
# Q_log_hist = log_hist - log_hist_min

# fit_min = fit_free.min()
# Q_fit_free = fit_free - fit_min

# energy_min = energy.min()
# Q_energy = energy - energy_min # my Q_min force init in zero

#-----------------------
##  Print grafics to save

# plt.scatter(data_x, data_y, s=0.8, label='DQ', color='black')
# plt.plot(data_x, fit_DQ, label='DQ fit', color='red')
# plt.legend()
# plt.xlabel("reaction coordinate Q")
# plt.ylabel("DQ")
# plt.title("Comparative")
# plt.show() ## pritn DQ


# plt.scatter(energy_x, Q_energy, s=0.8, label='Free Energy ajt', color='black')
# plt.plot(energy_x, Q_fit_free, label='Free Energy fit ajt', color='red')
# plt.plot(bin_centers, Q_log_hist, 'b--', label='Free Energy hist ajt')
# # plt.plot(energy_x, k_corr, 'g--', label='Free Energy hist_corr')
# plt.legend()
# plt.xlabel("reaction coordinate Q")
# plt.ylabel("FQ/kT")
# plt.title("Comparative")
# plt.show() ## print FQ

# plt.plot(data_x, V_interp, label='VQ', color='black')
# plt.legend()
# plt.xlabel("reaction coordinate Q")
# plt.ylabel("drift")
# plt.title("Drift")
# plt.show() ## pritn VQ

'- Gerar novas trajetorias com novo DQ_fit'
'- Mudar, passar a salvar DQ e FQ interpolados (teste)'
'- Salvar gráficos novos interpolados'
