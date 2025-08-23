'This script fiting curves to Free Energy and DQ PrP'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from scipy.stats import skewnorm

## ------------------------------------------ ##
'Models functions test'

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c ## Não funcionou

'Diffusion Coeficient: Funciona, mais valor de máxima verossimilhança esta auto demais'
def third_model(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d 

'Free Energy: Funciona, mais valor de máxima verossimilhança esta auto demais'
def quartic_model(x, a, b, c, d, e):
    return a*x**4+b*x**3+c*x**2+d*x+e

def duffing_model(x, a, b, c, d):
    return a*x**4+b*x**2+c*x+d ## Não funcionou

def polynomial_model(x, a, b, c, d, e, f, g):
    return a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g ## Não funcionou

def exponential_model(x, a, b):
    return a*np.exp(b*x) ## Não funcionou

def sum_gaussian_model(x, A, mu1, sigma1, B, mu2, sigma2, C):
    return (A*np.exp(-(x-mu1)**2/(2*sigma1**2)) +
            B*np.exp(-(x-mu2)**2/(2*sigma2**2)) + C) ## Não funcionou

def sub_gaussian_model(x, A, mu1, sigma1, B, mu2, sigma2, C):
    return C+(-A*np.exp(-(x-mu1)**2/(2*sigma1**2)) -
            B*np.exp(-(x-mu2)**2/(2*sigma2**2))) ## Não funcionou

def gaussian_model(x, a, b, c):
    return a*np.exp(-(x-b)**2/c**2) ## Não funcionou

'Diffusion Coeficient: Com chutes simples funciona, mais valor de máxima verossimilhança esta auto demais'
def skew_normal_model(x, A, loc, scale, alpha, C):
    return A*skewnorm.pdf(x, alpha, loc, scale) + C 

def gumbel_model(x, a, b, c, d):
    return a * np.exp(-((x-b)/c + np.exp(-(x-b)/c))) + d 

'Diffusion Coeficient: Com chutes simples funciona, sendo o melhor até o momento'
def coeficiente_test(y_obs, y_pred):
    sum_res = np.sum((y_obs - y_pred)**2)
    sum_tot = np.sum((y_obs - np.mean(y_obs))**2)
    return 1 - (sum_res / sum_tot)

## ------------------------------------------ ##
'Input datas'

x = np.genfromtxt('Free_energy_teste-Q-run-f50-T140.dat.dat')
xdata = x[:,0]
ydata = x[:,1]

q = np.genfromtxt('DQteste-Q-run-f50-T140.dat.2.6.1.dat')
qdata = q[:,0]
data = q[:,1]

## ------------------------------------------ ##
# chutes simples para o Free Energy

coef_init = np.polyfit(xdata, ydata, 4) 
p0_pol_free = coef_init[::-1]  

## ------------------------------------------ ##
# chutes simples para o DQ

A0 = (data.max()-data.min()) * (qdata.max()-qdata.min())
loc0 = qdata[np.argmax(data)]
scale0 = (qdata.max()-qdata.min())/6
beta0 = (qdata.max()-qdata.min())/8
alpha0 = 3.0
C0 = data.min()
coef_init = np.polyfit(qdata, data, 3)    

p0_pol = coef_init[::-1]  # ajuste de ordem
p0_sn = [A0, loc0, scale0, alpha0, C0]
p0_g = [A0, loc0, beta0, C0]

## ------------------------------------------ ##
'Calculating fiting curve DQ'

## Using polinimyal third model to DQ
popp, pcop = curve_fit(third_model, qdata, data, p0=p0_pol, maxfev=10000)
test_qp = np.linalg.cond(pcop)
fit_DQ_pol = third_model(qdata, *popp)
r2_DQ_pol = coeficiente_test(data, fit_DQ_pol)

## Using Skew function model to DQ
popsk, pcosk = curve_fit(skew_normal_model, qdata, data, p0=p0_sn, maxfev=10000, bounds=([0, x.min(), 1e-6, -20, -np.inf],
                                                            [np.inf, x.max(), np.inf, 20, np.inf]))
test_qsk = np.linalg.cond(pcosk)
fit_DQ_sk = skew_normal_model(qdata, *popsk)
r2_DQ_sk = coeficiente_test(data, fit_DQ_sk)

popq, pcoq = curve_fit(gumbel_model, qdata, data, p0=p0_g, maxfev=10000, bounds=([0, x.min(), 1e-6, -np.inf],
                                                     [np.inf, x.max(), np.inf, np.inf]))
test_q = np.linalg.cond(pcoq)
fit_DQ = gumbel_model(qdata, *popq)
r2_DQ = coeficiente_test(data, fit_DQ)

# plt.plot(xdata, quartic_model(xdata, *popx), color='green')
# plt.scatter(xdata, ydata, s=0.7, color='black')

plt.scatter(qdata, data, s=0.8, label='DQ_PrP', color='black')

plt.plot(qdata, gumbel_model(qdata, *popq), 'r-', label='fit_g: A0=%5.3f, loc0=%5.3f, beta0=%5.3f, C0=%5.3f, max_v=%5.3f' 
                                                    % (popq[0], popq[1], popq[2], popq[3], test_q))

plt.plot(qdata, skew_normal_model(qdata, *popsk), 'g--', label='fit_sk: A0=%5.3f, loc0=%5.3f, scale0=%5.3f, alpha0=%5.3f, C0=%5.3f, max_v=%5.3f' 
                                                    % (popsk[0], popsk[1], popsk[2], popsk[3], popsk[4], test_qsk))

plt.plot(qdata, third_model(qdata, *popp), 'bx', markeredgewidth=0.8, label='fit_pol: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, max_v=%5.3f' 
                                                    % (popp[0], popp[1], popp[2], popp[3], test_qp))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
# plt.savefig(f'Curve_fit_DQ.png', dpi=300)
# plt.close() 

# np.savetxt('curve_fit_DQ.dat', fit_DQ)

## ------------------------------------------ ##
'Calculating fiting curve Free Energy'

## Using Polinomyal model to Free Energy
popx, pcox = curve_fit(quartic_model, xdata, ydata, p0=p0_pol_free, maxfev=10000)
test_x = np.linalg.cond(pcox)
fit_free_pol = quartic_model(xdata, *popx)
r2_free_pol = coeficiente_test(ydata, fit_free_pol)

## Using sum gaussian model to Free Energy
# popd, pcod = curve_fit(sum_gaussian_model, xdata, ydata, p0=[10, 80, 20, 10, 250, 20, 5], maxfev=30000)
# test_d = np.linalg.cond(pcod)
# fit_free_d = sum_gaussian_model(xdata, *popd)
# r2_free_d = coeficiente_test(ydata, fit_free_d)

# print(test_d, r2_free_d, test_x, r2_free_pol)

plt.scatter(xdata, ydata, s=0.8, label='DQ_PrP', color='black')

plt.plot(xdata, quartic_model(xdata, *popx), 'bx', markeredgewidth=0.5, label='fit_g: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f, max_v=%5.3f' 
                                                    % (popx[0], popx[1], popx[2], popx[3], popx[4], test_x))

# plt.plot(xdata, sum_gaussian_model(xdata, *popg), 'r-', markeredgewidth=0.5, label='fit_g: A=%5.3f, mu1=%5.3f, sigma1=%5.3f, B=%5.3f, mu2=%5.3f, sigma2=%5.3f, C=%5.3f, max_v=%5.3f' 
#                                                     % (popg[0], popg[1], popg[2], popg[3], popg[4], popg[5], popg[6], test_g))
# plt.plot(xdata, sum_gaussian_model(xdata, *popd), 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# np.savetxt('curve_fit_free.dat', fit_free)