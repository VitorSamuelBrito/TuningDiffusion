'This script fiting curves to Free Energy and DQ PrP'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def polynomial_model(x, a, b, c):
    return a*x**6+b*x**4-c*x**2 ## Não funcionou

def exponential_model(x, a, b):
    return a*np.exp(b*x) ## Não funcionou

def sum_gaussian_model(x, A, mu, sigma, B, mu2, sigma2, C):
    return (A*np.exp(-(x-mu)**2/(2*sigma**2)) +
            B*np.exp(-(x-mu2)**2/(2*sigma2**2)) + C) ## Não funcionou

def gaussian_model(x, a, b):
    return a*np.exp**(b*x) ## Não

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
'Calculating fiting curves'

popx, pcov = curve_fit(quartic_model, xdata, ydata)
test_x = np.linalg.cond(pcov)
fit_free = quartic_model(xdata, *popx)
r2_free = coeficiente_test(ydata, fit_free)


## Using polinimyal third model to DQ
popp, pcop = curve_fit(third_model, qdata, data, p0=p0_pol)
test_qp = np.linalg.cond(pcop)
fit_DQ_pol = third_model(qdata, *popp)
r2_DQ_pol = coeficiente_test(data, fit_DQ_pol)

print(popp)

## Using Skew function model to DQ
popsk, pcosk = curve_fit(skew_normal_model, qdata, data, p0=p0_sn, bounds=([0, x.min(), 1e-6, -20, -np.inf],
                                                            [np.inf, x.max(), np.inf, 20, np.inf]))
test_qsk = np.linalg.cond(pcosk)
fit_DQ_sk = skew_normal_model(qdata, *popsk)
r2_DQ_sk = coeficiente_test(data, fit_DQ_sk)

popq, pcoq = curve_fit(gumbel_model, qdata, data, p0=p0_g, bounds=([0, x.min(), 1e-6, -np.inf],
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
# np.savetxt('curve_fit_free.dat', fit_free)