# coding: utf8

__author__ = "Vitor Samuel Alves de Brito"
__version__ = "0.0.7" # version three there is the curves fitting
__email__ = "vitorsamuelbr@gmail.com"

## Importing libraries
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
    # sd = 2*D*dt
    sd = np.sqrt(2*D*dt)
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
            # print('my RR used is: ', RR)
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
def cond_probV2(x, x0, x1, dx):
    "this function calculate the probability of distribution, the P(TP|Q) and calculate the histogram in determined dx"

    # to guarantee x0 < x1
    if x0 > x1:
        x0, x1 = x1, x0

    # Pré-calcular bins e histogramas
    xmin, xmax = np.min(x), np.max(x)
    bins = np.arange(xmin, xmax + dx, dx)
    bin_centers = bins[:-1] + dx / 2
    n_bins = len(bins) - 1

    # Frequência total por bin
    hist, _ = np.histogram(x, bins=bins)
    dist_den = hist / dx  # como dt = 1

    # Contador de transições
    trans_count = np.zeros(n_bins, dtype=float)

    s = 2  # init state (indefinido)
    tpx = [] # List to salve states

    for val in x:
        tpx.append(val)

        if s == 2:
            if val <= x0:
                s = 0
            elif val >= x1:
                s = 1

        elif val <= x0:
            if s == 1:
                # If before in state 1 salve transition
                idxs = np.digitize(tpx, bins) - 1
                valid = (idxs >= 0) & (idxs < n_bins)
                np.add.at(trans_count, idxs[valid], 1)
            s = 0
            tpx = [] # Reset list

        elif val >= x1:
            if s == 0:
                # If before in state 0 salve transition
                idxs = np.digitize(tpx, bins) - 1
                valid = (idxs >= 0) & (idxs < n_bins)
                np.add.at(trans_count, idxs[valid], 1)
            s = 1
            tpx = [] # Reset list

    # Calcula P(TP|Q) = número de transições / ocorrência total
    ptpx = np.divide(trans_count, hist, out=np.zeros_like(trans_count), where=hist > 0)

    return ptpx, bin_centers, dist_den

## curves to fiting 
def third_model(x, a, b, c, d):
    'Diffusion Coeficient'
    return a*x**3+b*x**2+c*x+d 
    
def make_polynomial_model(x0, s):
    def model(x, a, b, c, d, e, f, g):
        x_s = (x - x0) / s
        return (
            a*x_s**6
            + b*x_s**5
            + c*x_s**4
            + d*x_s**3
            + e*x_s**2
            + f*x_s
            + g
        )
    return model

## Derivatives (analytical forms)
def make_deriv(x0, s):
    def model(x, a, b, c, d, e, f, g):
        x_s = (x - x0) / s
        k = 1/s
        return (1/s*(
            6*a*x_s**5
            + 5*b*x_s**4
            + 4*c*x_s**3
            + 3*d*x_s**2
            + 2*e*x_s
            + f
        ))
    return model

def third_deriv(x, a, b, c, d):
    return 3*a*x**2+2*b*x+c 
    
## functions models of fitting curves (analytical forms)
def make_poly_model(degree, x_data, x0=None, scale=None):

    # -------------------------------------------------
    # Definir normalização fixa (IMPORTANTE)
    # -------------------------------------------------
    if x0 is None:
        x_ref = np.mean(x_data)
    else:
        x_ref = x0

    if scale is None:
        s = np.std(x_data)
    else:
        s = scale

    # Evitar divisão por zero
    if s == 0:
        s = 1.0

    # -------------------------------------------------
    # Função explícita para curve_fit
    # -------------------------------------------------
    def model_func(x, *params):
        x_norm = (x - x_ref) / s
        y = np.zeros_like(x_norm)
        for i, p in enumerate(params):
            y += p * x_norm**i
        return y

    # -------------------------------------------------
    # Função explícita derivada para curve_fit
    # -------------------------------------------------
    def dfunc(x, *params):
        x_norm = (x - x_ref) / s
        dy = np.zeros_like(x_norm)
        for i, p in enumerate(params):
            if i > 0:
                dy += i * p * x_norm**(i-1)
        return dy / s   # regra da cadeia

    # chute inicial
    p0 = np.zeros(degree + 1)
    p0[0] = np.mean(x_data)  # termo constante inicial razoável

    info = {
        "degree": degree,
        "x_ref": x_ref,
        "scale": s
    }

    return model_func, dfunc, p0, info

def make_shifted_double_well():

    # -------------------------------------------------
    # Função explícita para curve_fit
    # -------------------------------------------------
    def func(x, a, b, x0, c):
        z = x - x0
        return a*z**4 + b*z**2 + c

    # -------------------------------------------------
    # Função explícita derivada para curve_fit
    # -------------------------------------------------
    def dfunc(x, a, b, x0, c):
        z = x - x0
        return 4*a*z**3 + 2*b*z

    return {
        "name": "Shifted Double Well",
        "func": func,
        "dfunc": dfunc,
        "p0": [1, -1, 0, 0]
    }

def make_landau():

    # -------------------------------------------------
    # Função explícita para curve_fit
    # -------------------------------------------------
    def func(x, a, b, c):
        return a*x**2 + b*x**4 + c*x**6
    
    # -------------------------------------------------
    # Função explícita derivada para curve_fit
    # -------------------------------------------------
    def dfunc(x, a, b, c):
        return 2*a*x + 4*b*x**3 + 6*c*x**5

    return {
        "name": "Landau Potential",
        "func": func,
        "dfunc": dfunc,
        "p0": [-1, 1, 1]
    }

def make_gaussian_double(x_data):

    # -------------------------------------------------
    # Função explícita para curve_fit
    # -------------------------------------------------
    def func(x, A1, mu1, s1, A2, mu2, s2, c):
        g1 = A1*np.exp(-(x-mu1)**2/(2*s1**2))
        g2 = A2*np.exp(-(x-mu2)**2/(2*s2**2))
        return g1 + g2 + c

    # -------------------------------------------------
    # Função explícita derivada para curve_fit
    # -------------------------------------------------
    def dfunc(x, A1, mu1, s1, A2, mu2, s2, c):
        g1 = A1*np.exp(-(x-mu1)**2/(2*s1**2))
        g2 = A2*np.exp(-(x-mu2)**2/(2*s2**2))
        dg1 = g1 * (-(x-mu1)/(s1**2))
        dg2 = g2 * (-(x-mu2)/(s2**2))
        return dg1 + dg2

    return {
        "name": "Gaussian Double Well",
        "func": func,
        "dfunc": dfunc,
        "p0": [1, np.mean(x_data)-2, 2,
               1, np.mean(x_data)+2, 2,
               np.min(x_data)]
    }

def make_gaussian_sum(x_data, y_data):

    # -------------------------------------------------
    # Função explícita para curve_fit
    # -------------------------------------------------
    def func(x, a1, mu1, s1, a2, mu2, s2, c):
        g1 = a1 * np.exp(-(x-mu1)**2 / (2*s1**2))
        g2 = a2 * np.exp(-(x-mu2)**2 / (2*s2**2))
        return g1 + g2 + c

    # -------------------------------------------------
    # Função explícita derivada para curve_fit
    # -------------------------------------------------
    def dfunc(x, y, a1, mu1, s1, a2, mu2, s2, c):
        g1 = a1 * np.exp(-(x-mu1)**2 / (2*s1**2))
        g2 = a2 * np.exp(-(x-mu2)**2 / (2*s2**2))

        dg1 = g1 * (-(x-mu1)/(s1**2))
        dg2 = g2 * (-(x-mu2)/(s2**2))

        return dg1 + dg2

    # chutes iniciais robustos
    x_mean = np.mean(x_data)
    x_std  = np.std(x_data)

    p0 = [
        np.max(y_data),      # a1
        x_mean - x_std/2,    # mu1
        x_std/2,             # s1
        np.max(y_data)/2,    # a2
        x_mean + x_std/2,    # mu2
        x_std/2,             # s2
        np.min(y_data)       # offset
    ]

    return {
        "name": "Gaussian Sum (2)",
        "func": func,
        "dfunc": dfunc,
        "p0": p0
    }

## metrics to metode of minimos squares 
def compute_metrics(y_true, y_pred, k):

    n = len(y_true)

    residuals = y_true - y_pred
    rss = np.sum(residuals**2)

    # evitar log(0)
    if rss <= 0:
        rss = 1e-12

    mse = rss / n
    rmse = np.sqrt(mse)

    aic = n*np.log(rss/n) + 2*k
    bic = n*np.log(rss/n) + k*np.log(n)

    SST = np.sum((y_true - np.mean(y_true))**2)

    if SST == 0:
        r_score = 0
    else:
        r_score = 1 - rss/SST

    return mse, rmse, aic, bic, r_score, residuals

## select a best model for the curve fitting of the free energy and diffusion coefficient
def select_best_model(x, y, poly_degree_range=(2, 8)):

    models = []

    for deg in range(poly_degree_range[0], poly_degree_range[1] + 1):

        poly_func, poly_dfunc, poly_p0, poly_info = make_poly_model(deg, x)

        models.append({
            "name": f"Polinômio grau {deg} (normalizado)",
            "func": poly_func,
            "dfunc": poly_dfunc,
            "p0": poly_p0,
            "degree": deg
        })

    models.append(make_shifted_double_well())
    models.append(make_landau())
    models.append(make_gaussian_double(x))
    models.append(make_gaussian_sum(x, y))

    results = []

    for model in models:
        try:
            popt, _ = curve_fit(
                model["func"],
                x,
                y,
                p0=model["p0"],
                maxfev=1000000
            )

            y_fit = model["func"](x, *popt)
            mse, rmse, aic, bic, r_score, residuals = compute_metrics(y, y_fit, len(popt))

            results.append({
                "name": model["name"],
                "func": model["func"],
                "dfunc": model["dfunc"],
                "params": popt,
                "mse": mse,
                "rmse": rmse,
                "aic": aic,
                "bic": bic,
                "R_score": r_score,
                "residuos": residuals,
                "n_params": len(popt)
            })

        except:
            continue

    results = sorted(results, key=lambda r: r["bic"]) ## order by BIC

    return results
