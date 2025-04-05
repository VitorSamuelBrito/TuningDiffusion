import numpy as np

def hist_dx(x, dx, dt=1):
    hist, bins = np.histogram(x, bins=np.arange(min(x), max(x) + dx, dx)) # define os bins de acorodo com dx
    dist_den = hist / (dt * dx)
    bin_centers = bins[:-1] + dx / 2  # Centros dos bins
    return hist, bin_centers, dist_den

def cond_prob(x, x0, x1, dx):
    hist_zh, bin_centers, dist_den = hist_dx(x, dx, dt=1)
    trans_count = np.zeros_like(hist_zh, dtype=float) #zeros_like cria uma matriz com zeros do tamanho da referência
    
    s = 2  # Estado inicial indefinido
    tpx = []  # Lista para armazenar estados
    
    # Garantir que x0 < x1
    if x0 > x1:
        x0, x1 = x1, x0
    
    for val in x:
        tpx.append(val)

        if s == 2:  # Estado inicial indefinido
            if val <= x0:
                s = 0
            elif val >= x1:
                s = 1

        if val <= x0:
            if s == 1:  # Se antes estava no estado 1, registramos a transição
                hist_vals, _ = np.histogram(tpx, bins=np.arange(min(x), max(x) + dx, dx))
                trans_count += hist_vals
            s = 0
            tpx = []  # Reset da lista

        elif val >= x1:
            if s == 0:  # Se antes estava no estado 0, registramos a transição
                hist_vals, _ = np.histogram(tpx, bins=np.arange(min(x), max(x) + dx, dx))
                trans_count += hist_vals
            s = 1
            tpx = []  # Reset da lista

    # Calcular p(TP|x) = transições / frequência total
    ptpx = np.divide(trans_count, hist_zh, out=np.zeros_like(hist_zh, dtype=float), where=hist_zh > 0)

    return ptpx, bin_centers, dist_den

x = np.genfromtxt('TRAJECTORY_INV')#, dtype= None, delimiter= None)
# print(type(x))

# Parameters
x0 = 25
x1 = 35
dx = 0.5  # Ideal range: 0.1 to 0.5

ptpx_result, bins, density = cond_prob(x, x0, x1, dx)