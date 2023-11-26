#| ### Documentation: https://numpy.org/doc/stable/reference/index.html , https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html, https://docs.python.org/3/library/random.html
#| #### Instructions: This code generates diffusive trajectories that represent a folding of a protein.

import numpy as np

#-----------------------

## Loading the input data ##

vl = np.genfromtxt('../input_data/data.txt')#, dtype= None, delimiter= None)

#print(vl, type(vl))

#-----------------------

## Defining the variables and supplying them with data ##

X = vl[0]

DIFFX = vl[1]

STEPS = vl[2]

dt = vl[3]

basin1 = vl[4] 
basin2 = vl[5]

HEIGHT = vl[6]

NG = vl[7]

#print(X, DIFFX, STEPS, dt, basin1, basin2, HEIGHT, NG)

#-----------------------

## 

M=(basin2+basin1)/2
D=(basin2-basin1)/2

bounds = 100
grids = 100000
width = bounds*1.000000/grids

#print(M, D, width)

#-----------------------

## Checking the values of the loaded data ##

for l in vl:
    print(l)

#-----------------------

### 

Max = np.zeros(int(NG))
sigma = np.zeros(int(NG))
GH = np.zeros(int(NG))

for i in range(int(NG)):
    j = i*3
    Max[i] = vl[8+j] # The array takes the value of the reference plus three times ahead
    sigma[i] = vl[9+j]
    GH[i] = vl[10+j]
    
# print(Max)
# print(sigma)
# print(GH)  

#-----------------------

###

def grad24(M, D, HEIGHT, X):
    eq1 = (-2*HEIGHT*2*(X-M)/D**2 +4*HEIGHT*(X-M)**3/D**4)
    return eq1

def E24(M, D, HEIGHT, X):
    eq2 = (-HEIGHT*2*(X-M)**2/D**2 +HEIGHT*(X-M)**4/D**4)
    return eq2
    
def gradG(Max, sigma, HEIGHT, X):
    #Sg = [l**2 for l in sigma]
    eq3 = HEIGHT*np.exp(-(X-Max)**2/sigma**2)*2*(Max-X)/sigma**2 
    return eq3
    
def EG(Max, sigma, HEIGHT, X):
    #Sg = [l**2 for l in sigma]
    eq4 = HEIGHT*np.exp(-(X-Max)**2/sigma**2)
    return eq4


#print(grad24(M, D, HEIGHT, X), E24(M, D, HEIGHT, X), gradG(Max, sigma, HEIGHT, X), EG(Max, sigma, HEIGHT, X))

#-----------------------

###

def gaussian (DIFFX, dt):
    # sd is the rms value of the distribution.
    sd = 2*DIFFX*dt
    sd = np.sqrt(sd)
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
            break
    return RR

#print(gaussian(DIFFX, dt))

#-----------------------

### Surface calculation ###

FF=[]
ES =[]
Hm = []

for i in range(1, int(grids) + 1):
    H = i*width
    FX = 0
    EE = 0
    FX += grad24(M, D, HEIGHT, H)
    EE += E24(M,D,HEIGHT,H)
    
    for l in range(int(NG)):
        FX += gradG(Max[l], sigma[l], GH[l], H)
        EE += EG(Max[l], sigma[l], GH[l], H)
        
    FF.append(FX)
    ES.append(EE)
    Hm.append(H)
    
FF = np.asarray(FF)
EE = np.asarray(ES)
X = np.asarray(Hm)


# np.savetxt('SURFACE_X', X, fmt="%10.6f")
# np.savetxt('SURFACE_EE', EE, fmt="%10.6f")
# np.savetxt('SURFACE_FX', FF, fmt="%10.6f")

total =  np.stack((X, EE, FF), axis=-1)
np.savetxt("SURFACE", total, fmt="%10.6f")

#-----------------------

### Surface ###

#import matplotlib.pyplot as plt
#x = X 
#fig, ax = plt.subplots()
#ax.plot(x, FF, label = 'Surface_FX') 
#ax.plot(x, EE, label = 'Surface_EE')
#plt.xlabel('Varivel H')
#plt.ylabel('Variveis FX e EE')
#plt.xlim([None, 60])
#plt.ylim([-10, 60])
#plt.legend()
#plt.show()

#-----------------------

### Trajectory calculation ###
G = []
X = []


for i in range(1, int(STEPS) + 1):
    # You must 'correct' with '-1' because python's indexation starting on zero
    J = int(H/width) - 1
    FX =FF[J] 
    H += -DIFFX*dt*FX+gaussian(DIFFX,dt)
    
    if i % 100==0:  ## stride every 100 values
        T = dt *i
        G.append(H)
        X.append(T)
    
X = np.asarray(X)
G = np.asarray(G)


# np.savetxt('TRAJECTORY_Y', G, fmt="%5.2f" )
# np.savetxt('TRAJECTORY_X', X, fmt="%5.2f" )

total =  np.stack((G, X), axis=-1)
np.savetxt("TRAJECTORY", total, fmt="%5.2f")

#-----------------------

### Trajectory ###

#x = X 
#fig, ax = plt.subplots()
#ax.plot(x, G, label = 'Trajectory')
#plt.xlabel('Varivel T')
#plt.ylabel('Varivel G')
#plt.xlim([None, 60])
#plt.ylim([10, 60])
#plt.legend()
#plt.show()

#-----------------------

#-----------------------

