import numpy as np

#-----------------------

#| ### Documentation: https://numpy.org/doc/stable/reference/index.html , https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html, https://docs.python.org/3/library/random.html
#| #### Instructions: This code generates diffusive trajectories that represent a folding of a protein.

#-----------------------
## functions of script 

def DD(DIFFX, SINM, X, SINF):
    eqd = DIFFX+SINM*np.sin(X/SINF)
    return eqd

def DDslope(DIFFX, SINM, X, SINF):
    eqs = SINM/SINF*np.cos(X/SINF)
    return eqs

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

def gaussian (DIFFX, dt):
    # sd is the rms value of the distribution.
    sd = np.sqrt(2*DIFFX*dt)
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

#-----------------------
## Loading the input data

vl = np.genfromtxt('../input_data/data_sin.txt')#, dtype= None, delimiter= None)

#-----------------------
## Defining the variables and supplying them with data

X = vl[0]

DIFFX = vl[1]

SINM = vl[2] ##  um2/s

SINF=vl[3] ##  um2/s

STEPS = vl[4] ## 100.000.000.000 ##

dt = vl[5]

basin1 = vl[6] 
basin2 = vl[7]

HEIGHT = vl[8]

SLOPE = vl[9]

NG = vl[10]

## others variables 

M=(basin2+basin1)/2
D=(basin2-basin1)/2

bounds = 100
grids = 100000
width = bounds*1.000000/grids

Max = np.zeros(int(NG))
sigma = np.zeros(int(NG))
GH = np.zeros(int(NG))

for i in range(int(NG)):
    j = i*3 
    Max[i] = vl[11+j] # The array takes the value of the reference plus three times ahead
    sigma[i] = vl[12+j]
    GH[i] = vl[13+j]
    
#-----------------------
## Checking the values of the loaded data 

for l in vl:
    print(l)

print(M, D, width, Max, sigma, GH)
#-----------------------
## Surface calculation

FF=[]
ES =[]
Hm = []
DDV = []
DDM=[]

for i in range(1, int(grids) + 1):
    H = i*width
    FX = 0
    EE = 0
    
    FX += grad24(M, D, HEIGHT, H)
    EE += E24(M,D,HEIGHT,H)
    
    for l in range(int(NG)):

        FX += gradG(Max[l], sigma[l], GH[l], H)
        EE += EG(Max[l], sigma[l], GH[l], H)
    
    EE += SLOPE*H
    FX += SLOPE
    DDv = DD(DIFFX, SINM, H, SINF)
    DDm = DDslope(DIFFX, SINM, H, SINF)
    
    FF.append(FX)
    ES.append(EE)
    Hm.append(H)
    DDV.append(DDv)
    DDM.append(DDm)
    
FF = np.asarray(FF)
EE = np.asarray(ES)
DDV = np.asarray(DDV)
X = np.asarray(Hm)

total =  np.stack((X, FF,EE, DDV), axis=-1)
#np.savetxt("SURFACE", total, fmt="%10.6f")

#-----------------------
## Trajectory calculation

G = []
X = []

for i in range(1, int(STEPS) + 1):
    # You must 'correct' with '-1' because python's indexation starting on zero
    J = int(H/width) - 1

    FX =FF[(J)]
    DX=DDV[(J)]
    Dslope=DDM[(J)]

    H += (Dslope-DX*FX)*dt+gaussian(DX,dt);
    
    if i % 100==0:  ## spride ## every 100 values
        T = dt *i
        G.append(H)
        X.append(T)
    
X = np.asarray(X)
G = np.asarray(G)

total =  np.stack((X, G), axis=-1)
#np.savetxt("TRAJECTORY", total, fmt="%5.2f")
#np.savetxt("trajectory_file", G, fmt="%5.2f")

#-----------------------
