## Importing libraries

import numpy as np

#-----------------------
## Instructions: This code generates diffusive trajectories that represent a folding of a protein. 

## Documentation: https://numpy.org/doc/stable/reference/index.html

#-----------------------
## Function of script

def Dx(D, A, x, lamb):
    Dx = D+A*np.sin(x/lamb)
    return Dx

def Dxpartial(D, A, x, lamb):
    partial = A/lamb*np.cos(x/lamb)
    return partial

def Vx(C, W, HEIGHT, x): #ainda não defini se irei mudar o M e o D
    Vx = (-2*HEIGHT*2*(x-C)/W**2 +4*HEIGHT*(x-C)**3/W**4)
    return Vx

def Fx(C, W, HEIGHT, x): #ainda não defini se irei mudar o M e o D
    Fx = (-HEIGHT*2*(x-C)**2/W**2 +HEIGHT*(x-C)**4/W**4)
    return Fx
    
def VG(v, u, HEIGHT, x):
    VG = HEIGHT*np.exp(-(x-v)**2/u**2)*2*(v-x)/u**2 
    return VG
    
def FG(v, u, HEIGHT, x):
    FG = HEIGHT*np.exp(-(x-v)**2/u**2)
    return FG

def gaussian (D, dt):
    # sd is the rms value of the distribution.
    sd = 2*D*dt
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

#-----------------------
## Loading the input data

vl = np.genfromtxt('../input_data/data_sin.txt')#, dtype= None, delimiter= None)

#-----------------------
## Defining the variables and supplying them with data

# x = vl[0] # reaction coordinate 

D = vl[1] # this the coefficient diffusion

A = vl[2] # crest breadth

lamb =vl[3] # length waves

STEPS = vl[4] # optimal value 100.000.000.000 

dt = vl[5] # element infinitesimal of time 

basin1 = vl[6] 
basin2 = vl[7]

HEIGHT = vl[8]

SLOPE = vl[9] # responsibly for to add a slope in the function

dim = vl[10] # dimension of vetor

## others parametres 

C = (basin2+basin1)/2
W = (basin2-basin1)/2

bounds = 100
grids = 100000
width = bounds*1.000000/grids

## creating the vetores

v = np.zeros(int(dim)) # center of function gaussian
u = np.zeros(int(dim)) # width of function gaussian 
w = np.zeros(int(dim)) # height of function gaussian

for i in range(int(dim)):
    j = i*3 
    v[i] = vl[11+j] # The array takes the value of the reference plus three times ahead
    u[i] = vl[12+j]
    w[i] = vl[13+j]

#-----------------------
## Checking the values of the loaded data

for l in vl:
    print(l)

print(C, W, width, v, u, w)

#-----------------------
## Surface calculation 

VX = []
F = []
x = []
DX = []
DXpartial = []

for i in range(1, int(grids) + 1):
    X = i*width
    V = 0
    FX = 0
    
    V += Vx(C, W, HEIGHT, X)
    FX += Fx(C,W,HEIGHT, X)
    
    for l in range(int(dim)):

        V += VG(v[l], u[l], w[l], X)
        FX += FG(v[l], u[l], w[l], X)
    
    V += SLOPE
    FX += SLOPE*X

    DX.append(Dx(D, A, X, lamb))
    DXpartial.append(Dxpartial(D, A, X, lamb))
    
    # DX = Dx(D, A, X, lamb)
    # DXpartial = Dxpartial(D, A, X, lamb)

    VX.append(V)
    F.append(FX)
    x.append(X)
    
VX = np.asarray(VX)
FX = np.asarray(F)
x = np.asarray(x)
DX = np.asarray(DX)
DXpartial = np.asarray(DXpartial)

total =  np.stack((x, FX, VX, DX), axis=-1)
np.savetxt("SURFACE_SIN", total, fmt="%10.6f")

#-----------------------
## Trajectory calculation

Q = []
T = []

for i in range(1, int(STEPS) + 1):
    # You must 'correct' with '-1' because python's indexation starting on zero
    J = int(X/width) - 1

    V = VX[J]
    D = DX[J]
    DP = DXpartial[J]

    X += (DP-D*V)*dt+gaussian(D,dt)
    
    if i % 100==0:  ## spride ## every 100 values
        t = dt *i
        Q.append(X)
        T.append(t)
    
Q = np.asarray(Q)
T = np.asarray(T)

# total =  np.stack((T, Q), axis=-1)
# np.savetxt("TRAJECTORY_SIN", total, fmt="%5.2f")

invt =  np.stack((Q, T), axis=-1)
np.savetxt("TRAJECTORY_SIN", invt, fmt="%5.2f")

#-----------------------
