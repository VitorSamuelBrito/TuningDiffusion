## Importing libraries
import numpy as np

#-----------------------
## Instructions: This code generates diffusive trajectories that represent a folding of a protein. 
## Documentation: https://numpy.org/doc/stable/reference/index.html

#-----------------------
# Importing specific functions from library diffusion
import library_diffusion as libdiff


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
    
    V += libdiff.Vx(C, W, HEIGHT, X)
    FX += libdiff.Fx(C,W,HEIGHT, X)
    
    for l in range(int(dim)):

        V += libdiff.VG(v[l], u[l], w[l], X)
        FX += libdiff.FG(v[l], u[l], w[l], X)
    
    V += SLOPE
    FX += SLOPE*X

    DX.append(libdiff.Dxsin(D, A, X, lamb))
    DXpartial.append(libdiff.Dxsinpartial(D, A, X, lamb))
    
    # DX = libdiff.Dxsin(D, A, X, lamb)
    # DXpartial = libdiff.Dxsinpartial(D, A, X, lamb)

    VX.append(V)
    F.append(FX)
    x.append(X)
    
VX = np.asarray(VX)
FX = np.asarray(F)
x = np.asarray(x)
DX = np.asarray(DX)
DXpartial = np.asarray(DXpartial)

total =  np.stack((x, FX, VX, DX), axis=-1)
np.savetxt("SURFACE_SIN.dat", total, fmt="%12.6f")

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
    
    v = (DP-D*V)

    X += v*dt+libdiff.gaussian(D,dt)
    
    #if i % 100==0:  ## spride ## every 100 values
        #Q.append(X)
        #T.append(i*dt)
    
    Q.append(X)
    T.append(i*dt)
    
Q = np.asarray(Q)
T = np.asarray(T)

traj =  np.stack((T, Q), axis=-1)
np.savetxt("TRAJECTORY_SIN.dat", traj, fmt="%12.6f")

#-----------------------
