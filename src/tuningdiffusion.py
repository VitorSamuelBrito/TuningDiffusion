## Importing libraries
import numpy as np

#-----------------------
# Instructions: This code generates diffusive trajectories that represent a folding of a protein. 
# Documentation: https://numpy.org/doc/stable/reference/index.html

#-----------------------
# Importing specific functions from library diffusion
import library_diffusion as libdiff

#-----------------------
## Loading the input data
vl = np.genfromtxt('../input_data/data.txt')#, dtype= None, delimiter= None)

## Defining the variables and supplying them with data
# x = vl[0] # reaction coordinate 

D = vl[1] # this the coefficient diffusion

STEPS = vl[2] # 100.000.000.000 

dt = vl[3] # element infinitesimal of time 

basin1 = vl[4] 
basin2 = vl[5]

HEIGHT = vl[6]

dim = vl[7] # dimension of vetor

## others parametres 

C=(basin2+basin1)/2
W=(basin2-basin1)/2

bounds = 100
grids = 100000
width = bounds*1.000000/grids

## creating the vetores

v = np.zeros(int(dim)) # center of function gaussian
u = np.zeros(int(dim)) # width of function gaussian 
w = np.zeros(int(dim)) # height of function gaussian

for i in range(int(dim)):
    j = i*3 
    v[i] = vl[8+j] # The array takes the value of the reference plus three times ahead
    u[i] = vl[9+j]
    w[i] = vl[10+j]

#-----------------------
## Checking the values of the loaded data

for l in vl:
    print(l)

print(C, W, width, v, u, w)    
    
#-----------------------
## Surface calculation ##

V = []
F = []
x = []

for i in range(1, int(grids) + 1):
    X = i*width
    VX = 0
    FX = 0
    
    VX += libdiff.Vx(C, W, HEIGHT, X)
    FX += libdiff.Fx(C,W,HEIGHT, X)

    for l in range(int(dim)):

        VX += libdiff.VG(v[l], u[l], w[l], X)
        FX += libdiff.FG(v[l], u[l], w[l], X)

    V.append(VX)
    F.append(FX)
    x.append(X)
    
V = np.asarray(V)
FX = np.asarray(F)
x = np.asarray(x)

total =  np.stack((x, FX, V), axis=-1)
np.savetxt("SURFACE.dat", total, fmt="%12.6f")

#-----------------------

## Trajectory calculation ###

Q = []
T = []

for i in range(1, int(STEPS) + 1):
    # You must 'correct' with '-1' because python's indexation starting on zero
    J = int(X/width) - 1

    VX = V[J]
    
    v = -D*VX
    
    X += v*dt+libdiff.gaussian(D,dt)
    
    #if i % 100==0:  ## spride ## every 100 values
        #Q.append(X)
        #T.append(i*dt)
    Q.append(X)
    T.append(i*dt)
    
Q = np.asarray(Q)
T = np.asarray(T)

traj =  np.stack((T, Q), axis=-1)
np.savetxt("TRAJECTORY.dat", traj, fmt="%12.6f")

#-----------------------

