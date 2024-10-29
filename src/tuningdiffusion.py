import numpy as np

#| ### Documentation: https://numpy.org/doc/stable/reference/index.html , https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html, https://docs.python.org/3/library/random.html
#| #### Instructions: This code generates diffusive trajectories that represent a folding of a protein.

#-----------------------
## Function of script

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

#-----------------------
### Surface calculation 

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

# total =  np.stack((X, EE, FF), axis=-1)
# np.savetxt("SURFACE", total, fmt="%10.6f")

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

# total =  np.stack((G, X), axis=-1)
# np.savetxt("TRAJECTORY", total, fmt="%5.2f")

#-----------------------

