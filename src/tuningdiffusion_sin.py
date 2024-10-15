import numpy as np

#-----------------------

#| ### Documentation: https://numpy.org/doc/stable/reference/index.html , https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html, https://docs.python.org/3/library/random.html
#| #### Instructions: This code generates diffusive trajectories that represent a folding of a protein.

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
