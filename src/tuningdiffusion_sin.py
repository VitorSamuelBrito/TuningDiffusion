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
