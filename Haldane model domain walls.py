import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import cv2
import os

video_name = 'Haldane-model-animation'
path = '/Users/jonathanmichala/All Documents/DOMath 2018/' + video_name


'''Parameters'''
l1 = 5 #lamda_1
l2 = 5
l3 = 0
eps = .001 #epsilon

topology = 0
defect = False
showmatrix = False

animate = False
end = 60
frames = 9
el = 90.1
title = 'writeup4'
colormap = True

localize = False
mu=float(14)
sig=float(10)
kindex = 35

nonlin = False
before = False
after = not before
dt = .5
kappa = -.1
strangkappa = -.1
iterations = 50
l2diff = 0
evaldiff = 1
constindex = 2
comparison = constindex

projection = True
printek = False

n = 10
m = 10 #both in number of cells
top = m
bottom = m-1
edgest = bottom

size = 2*n*m
va0 = 1
vb0 = -va0
va1 = 0
vb1 = -va1
ppos = np.pi / 3 #phi
pneg = -ppos
t = 1
t20 = 0
t21 = .1
hn = np.zeros((2*m,2*m), complex)
h = np.zeros((size,size), complex)
center = range(m/2 - m/4, m/2 + m/4)
center = range(center[0] * 2, center[-1]*2)

kvals = np.linspace(0, 2 * np.pi, 100, True)
k = []
for kval in kvals:
    for i in range(2*m):
        k.append(kval)
e = []

def cell2ind(M,m,n,a):
    return 2*M*n + 2*m + a

def findstate(linstate0,Hk,kap,iter,eigv,ind,dim=1):
    state = np.copy(linstate0)
    eig = eigv
    minvals = []
    iterX = []
    newevalsY = []
    plt.subplot(211)
    for x in range(iter):
        Hknew = Hk
        for i in range(len(linstate0)):
            Hknew[i,i] += kap * np.vdot(state[i],state[i])
        
        if dim == 2:
            newevals, newestates = eigsh(Hknew, k=2, which='SM')
        else:
            newevals, newestates = np.linalg.eigh(Hknew)

        for neweval in newevals:
            iterX.append(x)
            newevalsY.append(neweval)
        
        if comparison == evaldiff:
            minval = abs(newevals[0] - eig)
            for i in range(len(newevals)):
                diff = abs(newevals[i] - eig)
                if diff <= minval:
                    minval = diff
                    foundstate = newestates[:, i]
                    foundeval = newevals[i]
                    if x == iter - 1:
                        foundstate = newestates[:, i+1]
                    
        if comparison == l2diff:
            for i in range(len(newestates)):
                diff = 0
                possiblestate = newestates[:, i]
                for j in range(len(newestates[0])):
                    diff += np.vdot((possiblestate[j] - state[j]),(possiblestate[j] - state[j]))
                if i == 0:
                    minval = diff
                if diff < minval:
                    minval = diff
                    foundstate = possiblestate
                    foundeval = newevals[i]
        
        if comparison == constindex:
            if dim == 2:
                if newevals[0] > newevals[1]:
                    ind = 0
                else:
                    ind = 1
            foundstate = newestates[:, ind]
            foundeval = newevals[ind]
            minval = 0

        #compute l2 difference between (H + |psi|^2)psi and E psi
        Hkpsi2 = Hk
        for i in range(len(foundstate)):
            Hkpsi2[i,i] += kap * np.vdot(foundstate[i],foundstate[i])
        LHS = np.matmul(Hkpsi2, foundstate)
        RHS = foundeval * foundstate
        diff2 = 0
        for j in range(len(foundstate)):
            diff2 += np.vdot((LHS[j] - RHS[j]),(LHS[j] - RHS[j]))
        if diff2 < .00000000000001:
            print diff2
            break

        eig = foundeval
        plt.scatter([x],[foundeval], color='red', zorder=3, marker='.')
        state = foundstate
        minvals.append(diff2)
    plt.scatter(iterX, newevalsY, zorder=1, marker='.')
    plt.subplot(212)
    plt.plot(range(len(minvals)), minvals)
    return state

def makeH(m,n,va1,va0,t,t20,t21,ppos,edges=0,topology=0,defect=0):
    vb1 = -va1
    vb0 = -va0
    size = 2*m*n
    pneg = -ppos
    center = range((m/2 - m/4)*2, (m/2 + m/4)*2)
    h = np.zeros((size,size), complex)
    defects = []

    #make H
    #we count with priority on m
    #so for ordered pair (m,n) we fill a vector: [(1,1),(2,1),...,(1,2),(2,2),...]
    #i/(2*m) = n and i%(2*m) = m, more or less
    va = va1
    vb = vb1
    t2 = t21
    p = ppos
    if topology == 0:
        va = -va0
        vb = -vb0
        t2 = 0
    for i in range(size):
        if i%2==0:
            if defect and i/(2*m) in [3*n/4, 3*n/4 + 1] and i%(2*m) in [center[0]]:
                print 'a:', i/(2*m)
                print 'b:', i%(2*m)
                defects.append((i%(2*m), i/(2*m)))
                h[i,i] = 1000
            else: h[i,i]=va
            if i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%(2*m)!=1 and i%(2*m)!=(2*m)-2:
                if edges:
                    if i-2*m+1 in range(size): h[i,(i-2*m+1)]=t
                    if i-1 in range(size): h[i,(i-1)]=t
                    if i+1 in range(size): h[i,(i+1)]=t
                    if i+2*m in range(size): h[i,(i+2*m)]=t2*np.exp(1j*p)
                    if i-2*m+2 in range(size): h[i,(i-2*m+2)]=t2*np.exp(1j*p)
                    if i-2 in range(size): h[i,(i-2)]=t2*np.exp(1j*p)
                    if i-2*m in range(size): h[i,(i-2*m)]=t2*np.exp(-1j*p)
                    if i+2*m-2 in range(size): h[i,(i+2*m-2)]=t2*np.exp(-1j*p)
                    if i+2 in range(size): h[i,(i+2)]=t2*np.exp(-1j*p)
                else:
                    h[i,(i-2*m+1)%size]=t
                    h[i,(i-1)%size]=t
                    h[i,(i+1)%size]=t
                    h[i,(i+2*m)%size]=t2*np.exp(1j*p)
                    h[i,(i-2*m+2)%size]=t2*np.exp(1j*p)
                    h[i,(i-2)%size]=t2*np.exp(1j*p)
                    h[i,(i-2*m)%size]=t2*np.exp(-1j*p)
                    h[i,(i+2*m-2)%size]=t2*np.exp(-1j*p)
                    h[i,(i+2)%size]=t2*np.exp(-1j*p)

        else:
            if defect and i/(2*m) in [3*n/4] and i%(2*m) in [center[1], center[3]]:
                print 'c:', i/(2*m)
                print 'd:', i%(2*m)
                defects.append((i%(2*m), i/(2*m)))
                h[i,i] = 1000
            else: h[i,i]=vb
            if i%(2*m)!=0 and i%(2*m)!=2*m-1 and i%(2*m)!=1 and i%(2*m)!=2*m-2:
                if edges:
                    if i+2*m-1 in range(size): h[i,(i+2*m-1)]=t
                    if i-1 in range(size): h[i,(i-1)]=t
                    if i+1 in range(size): h[i,(i+1)]=t
                    if i+2*m in range(size): h[i,(i+2*m)]=t2*np.exp(-1j*p)
                    if i-2*m+2 in range(size): h[i,(i-2*m+2)]=t2*np.exp(-1j*p)
                    if i-2 in range(size): h[i,(i-2)]=t2*np.exp(-1j*p)
                    if i-2*m in range(size): h[i,(i-2*m)]=t2*np.exp(1j*p)
                    if i+2*m-2 in range(size): h[i,(i+2*m-2)]=t2*np.exp(1j*p)
                    if i+2 in range(size): h[i,(i+2)]=t2*np.exp(1j*p)
                else:
                    h[i,(i+2*m-1)%size]=t
                    h[i,(i-1)%size]=t
                    h[i,(i+1)%size]=t
                    h[i,(i+2*m)%size]=t2*np.exp(-1j*p)
                    h[i,(i-2*m+2)%size]=t2*np.exp(-1j*p)
                    h[i,(i-2)%size]=t2*np.exp(-1j*p)
                    h[i,(i-2*m)%size]=t2*np.exp(1j*p)
                    h[i,(i+2*m-2)%size]=t2*np.exp(1j*p)
                    h[i,(i+2)%size]=t2*np.exp(1j*p)

    #continue making H
    #fill sites outside walls with chern number 0 entries
    va = va0
    vb = vb0
    t2 = t20
    p = pneg
    for i in range(size):
        if i%(2*m) not in center:
            if i%2==0:
                if h[i,i] != 1000:
                    h[i,i]=va
                if i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%(2*m)!=1 and i%(2*m)!=(2*m)-2:
                    h[i,(i-2*m+1)%size] = h[i,(i-1)%size] = h[i,(i+1)%size] = t
                    h[i,(i+2*m)%size] = h[i,(i-2*m+2)%size] = h[i,(i-2)%size] = t2*np.exp(1j*p)
                    h[i,(i-2*m)%size] = h[i,(i+2*m-2)%size] = h[i,(i+2)%size] = t2*np.exp(-1j*p)
                    #fill the same across the diagonal
                    h[(i-2*m+1)%size,i] = h[(i-1)%size,i] = h[(i+1)%size,i] = t
                    h[(i+2*m)%size,i] = h[(i-2*m+2)%size,i] = h[(i-2)%size,i] = t2*np.exp(1j*p)
                    h[(i-2*m)%size,i] = h[(i+2*m-2)%size,i] = h[(i+2)%size,i] = t2*np.exp(-1j*p)
                if i%(2*m) == 0:
                    h[i,(i-2*m+1)%size] = h[i,i+1] = h[i,i+2*m-1] = t
                    h[i,(i+2*m)%size] = h[i,i+2*m-2] = h[i,(i-2*m+2)%size] = t2*np.exp(1j*p)
                    h[i,(i-2*m)%size] = h[i,(i+4*m-2)%size] = h[i,i+2] = t2*np.exp(-1j*p)
                    #fill the same across the diagonal
                    h[(i-2*m+1)%size,i] = h[i+1,i] = h[i+2*m-1,i] = t
                    h[(i+2*m)%size,i] = h[i+2*m-2,i] = h[(i-2*m+2)%size,i] = t2*np.exp(1j*p)
                    h[(i-2*m)%size,i] = h[(i+4*m-2)%size,i] = h[i+2,i] = t2*np.exp(-1j*p)
                if i%(2*m) == (2*m)-2:
                    h[i,i-1] = h[i,(i-2*m+1)%size] = h[i,i+1] = t
                    h[i,i-2] = h[i,(i+2*m)%size] = h[i,(i-4*m+2)%size] = t2*np.exp(1j*p)
                    h[i,i-2*m+2] = h[i,(i-2*m)%size] = h[i,(i+2*m-2)%size] = t2*np.exp(-1j*p)
                    #fill the same across the diagonal
                    h[i-1,i] = h[(i-2*m+1)%size,i] = h[i+1,i] = t
                    h[i-2,i] = h[(i+2*m)%size,i] = h[(i-4*m+2)%size,i] = t2*np.exp(1j*p)
                    h[i-2*m+2,i] = h[(i-2*m)%size,i] = h[(i+2*m-2)%size,i] = t2*np.exp(-1j*p)

            else:
                if h[i,i] != 1000: 
                    h[i,i]=vb
                if i%(2*m)!=0 and i%(2*m)!=2*m-1 and i%(2*m)!=1 and i%(2*m)!=2*m-2:
                    h[i,(i+2*m-1)%size] = h[i,(i-1)%size] = h[i,(i+1)%size] = t
                    h[i,(i+2*m)%size] = h[i,(i-2*m+2)%size] = h[i,(i-2)%size] = t2*np.exp(-1j*p)
                    h[i,(i-2*m)%size] = h[i,(i+2*m-2)%size] = h[i,(i+2)%size] = t2*np.exp(1j*p)
                    #fill the same across the diagonal
                    h[(i+2*m-1)%size,i] = h[(i-1)%size,i] = h[(i+1)%size,i] = t
                    h[(i+2*m)%size,i] = h[(i-2*m+2)%size,i] = h[(i-2)%size,i] = t2*np.exp(-1j*p)
                    h[(i-2*m)%size,i] = h[(i+2*m-2)%size,i] = h[(i+2)%size,i] = t2*np.exp(1j*p)
                if i%(2*m) == 1:
                    h[i,i-1] = h[i,(i+2*m-1)%size] = h[i,i+1] = t
                    h[i,i+2] = h[i,(i+4*m-2)%size] = h[i,(i-2*m)%size] = t2*np.exp(1j*p)
                    h[i,i+2*m-2] = h[i,(i-2*m+2)%size] = h[i,(i+2*m)%size] = t2*np.exp(-1j*p)
                    #fill the same across the diagonal
                    h[i-1,i] = h[(i+2*m-1)%size,i] = h[i+1,i] = t
                    h[i+2,i] = h[(i+4*m-2)%size,i] = h[(i-2*m)%size,i] = t2*np.exp(1j*p)
                    h[i+2*m-2,i] = h[(i-2*m+2)%size,i] = h[(i+2*m)%size,i] = t2*np.exp(-1j*p)
                if i%(2*m) == 2*m-1:
                    h[i,i-1] = h[i,(i+2*m-1)%size] = h[i,i-2*m+1] = t
                    h[i,i-2*m+2] = h[i,(i+2*m-2)%size] = h[i,(i-2*m)%size] = t2*np.exp(1j*p)
                    h[i,i-2] = h[i,(i-4*m+2)%size] = h[i,(i+2*m)%size] = t2*np.exp(-1j*p)
                    #fill the same across the diagonal
                    h[i-1,i] = h[(i+2*m-1)%size,i] = h[i-2*m+1,i] = t
                    h[i-2*m+2,i] = h[(i+2*m-2)%size,i] = h[(i-2*m)%size,i] = t2*np.exp(1j*p)
                    h[i-2,i] = h[(i-4*m+2)%size,i] = h[(i+2*m)%size,i] = t2*np.exp(-1j*p)
    return h, defects

'''Psuedo Eigenspectra Section'''
def B(a,b,c):
    left = np.concatenate((c,a - np.multiply(1j,b)), axis=0)
    right = np.concatenate((a + np.multiply(1j,b),-c), axis=0)
    return np.concatenate((left,right),axis=1)

def inpseudo(x,y,h,l1,l2,l3,e):
    #Takes square matrices of equivalent size X,Y,H and scalars l1,l2,l3,e 
    #returns 1 if lamdas in psuedospectrum, 0 if not
    #i.e. if B(x-l1,y-l2,h-l3) has an eigenvalue < e
    for i in range(x.shape[0]):
        x[i,i] -= l1
        y[i,i] -= l2
        h[i,i] -= l3
    B0 = B(x,y,h)
    B0evals, _ = eigsh(B0, k=1, which='SM')
    return B0evals[0] #taking this line out gives bott index
    print B0evals[0]
    if abs(B0evals[0]) < e:
        return 1
    return 0

def pseudospectra(l3,eps,X,Y,h):
    l1mesh, l2mesh = np.meshgrid(range(2*m),range(n))
    l3mesh = np.zeros((n,2*m))
    for l2 in range(n):
        for l1 in range(2*m):
            l3mesh[l2,l1] = inpseudo(X,Y,h,l1,l2,l3,eps)
    plt.pcolor(l1mesh,l2mesh,l3mesh,cmap=cm.RdBu)

#make Hn
p = ppos
count = 0
for k1 in kvals:
    for i in range(2*m):
        j = i
        p = ppos
        if i%2==0:
            if i in center and topology != 0:
                hn[i,j]= va1 + t21 * (np.exp(1j*(p+k1))+np.exp(1j*(-p-k1)))
                hn[i,(j-2) % (2*m)]= t21 * (np.exp(1j*(p))+np.exp(1j*(-p+k1)))
                hn[i,(j+2) % (2*m)]= t21 * (np.exp(1j*(-p))+np.exp(1j*(p-k1)))
                hn[(i+2) % (2*m), j] = t21 * (np.exp(1j*(-p))+np.exp(1j*(p-k1)))
                hn[(i-2) % (2*m), j] = t21 * (np.exp(1j*(p))+np.exp(1j*(-p+k1)))
            elif i in center and topology == 0:
                hn[i,j]= -va0 
            else:
                p = pneg
                hn[i,j]= va0 + t20 * (np.exp(1j*(p+k1))+np.exp(1j*(-p-k1)))
                hn[i,(j-2) % (2*m)]= t20 * (np.exp(1j*(p))+np.exp(1j*(-p+k1)))
                hn[i,(j+2) % (2*m)]= t20 * (np.exp(1j*(-p))+np.exp(1j*(p-k1)))
                hn[(i+2) % (2*m), j] = t20 * (np.exp(1j*(-p))+np.exp(1j*(p-k1)))
                hn[(i-2) % (2*m), j] = t20 * (np.exp(1j*(p))+np.exp(1j*(-p+k1)))
            hn[i,(j-1) % (2*m)]= t
            hn[i,(j+1) % (2*m)]= t * (1+np.exp(1j*(-k1)))
            
        if i%2==1:
            if i in center and topology != 0:
                hn[i,j]= vb1 + t21 * (np.exp(1j*(p-k1))+np.exp(1j*(-p+k1)))
                hn[i,(j-2) % (2*m)]= t21 * (np.exp(1j*(-p))+np.exp(1j*(p+k1)))
                hn[i,(j+2) % (2*m)]= t21 * (np.exp(1j*(p))+np.exp(1j*(-p-k1)))
                hn[(i+2) % (2*m), j]= t21 * (np.exp(1j*(p))+np.exp(1j*(-p-k1)))
                hn[(i-2) % (2*m), j]= t21 * (np.exp(1j*(-p))+np.exp(1j*(p+k1)))
            elif i in center and topology == 0:
                hn[i,j]= -vb0
            else:
                p = pneg
                hn[i,j]= vb0 + t20 * (np.exp(1j*(p-k1))+np.exp(1j*(-p+k1)))
                hn[i,(j-2) % (2*m)]= t20 * (np.exp(1j*(-p))+np.exp(1j*(p+k1)))
                hn[i,(j+2) % (2*m)]= t20 * (np.exp(1j*(p))+np.exp(1j*(-p-k1)))
                hn[(i+2) % (2*m), j]= t20 * (np.exp(1j*(p))+np.exp(1j*(-p-k1)))
                hn[(i-2) % (2*m), j]= t20 * (np.exp(1j*(-p))+np.exp(1j*(p+k1)))
            hn[i,(j-1) % (2*m)]= t * (1+np.exp(1j*(k1)))
            hn[i,(j+1) % (2*m)]= t
             

    evalues, evectors = np.linalg.eigh(hn)
    if count == kindex:
        hngood = np.copy(hn)
        evalue = evalues[edgest].real
        state0 = evectors[:, edgest]
        statek = [k1]
        stateeval = [evalue]
    for i in range(len(evalues)):
        evalue = evalues[i].real
        e.append(evalue)
    count += 1

if showmatrix:
    plt.matshow(hn.real)

defects = []

#make X which along the diagonal goes 1,2,...,2m,1,2,...,2m,1,2,...
#make Y which along the diagonal goes 1,1,...,2,2,...,2m,2m,...
X = np.zeros((size,size), complex)
Y = np.zeros((size,size), complex)
for i in range(size):
    X[i,i] = i%(2*m) + 1
    Y[i,i] = i/(2*m) + 1

h, defects = makeH(m,n,va1,va0,t,t20,t21,ppos,topology=0,defect=0)

pseudospectra(0,50,X,Y,h)
'''I think we'll be able to turn down epsilon once we use the lattice
with edges but no domain walls'''

if showmatrix:
    plt.matshow(h.real)

if nonlin and before:
    newstate = findstate(state0,hngood,kappa,iterations,stateeval,edgest)
    state0 = newstate

state0firstrow = state0
for i in range(1,n):
    state0 = np.append(state0, np.exp(1j*statek[0]*i) * state0firstrow)

if nonlin and after:
    newstate = findstate(state0,h,kappa,iterations,stateeval,edgest,dim=2)
    state0 = newstate

if localize:
    s=state0
    state0=np.zeros(size, complex)
    for i in range(size):
        state0[i]=s[i]/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*(((i/m)-mu)/sig)**2)
        #state0[i]=s[i] * np.exp(-1*(((i/m)-mu)/sig)**2)

E = state0.real
E = E.reshape(n,2*m)
Xrange = range(2*m)
Yrange = range(n)
Xrange, Yrange = np.meshgrid(Xrange,Yrange)
fig = plt.figure()
if colormap:
    plt.pcolor(Yrange,Xrange,E,cmap=cm.RdBu)
    plt.text(25,35,'$t = 0.00$')
    plt.ylim(39,0)
else:
    ax = fig.add_subplot(111, projection = '3d')
    #ax.set_zlim(-.03,.03)
    ax.azim = -.1
    ax.elev = el
    #ax.dist = 8
    ax.plot_surface(Xrange, Yrange, E, cmap=cm.RdBu, linewidth=0, antialiased=False, zorder = 2)
    defectsX = [x for (x,y) in defects]
    defectsY = [y for (x,y) in defects]
    ax.scatter(defectsX, defectsY, [0.01]*len(defectsX), s=10, c='black', zorder = 1)
plt.savefig(path + '/img00.png', format='png')

def strang(psi0, ens, psins, dt, t, kappa, prin):
    statet = np.copy(psi0)
    for _ in np.linspace(0,t,int(t/dt)):

        statedt4 = [complex(0)]*len(psins[0])
        for i in range(len(ens)):
            statedt4 += np.exp(-1j*ens[i]*dt/4) * np.vdot(psins[:, i],statet) * psins[:, i]
        
        state3dt4 = [complex(0)]*len(psins[0])
        for i in range(len(state3dt4)):
            state3dt4[i] = np.exp(-1j*kappa*dt*np.vdot(statedt4[i],statedt4[i])/2) * statedt4[i]
        
        statedt = [complex(0)]*len(psins[0])
        for i in range(len(ens)):
            statedt += np.exp(-1j*ens[i]*dt/4) * np.vdot(psins[:, i],state3dt4) * psins[:, i]
        
        statet = np.copy(statedt)
    
    return np.asarray(statet)

if animate:
    num = 1
    statet = np.copy(state0)
    timestamp = 0
    for time in np.linspace(0,end,frames):
        if nonlin:
            statetdt = strang(statet,ens,psins,dt,time - timestamp, strangkappa, num)
            timestamp = time
            statet = np.copy(statetdt)
        elif projection:
            statet = [complex(0)]*len(psins[0])
            for i in range(len(ens)):
                statet += np.exp(-1j*ens[i]*time) * np.vdot(psins[:, i],state0) * psins[:, i]
        else:
            statet = np.matmul(expm(-1j*time*h), state0)
        E = statet.real
        E = E.reshape(n,2*m)
        fig2 = plt.figure()
        if colormap:
            plt.pcolor(Yrange,Xrange,E,cmap=cm.RdBu)
            plt.text(25,35,'$t = %03.2f$' % time)
            plt.ylim(39,0)
        else:
            ax = fig2.add_subplot(111, projection = '3d')
            #ax.set_zlim(-.03,.03)
            ax.azim = -.1
            ax.elev = el
            #ax.dist = 8
            ax.plot_surface(Xrange, Yrange, E, cmap=cm.RdBu, linewidth=0, antialiased=False)
        plt.savefig(path + '/img' + str(num).zfill(2) + '.png', format='png')
        plt.close(fig2)

        num += 1

    images = [cv2.imread(os.path.join(path, img)) for img in os.listdir(path) if img.endswith(".png")]
    height,width,layers = images[0].shape

    video = cv2.VideoWriter(path + '/' + title + '.mp4',-1,15,(width,height))

    for j in range(len(images)):
        video.write(images[j])

    cv2.destroyAllWindows()
    video.release()

fig = plt.figure()
plt.title('Eigenvalues of (-1,1,-1) Case')
plt.xlabel('k (wavenumber)')
plt.ylabel('Eigenvalue / Energy')
plt.scatter(k,e)
plt.scatter(statek, stateeval, color='red')

if printek:
    print 'E: ', stateeval[0]
    print 'k: ', statek[0]
    print 'E/k: ', stateeval[0] / statek[0]
plt.show()