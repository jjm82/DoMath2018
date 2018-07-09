import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm
import cv2
import os

video_name = 'Haldane-model-animation'
path = '/Users/jonathanmichala/All Documents/DOMath 2018/' + video_name


'''Parameters'''
topology = 1
defect = False
showmatrix = False

animate = False
end = 250
frames = 50
el = 90.1
title = 'nonlinear initial state with propagation'

localize = True
mu=float(14)
sig=float(10)
kindex = 16

nonlin = True
before = True
after = not before
dt = .5
kappa = -.01
iterations = 500
l2diff = 0
evaldiff = 1
constindex = 2
comparison = constindex

projection = True
printek = False

n = 30
m = 20 #both in number of cells
top = m
bottom = m-1
edgest = top

size = 2*n*m
va0 = 1
vb0 = -va0
va1 = 0
vb1 = -va1
p = np.pi / 2
t = 1
t20 = 0
t21 = .1
hn = np.zeros((2*m,2*m), complex)
h = np.zeros((size,size), complex)
center = range(m/2 - m/4, m/2 + m/4)
center = range(center[0] * 2, center[-1]*2)

kvals = np.linspace(0, 2 * np.pi, 50, True)
k = []
for kval in kvals:
    for i in range(2*m):
        k.append(kval)
e = []

def findstate(linstate0,Hk,kap,iter,eigv,ind):
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
            foundstate = newestates[:, ind]
            foundeval = newevals[ind]
            minval = 0
            #laststate = newestates[:, ind + 1]
        eig = foundeval
        plt.scatter([x],[foundeval], color='red', zorder=3, marker='.')
        state = foundstate
        minvals.append(minval)
    plt.scatter(iterX, newevalsY, zorder=1, marker='.')
    plt.subplot(212)
    plt.plot(range(iter), minvals)
    return state

#make Hn
count = 0
for k1 in kvals:
    for i in range(2*m):
        j = i

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
                hn[i,j]= va0 #because t20 = 0
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
                hn[i,j]= vb0 #because t20 = 0
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

#make H
va = va1
vb = vb1
t2 = t21
if topology == 0:
    va = -va0
    vb = -vb0
    t2 = 0
for i in range(size):
    if i%2==0:
        if defect and i/(2*m) in [3*n/4, 3*n/4 + 1] and i%(2*m) in [center[0]]:
            defects.append((i%(2*m), i/(2*m)))
            h[i,i] = 1000
        else: h[i,i]=va
        if i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%(2*m)!=1 and i%(2*m)!=(2*m)-2:
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
            defects.append((i%(2*m), i/(2*m)))
            h[i,i] = 1000
        else: h[i,i]=vb
        if i%(2*m)!=0 and i%(2*m)!=2*m-1 and i%(2*m)!=1 and i%(2*m)!=2*m-2:
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
if projection:
    ens, psins = np.linalg.eigh(h)

if showmatrix:
    plt.matshow(h.real)

if nonlin and before:
    newstate = findstate(state0,hngood,kappa,iterations,stateeval,edgest)
    state0 = newstate

state0firstrow = state0
for i in range(1,n):
    state0 = np.append(state0, np.exp(1j*statek[0]*i) * state0firstrow)

if nonlin and after:
    newstate = findstate(state0,h,kappa,iterations,stateeval,edgest)
    state0 = newstate

if localize:
    s=state0
    state0=np.zeros(size, complex)
    for i in range(size):
        state0[i]=s[i]/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*(((i/m)-mu)/sig)**2)
        #state0[i]=s[i] * np.exp(-1*(((i/m)-mu)/sig)**2)

E = state0.real
E = E.reshape(n,2*m)
X = range(2*m)
Y = range(n)
X, Y = np.meshgrid(X,Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
#ax.set_zlim(-.03,.03)
ax.azim = -.1
ax.elev = el
#ax.dist = 8
ax.plot_surface(X, Y, E, cmap=cm.RdBu, linewidth=0, antialiased=False, zorder = 2)
defectsX = [x for (x,y) in defects]
defectsY = [y for (x,y) in defects]
ax.scatter(defectsX, defectsY, [0.01]*len(defectsX), s=10, c='black', zorder = 1)
plt.savefig(path + '/img00.png')

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
            statetdt = strang(statet,ens,psins,dt,time - timestamp, kappa, num)
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
        ax = fig2.add_subplot(111, projection = '3d')
        #ax.set_zlim(-.03,.03)
        ax.azim = -.1
        ax.elev = el
        #ax.dist = 8
        ax.plot_surface(X, Y, E, cmap=cm.RdBu, linewidth=0, antialiased=False)
        plt.savefig(path + '/img' + str(num).zfill(2) + '.jpg', format='jpg')
        plt.close(fig2)

        num += 1

    images = [cv2.imread(os.path.join(path, img)) for img in os.listdir(path) if img.endswith(".jpg")]
    height,width,layers = images[0].shape

    video = cv2.VideoWriter(path + '/' + title + '.mp4',-1,15,(width,height))

    for j in range(len(images)):
        video.write(images[j])

    cv2.destroyAllWindows()
    video.release()

fig = plt.figure()
plt.scatter(k,e)
plt.scatter(statek, stateeval, color='red')
if printek:
    print 'E: ', stateeval[0]
    print 'k: ', statek[0]
    print 'E/k: ', stateeval[0] / statek[0]
plt.show()