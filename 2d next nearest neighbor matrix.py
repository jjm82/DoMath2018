import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm

#n and m are the number of cells in each direction and therefore size is the total # of atoms
n = 20
m = 10
size = 2*n*m

va0 = 1
vb0 = -va0
va1 = 0
vb1 = -va1

t20 = 0
t21 = .1

t = 1
p = .1

center = range(3,7) # in cells
center = range(center[0]*2, center[-1]*2) # in sites

h=np.zeros((size,size), complex)

#This method of filling the matrix is such that all talking between sites satisfies chern # = 0
# except for talking between 2 sites that are both inside the walls 
# rather than only requiring 1 site to be inside the walls

#fill matrix bulk sites with chern number 1 entries
va = va1
vb = vb1
t2 = t21
for i in range(size):
    if i%2==0:
        h[i,i]=va
        if i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%(2*m)!=1 and i%(2*m)!=(2*m)-2:
            h[i,(i-2*m+1)%size]=t
            h[i,(i-1)%size]=t
            h[i,(i+1)%size]=t
            h[i,(i+2*m)%size]=t2*np.exp(1j*p) #i just put in t2 times p or minus p for now to not mess with exponentials yet
            h[i,(i-2*m+2)%size]=t2*np.exp(1j*p)
            h[i,(i-2)%size]=t2*np.exp(1j*p)
            h[i,(i-2*m)%size]=t2*np.exp(-1j*p)
            h[i,(i+2*m-2)%size]=t2*np.exp(-1j*p)
            h[i,(i+2)%size]=t2*np.exp(-1j*p)

    else: 
        h[i,i]=vb
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

#fill sites outside walls with chern number 0 entries
va = va0
vb = vb0
t2 = t20
for i in range(size):
    if i%(2*m) not in center:
        if i%2==0:
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

evals, estates = np.linalg.eigh(h)
plt.matshow(estates.real, aspect='auto', cmap=cm.RdBu)
#plt.matshow(h.real)
#plt.matshow(h.imag)

X = range(2*m)
Y = range(n)
X, Y = np.meshgrid(X,Y)

psi0 = estates[:, 97]
E = psi0.real
E = E.reshape(n,2*m)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, E, cmap=cm.RdBu, linewidth=0, antialiased=False)

time = .3
psit = np.matmul(expm(-1j*time*h), psi0)
E = psit.real
E = E.reshape(n,2*m)
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, E, cmap=cm.RdBu, linewidth=0, antialiased=False)

plt.show()
