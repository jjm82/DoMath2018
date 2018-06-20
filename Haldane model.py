import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


'''Parameters'''

size = 30
va = .3
vb = -va
p = 3 * np.pi / 6
t = 1
t2 = .25
h = np.zeros((2*size,2*size), complex)


''' (1) Plotting eigenvalues for k1 on [0, 2pi]'''


kvals = np.linspace(0, 2 * np.pi, 50, True)
k = []
for kval in kvals:
    for i in range(2*size):
        k.append(kval)
e = []

statefound = False

for k1 in kvals:
    for i in range(2*size):
        for j in range(2*size):
            if i%2==0:
                if i==j:
                    h[i,j]= va - t2 * (np.exp(1j*(p+k1))+np.exp(1j*(-p-k1)))
                if i==j+1:
                    h[i,j]= -t
                if i==j+2:
                    h[i,j]= -t2 * (np.exp(1j*(p))+np.exp(1j*(-p+k1)))
                if i==j-1:
                    h[i,j]= -t * (1+np.exp(1j*(-k1)))
                if i==j-2:
                    h[i,j]= -t2 * (np.exp(1j*(-p))+np.exp(1j*(p-k1)))
            if i%2==1:
                if i==j:
                    h[i,j]= vb - t2 * (np.exp(1j*(p-k1))+np.exp(1j*(-p+k1)))
                if i==j+1:
                    h[i,j]= -t * (1+np.exp(1j*(k1)))
                if i==j+2:
                    h[i,j]= -t2 * (np.exp(1j*(-p))+np.exp(1j*(p+k1)))
                if i==j-1:
                    h[i,j]= -t
                if i==j-2:
                    h[i,j]= -t2 * (np.exp(1j*(p))+np.exp(1j*(-p-k1)))

    evalues, evectors = np.linalg.eig(h)
    for i in range(len(evalues)):
        evalue = evalues[i].real
        if evalue < .05 and evalue > -.05 and not statefound:
            state = evectors[i].real
            statek = [k1]
            stateeval = [evalue]
            statefound = True
        e.append(evalue)

fig = plt.figure()
plt.subplot(211)
plt.scatter(k,e)
if statefound:
    plt.scatter(statek, stateeval, color='red')
    plt.subplot(212)
    x = range(len(state))
    plt.plot(x, state, color='red')


''' (2) Plotting Bloch energy band surfaces for k1xk2 on [0, 2pi]x[0,2pi]
and (3) Plotting Berry curvature on the same space'''


samp = 100
k1s = k2s = np.linspace(-2 * np.pi, 2 * np.pi, samp, True)
k1smesh, k2smesh = np.meshgrid(k1s, k2s)
e1 = np.zeros((samp, samp))
e2 = np.zeros((samp, samp))
fnmesh = np.zeros((samp,samp))

def fn(k1,k2):
    h2 = np.zeros((2,2), complex) #2x2 Hamiltonian
    h2[0,0] = va - 2 * t2 * (np.cos(p + k1) + np.cos(p - k2) + np.cos(p - k1 + k2))
    h2[0,1] = -t * (1 + np.exp(-1j * k1) + np.exp(-1j * k2))
    h2[1,0] = -t * (1 + np.exp(1j * k1) + np.exp(1j * k2))
    h2[1,1] = vb - 2 * t2 * (np.cos(p - k1) + np.cos(p + k2) + np.cos(p + k1 - k2))

    hk1 = np.zeros((2,2), complex) #dH/dk1
    hk1[0,0] = -2 * t2 * (-np.sin(p + k1) + np.sin(p - k1 + k2))
    hk1[0,1] = t * 1j * np.exp(-1j * k1)
    hk1[1,0] = t * -1j * np.exp(1j * k1)
    hk1[1,1] = -2 * t2 * (np.sin(p - k1) - np.sin(p + k1 - k2))

    hk2 = np.zeros((2,2), complex) #dH/dk2
    hk2[0,0] = -2 * t2 * (np.sin(p - k2) - np.sin(p - k1 + k2))
    hk2[0,1] = t * 1j * np.exp(-1j * k2)
    hk2[1,0] = t * -1j * np.exp(1j * k2)
    hk2[1,1] = -2 * t2 * (-np.sin(p + k2) + np.sin(p + k1 - k2))

    return 0



i = j = 0
for k1 in k1s:
    for k2 in k2s:
        h2 = np.zeros((2,2), complex)
        h2[0,0] = va - 2 * t2 * (np.cos(p + k1) + np.cos(p - k2) + np.cos(p - k1 + k2))
        h2[0,1] = -t * (1 + np.exp(-1j * k1) + np.exp(-1j * k2))
        h2[1,0] = -t * (1 + np.exp(1j * k1) + np.exp(1j * k2))
        h2[1,1] = vb - 2 * t2 * (np.cos(p - k1) + np.cos(p + k2) + np.cos(p + k1 - k2))
        evals, estates = np.linalg.eig(h2)
        fnmesh[i,j] = fn(k1,k2)
        evals = evals.real
        e1[i,j] = max(evals)
        e2[i,j] = min(evals)
        i += 1
    j += 1
    i = 0

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection = '3d')
ax.plot_surface(k1smesh, k2smesh, e1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_surface(k1smesh, k2smesh, e2, cmap=cm.coolwarm, linewidth=0, antialiased=False)

'''fig3 = plt.figure()
ax = fig3.add_subplot(111, projection = '3d')
ax.plot_surface(k1smesh, k2smesh, fnmesh, cmap=cm.coolwarm, linewidth=0, antialiased=False)'''

plt.show()