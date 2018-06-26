import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import integrate
from scipy import misc
from sympy import symbols, diff
import cmath
import math

'''Parameters'''

size = 30
va = .1
vb = -va
p = np.pi / 2
t = 1.
t2 = .1

def fn(k1,k2):
    global va, vb, p, t, t2

    a = va - 2. * t2 * (math.cos(p + k1) + math.cos(p - k2) + math.cos(p - k1 + k2))
    b = -t * (1. + np.exp(-1j * k1) + np.exp(-1j * k2))
    c = -t * (1. + np.exp(1j * k1) + np.exp(1j * k2))
    d = vb - 2. * t2 * (math.cos(p - k1) + math.cos(p + k2) + math.cos(p + k1 - k2))

    h2 = np.zeros((2,2), complex) #2x2 Hamiltonian
    h2[0,0] = a
    h2[0,1] = b
    h2[1,0] = c
    h2[1,1] = d

    hk1 = np.zeros((2,2), complex) #dH/dk1
    hk1[0,0] = -2 * t2 * (-math.sin(p + k1) + math.sin(p - k1 + k2))
    hk1[0,1] = t * 1j * np.exp(-1j * k1)
    hk1[1,0] = -t * 1j * np.exp(1j * k1)
    hk1[1,1] = -2 * t2 * (math.sin(p - k1) - math.sin(p + k1 - k2))

    hk2 = np.zeros((2,2), complex) #dH/dk2
    hk2[0,0] = -2 * t2 * (math.sin(p - k2) - math.sin(p - k1 + k2))
    hk2[0,1] = t * 1j * np.exp(-1j * k2)
    hk2[1,0] = -t * 1j * np.exp(1j * k2)
    hk2[1,1] = -2 * t2 * (-math.sin(p + k2) + math.sin(p + k1 - k2))

    
    fevals, festates = np.linalg.eigh(h2)
    emin = fevals[0]
    emax = fevals[1]
    statemin = festates[:, 0]
    statemax = festates[:, 1]
    
    ans = ((np.vdot(statemax,np.matmul(hk1,statemin)))*(np.vdot(statemin,np.matmul(hk2,statemax)))-(np.vdot(statemax,np.matmul(hk2,statemin)))*(np.vdot(statemin,np.matmul(hk1,statemax)))).imag / ((emin - emax).real ** 2)
    
    return ans

def chern(vanew, pnew):
    global va
    va = vanew 
    global p
    p = pnew
    global vb
    vb = -vanew
    return integrate.dblquad(fn, 0, 2*np.pi, lambda x: 0, lambda y: 2*np.pi)[0] / (2*np.pi)

'''go from -pi to pi on x-axis: phi
go from -2pi to 2pi on y-axis: Va/t2'''
plot = True

if plot:
    samp = 10
    mesh = np.zeros((samp,samp))

    i = j = 0
    for pnew in np.linspace(-np.pi, np.pi, samp):
        for vanew in np.linspace(-.7, .7, samp):
            mesh[samp - 1 - i,j] = chern(vanew, pnew)
            i += 1
        i = 0
        j += 1

    plt.imshow(mesh, extent=(-np.pi, np.pi, -7, 7), interpolation='nearest', cmap=cm.RdBu)
    plt.show()