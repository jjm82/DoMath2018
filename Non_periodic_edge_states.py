import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import cv2
import os

def B(a,b,c):
    left = np.concatenate((c,a - np.multiply(1j,b)), axis=0)
    right = np.concatenate((a + np.multiply(1j,b),-c), axis=0)
    return np.concatenate((left,right),axis=1)

def site_analysis(x,y,h,l1,l2,l3,e,check_psuedo=True,check_loring=False):
    '''
    x,y,h should be square numpy matrices of equivalent size
    l1,l2,l3,e are scalars
    corresponding to lamda1,2,3, and epsilon.
    We compute the eigenvalues of B(x-l1,y-l2,h-l3).

    check_psuedo:
    If the eigenvalue of least magnitude is less than e
    then this site is in the psuedospectrum and we return 1.
    If not we return 0.

    check_loring:
    We return the loring index = (#evals>0)-(#evals<0)/2

    Return a list of the above return values.
    '''
    ret = [-10,-10,[]]

    B0 = B(x - (np.diag(len(x)*[l1])),
           y - (np.diag(len(y)*[l2])),
           h - (np.diag(len(h)*[l3])))

    if check_psuedo:
        in_psuedo = 0
        B0eval, _ = eigsh(B0, k=1, which='SM')
        if abs(B0eval[0]) < e:
            in_psuedo = 1
        ret[0] = in_psuedo
        ret[0] = B0eval[0]
    
    if check_loring:
        B0evals, _ = np.linalg.eigh(B0)
        ret[2] = B0evals
        sz = len(B0evals)
        loring = 0
        if sz % 2 == 0:
            if B0evals[sz/2] > 0:
                i = sz/2 - 1
                while B0evals[i] > 0:
                    loring += 1
                    i -= 1
            else:
                i = sz/2
                while B0evals[i] < 0:
                    loring -= 1
                    i += 1
        else:
            if B0evals[sz/2] > 0:
                i = sz/2 - 1
                loring = .5
                while B0evals[i] > 0:
                    loring += 1
                    i -= 1
            else:
                i = sz/2 + 1
                loring = -.5
                while B0evals[i] < 0:
                    loring -= 1
                    i += 1
        ret[1] = loring
    
    return ret

def cell_to_ind(m,x,y,site):
    '''site = 0 => A site
       site = 1 => B site'''
    return 2*m*y + 2*x + site

def hamiltonian(m,n,va,t,t2=0,phi=0):
    vb = -va
    size = 2*m*n
    t2pos = t2*np.exp(1j*phi)
    t2neg = t2*np.exp(-1j*phi)
    h = np.zeros((size,size), complex)
    
    for i in range(m):
        for j in range(n):
            #A site
            a_ind = cell_to_ind(m,i,j,0)
            b_ind = a_ind + 1
            h[a_ind,a_ind] = va
            h[b_ind,b_ind] = vb
            h[a_ind,b_ind] = t
            h[b_ind,a_ind] = t
            if i-1 in range(m):
                h[a_ind,cell_to_ind(m,i-1,j,1)] = t
                h[a_ind,cell_to_ind(m,i-1,j,0)] = t2pos
                h[b_ind,cell_to_ind(m,i-1,j,1)] = t2neg
            if j-1 in range(n):
                h[a_ind,cell_to_ind(m,i,j-1,1)] = t
                h[a_ind,cell_to_ind(m,i,j-1,0)] = t2neg
                h[b_ind,cell_to_ind(m,i,j-1,1)] = t2pos
            if i+1 in range(m):
                h[b_ind,cell_to_ind(m,i+1,j,0)] = t
                h[b_ind,cell_to_ind(m,i+1,j,1)] = t2pos
                h[a_ind,cell_to_ind(m,i+1,j,0)] = t2neg
            if j+1 in range(n):
                h[b_ind,cell_to_ind(m,i,j+1,0)] = t
                h[b_ind,cell_to_ind(m,i,j+1,1)] = t2neg
                h[a_ind,cell_to_ind(m,i,j+1,0)] = t2pos
            if i-1 in range(m) and j+1 in range(n):
                h[a_ind,cell_to_ind(m,i-1,j+1,0)] = t2pos
                h[b_ind,cell_to_ind(m,i-1,j+1,1)] = t2neg
            if i+1 in range(m) and j-1 in range(n):
                h[a_ind,cell_to_ind(m,i+1,j-1,0)] = t2neg
                h[b_ind,cell_to_ind(m,i+1,j-1,1)] = t2pos
    return h

def X_Y(m,n):
    x = np.diag((sorted(range(m)*2)*n))
    y = np.diag(sorted(range(n)*2*m))
    return x,y

m = 10
n = 10
H = hamiltonian(m,n,0,1,.5,np.pi/2)
X,Y = X_Y(m,n)

if True:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [i for i in range(m) for _ in range(n)]
    ys = range(n) * m
    zs = [site_analysis(X,Y,H,x,y,3,1)[0] for x,y in zip(xs,ys)]

    ax.scatter(xs, ys, zs)

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for x in range(m):
        for y in range(n):
            if site_analysis(X,Y,H,x,y,0,1,check_loring=True)[1] == 1:
                x1.append(x)
                y1.append(y)
            else:
                x0.append(x)
                y0.append(y)
    li0 = plt.scatter(x0,y0,facecolors='',edgecolors='black')
    li1 = plt.scatter(x1,y1,c='black')
    plt.legend((li0,li1),('Loring index = 0','Loring index = 1'),bbox_to_anchor=(1.05, 1), loc=2)
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$')
    plt.title('Loring Index on Edged Lattice')
    plt.text(11,3, '$V_a = 0$\n$t = 1$\n$t\'=.5$\n$\phi = \pi / 2$\n$\lambda_3 = 0$')

print 'DONE'
plt.show()
print 'EXITED'