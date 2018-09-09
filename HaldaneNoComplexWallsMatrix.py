import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import cv2
import os

va=1
t=1

vb = -va
size = 2*m*n
h = np.zeros((size,size), complex)
for i in range(size):
    if i%2==0:
        h[i,i]=va
        h[i,i+1]=t
    else:
        h[i,i]=vb
        h[i,i-1]=t
for i in range(size):
    if np.floor(i/(2*m)!=0 and np.floor(i/(2*m)!=n-1 and i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%2==0:
        h[i,i-1]=t
        h[i, i-2*m+1]=t
    elif np.floor(i/(2*m)!=0 and np.floor(i/(2*m)!=n-1 and i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%2==1:
        h[i,i+1]=t
        h[i,i+2*m-1]=t
    elif i%(2*m)==0 and np.floor(i/(2*m)!=0 and np.floor(i/(2*m)!=n-1:
        h[i,i-2*m+1]=t
    elif i%(2*m)==(2*m-1) and np.floor(i/(2*m)!=0 and np.floor(i/(2*m)!=n-1:
        h[i,i+2*m-1]=t 
    elif np.floor(i/(2*m)==0 and i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%2==0:
        h[i,i-1]=t
    elif np.floor(i/(2*m)==0 and i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%2==1:
        h[i,i+1]=t
    elif np.floor(i/(2*m)==n-1 and i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%2==0:
        h[i,i-1]=t
    elif np.floor(i/(2*m)==n-1 and i%(2*m)!=0 and i%(2*m)!=(2*m)-1 and i%2==1:
        h[i,i+1]=t
    elif np.floor(i/(2*m)==n-1 and i%(2*m)==0
        h[i,i+1]=t
    elif np.floor(i/(2*m)==0 and i%(2*m)==(2*m)-1
        h[i,i-1]=t
