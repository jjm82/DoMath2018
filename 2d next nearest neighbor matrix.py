import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

#n and m are the number of cells in each direction and therefore size is the total # of atoms
n=4
m=4
size = 2*n*m
va=10
vb=5
t=3
t2=1
p= .3

h=np.zeros((size,size))
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
        if i%(2*m) == 0:
            h[i,(i-2*m+1)%size] = h[i,i+1] = h[i,i+2*m-1] = t
            h[i,(i+2*m)%size] = h[i,i+2*m-2] = h[i,(i-2*m+2)%size] = t2*np.exp(1j*p)
            h[i,(i-2*m)%size] = h[i,(i+4*m-2)%size] = h[i,i+2] = t2*np.exp(-1j*p)
        if i%(2*m) == (2*m)-2:
            h[i,i-1] = h[i,(i-2*m+1)%size] = h[i,i+1] = t
            h[i,i-2] = h[i,(i+2*m)%size] = h[i,(i-4*m+2)%size] = t2*np.exp(1j*p)
            h[i,i-2*m+2] = h[i,(i-2*m)%size] = h[i,(i+2*m-2)%size] = t2*np.exp(-1j*p)

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
        if i%(2*m) == 1:
            h[i,i-1] = h[i,(i+2*m-1)%size] = h[i,i+1] = t
            h[i,i+2] = h[i,(i+4*m-2)%size] = h[i,(i-2*m)%size] = t2*np.exp(1j*p)
            h[i,i+2*m-2] = h[i,(i-2*m+2)%size] = h[i,(i+2*m)%size] = t2*np.exp(-1j*p)
        if i%(2*m) == 2*m-1:
            h[i,i-1] = h[i,(i+2*m-1)%size] = h[i,i-2*m+1] = t
            h[i,i-2*m+2] = h[i,(i+2*m-2)%size] = h[i,(i-2*m)%size] = t2*np.exp(1j*p)
            h[i,i-2] = h[i,(i-4*m+2)%size] = h[i,(i+2*m)%size] = t2*np.exp(-1j*p)
print(h)

plt.matshow(h)
plt.show()