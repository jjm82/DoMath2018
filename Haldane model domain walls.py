import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm
import IPython.display as IPdisplay
import glob
from PIL import Image as PIL_Image
from images2gif import writeGif

gif_filename = 'Haldane-model-animation'
path = '/Users/jonathanmichala/All Documents/DOMath 2018/' + gif_filename


'''Parameters'''
showmatrices = False
animate = True
end = 20
frames = 5

localize = True
mu=float(16)
sig=float(1)

n = 30
m = 40 #both in number of cells
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

top = m
bottom = m-1
edgest = bottom

kvals = np.linspace(0, 2 * np.pi, 50, True)
k = []
for kval in kvals:
    for i in range(2*m):
        k.append(kval)
e = []

#make Hn
count = 0
for k1 in kvals:
    for i in range(2*m):
        j = i

        if i%2==0:
            if i in center:
                hn[i,j]= va1 + t21 * (np.exp(1j*(p+k1))+np.exp(1j*(-p-k1)))
                hn[i,(j-2) % (2*m)]= t21 * (np.exp(1j*(p))+np.exp(1j*(-p+k1)))
                hn[i,(j+2) % (2*m)]= t21 * (np.exp(1j*(-p))+np.exp(1j*(p-k1)))
                hn[(i+2) % (2*m), j] = t21 * (np.exp(1j*(-p))+np.exp(1j*(p-k1)))
                hn[(i-2) % (2*m), j] = t21 * (np.exp(1j*(p))+np.exp(1j*(-p+k1)))
            else:
                hn[i,j]= va0
            hn[i,(j-1) % (2*m)]= t
            hn[i,(j+1) % (2*m)]= t * (1+np.exp(1j*(-k1)))
            
        if i%2==1:
            if i in center:
                hn[i,j]= vb1 + t21 * (np.exp(1j*(p-k1))+np.exp(1j*(-p+k1)))
                hn[i,(j-2) % (2*m)]= t21 * (np.exp(1j*(-p))+np.exp(1j*(p+k1)))
                hn[i,(j+2) % (2*m)]= t21 * (np.exp(1j*(p))+np.exp(1j*(-p-k1)))
                hn[(i+2) % (2*m), j]= t21 * (np.exp(1j*(p))+np.exp(1j*(-p-k1)))
                hn[(i-2) % (2*m), j]= t21 * (np.exp(1j*(-p))+np.exp(1j*(p+k1)))
            else:
                hn[i,j]= vb0
            hn[i,(j-1) % (2*m)]= t * (1+np.exp(1j*(k1)))
            hn[i,(j+1) % (2*m)]= t
             

    evalues, evectors = np.linalg.eigh(hn)
    if count == 17:
        evalue = evalues[edgest].real
        state0 = evectors[:, edgest]
        statek = [k1]
        stateeval = [evalue]
    for i in range(len(evalues)):
        evalue = evalues[i].real
        e.append(evalue)
    count += 1

#make H
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

#continue making H
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

if showmatrices:
    plt.matshow(h.real)
    for time in np.linspace(.1,end,frames):
        plt.matshow(expm(-1j*time*h).real)

state0firstrow = state0
for i in range(1,n):
    state0 = np.append(state0, np.exp(1j*statek[0]*i) * state0firstrow)

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
#ax.set_zlim(-.05,.05)
ax.azim = -.1
ax.elev = .1
ax.dist = 8
ax.plot_surface(X, Y, E, cmap=cm.RdBu, linewidth=0, antialiased=False)
plt.savefig(path + '/img00.png')


if animate:
    num = 1
    for time in np.linspace(.1,end,frames):
        statet = np.matmul(expm(-1j*time*h), state0)
        E = statet.real
        E = E.reshape(n,2*m)
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection = '3d')
        #ax.set_zlim(-.05,.05)
        ax.azim = -.1
        ax.elev = .1
        ax.dist = 8
        ax.plot_surface(X, Y, E, cmap=cm.RdBu, linewidth=0, antialiased=False)
        plt.savefig(path + '/img' + str(num).zfill(2) + '.png')

        num += 1
    
    '''plt.close()
    images = [PIL_Image.open(image) for image in glob.glob(path + '/*.png')]
    file_path_name = path + '.gif'
    writeGif(file_path_name, images, duration=0.3)
    IPdisplay.Image(url=file_path_name)'''

fig = plt.figure()
plt.scatter(k,e)
plt.scatter(statek, stateeval, color='red')

plt.show()