import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

n1range = 7
n2range = 10
sz = 2 * n1range * n2range

tvals = np.linspace(-10, 10, 50, True)
t = []
for tval in tvals:
    for i in range(sz):
        t.append(tval)
e = []

for tval in tvals:
    H = np.zeros((sz,sz))
    w = 2 * n1range - 1

    # vector of size sz = [v1,v2,v3,...,vn]
    # for cells, incriment n1 if you can, or if you can't, increment n2 and set n1 back to 0.

    for atom in range(sz):
        #first and last atoms
        if atom == 0:
            H[atom, atom + 1] = H[atom, atom + w] = tval
        elif atom == sz - 1:
            H[atom, atom - w] = H[atom, atom - 1] = tval
        
        #n1 max B sites
        elif atom % (2 * n1range) == w:
            H[atom, atom - w] = H[atom, atom - 1] = H[atom, atom + w] = tval
        
        #n1 min A sites
        elif atom % (2 * n1range) == 0:
            H[atom, atom - w] = H[atom, atom + 1] = H[atom, atom + w] = tval
        
        #bulk A sites
        elif atom % 2 == 0 and atom > w:
            H[atom, atom - w] = H[atom, atom - 1] = H[atom, atom + 1] = tval
        
        #bulk B sites
        elif atom % 2 == 1 and atom < (sz - w):
            H[atom, atom - 1] = H[atom, atom + 1] = H[atom, atom + w] = tval
        
        #rest of n2 min and max atoms
        else:
            H[atom, atom - 1] = H[atom, atom + 1] = tval

    evalues, evectors = np.linalg.eig(H)
    for evalue in evalues:
        e.append(evalue)

'''plt.scatter(t,e)
plt.ylabel('Eigenvalues')
plt.xlabel('$t$')
plt.title('2D honeycomb, $n_1$ = %d, $n_2$ = %d' % (n1range, n2range))
plt.show()'''

index = 0
count = 0
for evalue in evalues:
    if evalue < 1 and evalue > -1:
        index = count
    count += 1
print evalues[index]
E = evectors[: ,index]
E = E.reshape(n2range,2 * n1range)

X = range(2 * n1range)
Y = range(n2range)
X, Y = np.meshgrid(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, E, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()