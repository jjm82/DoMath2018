import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm, ldl
from scipy.sparse.linalg import eigsh
import cv2
import os
from numpy.random import uniform
import matplotlib.patches as mpatches
from scipy.sparse import csc_matrix
from sklearn.neighbors import KDTree
from scipy.spatial import Voronoi, voronoi_plot_2d, distance_matrix
from matplotlib import collections  as mc
import pylab as pl

'''
Global Variables
'''
path = '/Users/jonathanmichala/All Documents/Independent Study Fall 2018/images'
m = 15
n = 15
va = 0
t = 1
t2 = 1
phi = np.pi/2
l3 = 0
e = .1
disorder = 3
periodic = False

mu = 2
t = 1
d = 1 #delta
posdis = 0

'''
Functions
ab x   cx
cd y = cy
ax + by = cx
cx + dy = cy
'''

def shift(H,x,y,m,n):
    '''
    shifts H to new center x,y
    '''
    Hnew = np.copy(H)
    xshift = x - m/2
    yshift = y - n/2
    for i in range(m):
        for j in range(n):
            oldi = (i + xshift) % m
            oldj = (j + yshift) % n
            oldind = cell_to_ind(m,oldi,oldj,0)
            newind = cell_to_ind(m,i,j,0)
            Hnew[newind,newind] = H[oldind,oldind]
            Hnew[newind+1,newind+1] = H[oldind+1,oldind+1]
    return Hnew

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def sig(bd):
    '''
    Find the signature of a 2-by-2 block diagonal complex matrix
    '''
    sig = 0
    i = 0
    while i < len(bd):
        if i == len(bd)-1 or bd[i,i+1] == 0 and bd[i+1,i] == 0:
            if bd[i,i] > 0: sig += .5
            else: sig -= .5
            i += 1
        else: i += 2
    return sig

def B(a,b,c):
    '''
    Returns the matrix B(A,B,C)
    '''
    left = np.concatenate((c,a - np.multiply(1j,b)), axis=0)
    right = np.concatenate((a + np.multiply(1j,b),-c), axis=0)
    return np.concatenate((left,right),axis=1)

def newB(X,Y,H,l1,l2):
    global m,n,l3
    X = np.exp(2j*np.pi*(1/float(m)) * X)
    Y = np.exp(2j*np.pi*(1/float(n)) * Y)
    H = H - (np.diag((2*m*n)*[l3]))
    left = np.concatenate((H,X - np.exp(2j*np.pi*l1/float(m)) - np.multiply(1j,Y - np.exp(2j*np.pi*l2/float(n)))), axis=0)
    right = np.concatenate((X - np.exp(2j*np.pi*l1/float(m)) + np.multiply(1j,Y - np.exp(2j*np.pi*l2/float(n))),-H), axis=0)
    return np.concatenate((left,right),axis=1)

def site_analysis(x,y,h,l1,l2):
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
    global l3, e
    ret = None

    B0 = B(x - (np.diag(len(x)*[l1])),
           y - (np.diag(len(y)*[l2])),
           h - (np.diag(len(h)*[l3])))

    '''_, Bdia, _ = ldl(B0)
    print Bdia
    for i in Bdia.diagonal():
        print i
        if abs(i.real) < e:
            return 'in_psuedo'
        if i > 0:
            pcount += 1
        else:
            ncount += 1
    return (pcount - ncount) / 2'''

    B0eval, _ = eigsh(B0, k=1, which='SM')
    if abs(B0eval[0]) < e:
        ret = 'in_psuedo'

    else:
        #_, Bdia, _ = ldl(B0)
        #B0evals = sorted(Bdia.diagonal().real)
        B0evals, _ = np.linalg.eigh(B0)
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
        ret = loring
    return ret

def ldl_site_analysis(x,y,h,l1,l2):
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
    global l3, e
    ret = None

    B0 = B(x - (np.diag(len(x)*[l1])),
           y - (np.diag(len(y)*[l2])),
           h - (np.diag(len(h)*[l3])))

    _, BD, _ = ldl(B0)
    ret = sig(BD)

    B0eval, _ = eigsh(BD, k=1, which='SM')
    if abs(B0eval[0]) < e:
        ret = 'in_psuedo'
    return ret

def check_psuedo(x,y,h,l1,l2):
    '''Only returns whether or not l1,l2 is in the pusedo spectrum'''
    global l3, e
    B0 = B(x - (np.diag(len(x)*[l1])),
           y - (np.diag(len(y)*[l2])),
           h - (np.diag(len(h)*[l3])))
    B0eval, _ = eigsh(B0, k=1, which='SM')
    if abs(B0eval[0]) < e:
        return True
    return False

def index(x,y,h,l1,l2):
    '''returns only the loring index.
    Used when l1,l2 is not in the psuedospectrum'''
    global l3, e
    B0 = B(x - (np.diag(len(x)*[l1])),
           y - (np.diag(len(y)*[l2])),
           h - (np.diag(len(h)*[l3])))

    _, BD, _ = ldl(B0)
    return sig(BD)

def cell_to_ind(m,x,y,site):
    '''
    converts cell ordered pair to matrix index
    site = 0 -> A site
    site = 1 -> B site
    '''
    return 2*m*y + 2*x + site

def hamiltonian():
    '''
    NEED TO DO:
    redo this using scipy.sparse.spdiags
    '''
    '''
    Returns the hamiltonian of a hexagonal lattice
    with the given length, width, and hopping amplitudes 
    '''
    global m,n,va,t,t2,phi,disorder,periodic
    size = 2*m*n
    t2pos = t2*np.exp(1j*phi)
    t2neg = t2*np.exp(-1j*phi)
    h = np.zeros((size,size), complex)

    for i in range(m):
        for j in range(n):
            a_ind = cell_to_ind(m,i,j,0)
            b_ind = a_ind + 1
            dis = disorder * uniform(-.5,.5)
            h[a_ind,a_ind] = va + dis
            h[b_ind,b_ind] = -va - dis
            h[a_ind,b_ind] = t
            h[b_ind,a_ind] = t
            if periodic:
                h[a_ind,cell_to_ind(m,(i-1)%m,j,1)] = t
                h[a_ind,cell_to_ind(m,(i-1)%m,j,0)] = t2pos
                h[b_ind,cell_to_ind(m,(i-1)%m,j,1)] = t2neg
                h[a_ind,cell_to_ind(m,i,(j-1)%n,1)] = t
                h[a_ind,cell_to_ind(m,i,(j-1)%n,0)] = t2neg
                h[b_ind,cell_to_ind(m,i,(j-1)%n,1)] = t2pos
                h[b_ind,cell_to_ind(m,(i+1)%m,j,0)] = t
                h[b_ind,cell_to_ind(m,(i+1)%m,j,1)] = t2pos
                h[a_ind,cell_to_ind(m,(i+1)%m,j,0)] = t2neg
                h[b_ind,cell_to_ind(m,i,(j+1)%n,0)] = t
                h[b_ind,cell_to_ind(m,i,(j+1)%n,1)] = t2neg
                h[a_ind,cell_to_ind(m,i,(j+1)%n,0)] = t2pos
                h[a_ind,cell_to_ind(m,(i-1)%m,(j+1)%n,0)] = t2neg
                h[b_ind,cell_to_ind(m,(i-1)%m,(j+1)%n,1)] = t2pos
                h[a_ind,cell_to_ind(m,(i+1)%m,(j-1)%m,0)] = t2pos
                h[b_ind,cell_to_ind(m,(i+1)%m,(j-1)%m,1)] = t2neg
            else:
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
                    h[a_ind,cell_to_ind(m,i-1,j+1,0)] = t2neg
                    h[b_ind,cell_to_ind(m,i-1,j+1,1)] = t2pos
                if i+1 in range(m) and j-1 in range(n):
                    h[a_ind,cell_to_ind(m,i+1,j-1,0)] = t2pos
                    h[b_ind,cell_to_ind(m,i+1,j-1,1)] = t2neg
    
    #h2 = csc_matrix(h)

    return h

def X_Y(m,n):
    '''
    Returns X and Y matrices in an ordered pair X,Y
    '''
    x = np.diag([i for i in (sorted(range(m)*2)*n)])
    y = np.diag([i for i in sorted(range(n)*2*m)])
    return x,y

def graph_low_evals_of_B():
    global m,n,va,t,t2,l3,phi,periodic
    '''
    Returns a 3d figure of the lowest eigenvalues of B(X-l1,Y-l2,H-l3)
    for different values of l1 and l2
    '''
    H = hamiltonian()
    X,Y = X_Y(m,n)

    '''fig1 = plt.figure()
    plt.imshow(X)
    plt.colorbar()
    plt.title('X normal')
    fig2 = plt.figure()
    plt.imshow(Y)
    plt.colorbar()
    plt.title('Y normal')'''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [i for i in range(m) for _ in range(n)]
    ys = range(n) * m
    zs = []
    for x,y in zip(xs,ys):
        '''B0 = B(X - (np.diag(len(X)*[x])),
           Y - (np.diag(len(Y)*[y])),
           H - (np.diag(len(X)*[l3])))'''
        B0 = newB(X,Y,H,x,y)
        #B0 = csc_matrix(B0)
        evals, _ = eigsh(B0,k=1,which='SM',maxiter=5000)
        #evals, _ = np.linalg.eig(B0)
        #print sorted(evals.real)[9]
        zs.append(sorted(evals.real)[0])

    #ax.set_zlim(-1.5,1.5)
    ax.azim = -20
    ax.elev = 20
    ax.scatter(xs, ys, zs)
    ax.text(0,0,1.5,'Normal, $V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f}, $\lambda_3$ = {:.2f}'.format(va,t,t2,l3))
    ax.set_xlabel('$\lambda_1$')
    ax.set_ylabel('$\lambda_2$')
    ax.set_zlabel('$E$')

    return fig

def graph_all_evals_of_B():
    global m,n,va,t,t2,l3,phi,periodic
    '''
    Returns a 3d figure of the lowest eigenvalues of B(X-l1,Y-l2,H-l3)
    for different values of l1 and l2
    '''
    H = hamiltonian()
    X,Y = X_Y(m,n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [i for i in range(m) for _ in range(n)]
    ys = range(n) * m
    for x,y in zip(xs,ys):
        '''B0 = B(X - (np.diag(len(X)*[x])),
           Y - (np.diag(len(Y)*[y])),
           H - (np.diag(len(X)*[l3])))'''
        B0 = newB(X,Y,H,x,y)
        evals, _ = np.linalg.eig(B0)
        for e in evals:
            if e > -1 and e < 1:
                ax.scatter(x,y)
        #ax.scatter([x]*len(B0),[y]*len(B0),evals.real)

    ax.set_zlim(-1,1)
    ax.text(0,0,.8,'$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f}, $\lambda_3$ = {:.2f}'.format(va,t,t2,l3))
    ax.set_xlabel('$\lambda_1$')
    ax.set_ylabel('$\lambda_2$')
    ax.set_zlabel('$E$')

    return fig

def graph_loring():
    global m,n,va,t,t2,l3,phi,e,periodic
    H = hamiltonian()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    leg1 = leg2 = leg3 = None
    X,Y = X_Y(m,n)
    for x in np.linspace(0,m):
        for y in np.linspace(0,n):
            analysis = site_analysis(X,Y,H,x,y)
            if analysis == 'in_psuedo':
                plt.scatter(x,y,c='red')
                if leg1 == None: leg1 = plt.scatter(x,y,c='red')
            elif analysis == 1:
                plt.scatter(x,y,c='black')
                if leg2 == None: leg2 = plt.scatter(x,y,c='black')
            elif analysis == 0:
                plt.scatter(x,y,facecolors='',edgecolors='black')
                if leg3 == None: leg3 = plt.scatter(x,y,facecolors='',edgecolors='black')
            else: plt.text(x,y,analysis)
    plt.legend((leg1,leg2,leg3),('In psuedospec','Loring index = 1','Loring index = 0'),bbox_to_anchor=(1,1), loc=4, fontsize='small')
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$')
    if periodic:
        latticeType = 'Periodic'
    else:
        latticeType = 'Edged'
    plt.title('Loring Index on {} Lattice\n'
                '$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f},\n'
                ' $\lambda_3$ = {:.2f}, dis = {}, e = {}'.format(latticeType,va,t,t2,l3,disorder,e), loc='left', fontsize='medium')
    

    return fig

def graph_loring_periodic():
    global m,n,va,t,t2,l3,phi,e,periodic
    H = hamiltonian()
    fig = plt.figure()
    leg1 = leg2 = leg3 = None
    X,Y = X_Y(m,n)
    for x in np.linspace(0,m):
        for y in np.linspace(0,n):
            Hnew = shift(H,int(x),int(y),m,n)
            analysis = site_analysis(X,Y,Hnew,m/2 + x - int(x),n/2 + y - int(y))
            if analysis == 'in_psuedo':
                plt.scatter(x,y,c='red')
                if leg1 == None: leg1 = plt.scatter(x,y,c='red')
            elif analysis == 1:
                plt.scatter(x,y,c='black')
                if leg2 == None: leg2 = plt.scatter(x,y,c='black')
            elif analysis == 0:
                plt.scatter(x,y,facecolors='',edgecolors='black')
                if leg3 == None: leg3 = plt.scatter(x,y,facecolors='',edgecolors='black')
            else: plt.text(x,y,analysis)
    plt.legend((leg1,leg2,leg3),('In psuedospec','Loring index = 1','Loring index = 0'),bbox_to_anchor=(1,1), loc=4, fontsize='small')
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$')
    plt.title('Loring Index on Periodic Lattice\n'
                '$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f},\n'
                ' $\lambda_3$ = {:.2f}, dis = {}, e = {}'.format(va,t,t2,l3,disorder,e), loc='left', fontsize='medium')
    

    return fig

def graph_state(vect,ax=None):
    global m,n
    absvect = abs(vect)
    state = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            ind = cell_to_ind(m,i,j,0)
            state[i,j] = absvect[ind] + absvect[ind+1]
    if ax == None:
        fig = plt.figure()
        plt.imshow(state,cmap=cm.RdBu)
        plt.gca().invert_yaxis()
        return fig
    else:
        ax.imshow(state,cmap=cm.RdBu)
        ax.invert_yaxis()
        return ax

def grid_graph_state(vect,points,ax=None):
    global m,n
    absvect = abs(vect)
    state = [0]*(len(absvect)/2)
    for i in range(len(absvect)/2):
        state[i] = absvect[2*i] + absvect[2*i+1]
    fig = plt.figure()
    for i in range(m):
        for j in range(n):
            ind = cell_to_ind(m,i,j,0)/2
            plt.scatter(points[ind][0],points[ind][1],c='red',s=10*((state[ind]+1)**10-1))
    return fig

def graph_eigen_state(h,m,n,ind,va,t,t2,l3):
    Hevals, Hevects = np.linalg.eigh(h)
    fig = plt.figure()
    ax0 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((1, 4), (0, 3))
    plt.suptitle('Eigenstate and Eigenvectors\n'
                '$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f},'
                ' $\lambda_3$ = {:.2f}'.format(va,t,t2,l3))
    graph_state(Hevects[:,ind].real,m,n,ax=ax0)
    ax1.scatter([0]*len(Hevals),Hevals,c='black')
    ax1.scatter(0,Hevals[ind],c='red')
    ax0.set_xlabel('$m$')
    ax0.set_ylabel('$n$')
    ax1.set_ylabel('eigenvalue')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    plt.tight_layout()
    return fig

def propagate(h,state,end,frames,title,loc=True):
    global m,n,mu,t,d,l3
    path = '/Users/jonathanmichala/All Documents/Independent Study Fall 2018/images'
    num = 1
    statet = np.copy(state)
    for time in np.linspace(0,end,frames):
        statet = np.matmul(expm(-1j*time*h), state)
        fig = graph_state(statet)
        plt.xlabel('$m$')
        plt.ylabel('$n$')
        plt.title('State after time {:.2f}\n'
                '$\mu$ = {:.2f}, $t$ = {:.2f}, $\Delta$ = {:.2f},'
                ' $\lambda_3$ = {:.2f}'.format(time,mu,t,d,l3))
        plt.savefig(path + '/img' + str(num).zfill(2) + '.png', format='png')
        plt.close(fig)

        num += 1

    images = [cv2.imread(os.path.join(path, img)) for img in os.listdir(path) if img.endswith(".png")]
    height,width,layers = images[0].shape

    video = cv2.VideoWriter(path + '/' + title + '.mp4',-1,15,(width,height))

    for j in range(len(images)):
        video.write(images[j])

    cv2.destroyAllWindows()
    video.release()

def grid_propagate(h,points,state,end,frames,title,loc=True):
    global m,n,mu,t,d,l3
    path = '/Users/jonathanmichala/All Documents/Independent Study Fall 2018/images'
    num = 1
    statet = np.copy(state)
    for time in np.linspace(0,end,frames):
        statet = np.matmul(expm(-1j*time*h), state)
        fig = grid_graph_state(statet,points)
        plt.xlim(-1,m)
        plt.ylim(-1,n)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('State after time {:.2f}\n'
                '$\mu$ = {:.2f}, $t$ = {:.2f}, $\Delta$ = {:.2f},'
                ' $\lambda_3$ = {:.2f}'.format(time,mu,t,d,l3))
        plt.savefig(path + '/img' + str(num).zfill(2) + '.png', format='png')
        plt.close(fig)

        num += 1

    images = [cv2.imread(os.path.join(path, img)) for img in os.listdir(path) if img.endswith(".png")]
    height,width,layers = images[0].shape

    video = cv2.VideoWriter(path + '/' + title + '.mp4',-1,15,(width,height))

    for j in range(len(images)):
        video.write(images[j])

    cv2.destroyAllWindows()
    video.release()

def localize(state,mean,sig):
    global m,n
    s=state
    state0 = np.zeros(len(state), complex)
    state1 = np.zeros(len(state), complex)
    for i in range(len(state)):
        state0[i] = s[i]/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*(((i/(2*m))-mean)/sig)**2)
    for i in range(len(state)):
        state1[i] = state0[i] / (i%(2*m)+1)
    return state1

def grid_localize(state,points,p1,p2):
    global m,n
    s=state
    state0 = np.zeros(len(state), complex)
    state1 = np.zeros(len(state), complex)
    for i in range(len(points)):
        (x,y) = points[i]
        yl = ((p2[1]-p1[1])/(p2[0]-p1[0]))*(x-p1[0])+p1[1]
        xl = ((p2[0]-p1[0])/(p2[1]-p1[1]))*(y-p1[1])+p1[0]
        d = ((p1[0]+p2[0])/2-x)**2 + ((p1[1]+p2[1])/2-y)**2
        md = ((p1[0]+p2[0])/2-p1[0])**2 + ((p1[1]+p2[1])/2-p1[1])**2
        if x > min([p1[0],p2[0]]) and x < max([p1[0],p2[0]]) and y > min([p1[1],p2[1]]) and y < max([p1[1],p2[1]]):
            if abs(x-xl) < .5 or abs(y-yl) < .5:
                state0[2*i] = s[2*i]*(1-d/md)
                state0[2*i+1] = s[2*i+1]*(1-d/md)
            else:
                state0[2*i] = s[2*i] * .1 * (1-d/(m**2+n**2))
                state0[2*i+1] = s[2*i+1] * .1 * (1-d/(m**2+n**2))
        else:
            state0[2*i] = s[2*i] * .1 * (1-d/(m**2+n**2))
            state0[2*i+1] = s[2*i+1] * .1 * (1-d/(m**2+n**2))
    if False:
        for i in range(len(state)):
            state0[i] = s[i]/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*((i/(2*m)-y)/sig)**2)
        for i in range(len(state)):
            state1[i] = state0[i] / (abs(i%(2*m)/2-x))
    return state0

def loring_over_e(start,end,frames,title):
    global e
    path = '/Users/jonathanmichala/All Documents/Independent Study Fall 2018/images'
    num = 1
    for etemp in np.linspace(start,end,frames):
        e = etemp
        fig = graph_loring()
        plt.savefig(path + '/img' + str(num).zfill(2) + '.png', format='png')
        plt.close(fig)
        num += 1
    
    images = [cv2.imread(os.path.join(path, img)) for img in os.listdir(path) if img.endswith(".png")]
    height,width,layers = images[0].shape

    video = cv2.VideoWriter(path + '/' + title + '.mp4',-1,15,(width,height))

    for j in range(len(images)):
        video.write(images[j])

    cv2.destroyAllWindows()
    video.release()

def loring_over_va(start,end,frames,title):
    global va
    path = '/Users/jonathanmichala/All Documents/Independent Study Fall 2018/images'
    num = 1
    for vatemp in np.linspace(start,end,frames):
        va = vatemp
        fig = graph_loring()
        plt.savefig(path + '/img' + str(num).zfill(2) + '.png', format='png')
        plt.close(fig)
        num += 1
    
    images = [cv2.imread(os.path.join(path, img)) for img in os.listdir(path) if img.endswith(".png")]
    height,width,layers = images[0].shape

    video = cv2.VideoWriter(path + '/' + title + '.mp4',-1,15,(width,height))

    for j in range(len(images)):
        video.write(images[j])

    cv2.destroyAllWindows()
    video.release()

def angle(p1,p2):
    v = [p2[0]-p1[0],p2[1]-p1[1]]
    return np.arccos(np.dot(v,[1,0])/(np.sqrt(v[0]**2 + v[1]**2)))

def vhamiltonian(points):
    '''
    Returns Hamiltonian of a given Nx2 matrix of N sites
    A site hops to its k nearest neighbors
    '''
    v = Voronoi(points)
    fig1 = voronoi_plot_2d(v)
    sites = v.vertices
    ridge_verts = v.ridge_vertices
    H = np.zeros((2*len(sites),2*len(sites)),complex)
    for i in range(len(sites)):
        H[2*i,2*i] = -mu
        H[2*i+1,2*i+1] = mu
    
    for vert_pair in ridge_verts:
        if vert_pair[0] != -1:
            i = vert_pair[0]
            j = vert_pair[1]
            alpha = angle(sites[i],sites[j])
            H[2*i,2*j] = -t
            H[2*i,2*j+1] = -.5*d*(1j*np.cos(alpha) + np.sin(alpha))
            H[2*i+1,2*j] = -.5*d*(1j*np.cos(alpha) - np.sin(alpha))
            H[2*i+1,2*j+1] = t
            alpha2 = alpha + np.pi
            H[2*j,2*i] = -t
            H[2*j,2*i+1] = -.5*d*(1j*np.cos(alpha2) + np.sin(alpha2))
            H[2*j+1,2*i] = -.5*d*(1j*np.cos(alpha2) - np.sin(alpha2))
            H[2*j+1,2*i+1] = t
            
    return H

def v_graph_loring():
    global m,n,va,t,t2,l3,phi,e,periodic
    points = []
    for x in range(2*m):
        for y in range(n):
            randomness = uniform(-.5,.5,2)
            points.append([x+randomness[0],y+randomness[1]])

    points_array = np.array(points)
    H = vhamiltonian(points_array)
    fig = plt.figure()
    leg1 = leg2 = leg3 = None
    X,Y = X_Y(m,n)
    for x in np.linspace(0,m):
        for y in np.linspace(0,n):
            Hnew = shift(H,int(x),int(y),m,n)
            analysis = site_analysis(X,Y,Hnew,m/2 + x - int(x),n/2 + y - int(y))
            if analysis == 'in_psuedo':
                plt.scatter(x,y,c='red')
                if leg1 == None: leg1 = plt.scatter(x,y,c='red')
            elif analysis == 1:
                plt.scatter(x,y,c='black')
                if leg2 == None: leg2 = plt.scatter(x,y,c='black')
            elif analysis == 0:
                plt.scatter(x,y,facecolors='',edgecolors='black')
                if leg3 == None: leg3 = plt.scatter(x,y,facecolors='',edgecolors='black')
            else: plt.text(x,y,analysis)
    plt.legend((leg1,leg2,leg3),('In psuedospec','Loring index = 1','Loring index = 0'),bbox_to_anchor=(1,1), loc=4, fontsize='small')
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$')
    plt.title('Loring Index on Periodic Lattice\n'
                '$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f},\n'
                ' $\lambda_3$ = {:.2f}, dis = {}, e = {}'.format(va,t,t2,l3,disorder,e), loc='left', fontsize='medium')
    

    return fig

def grid_pointset(graph=False):
    global m,n,posdis,disorder
    points = []
    for i in range(n):
        for j in range(m):
            points.append((uniform(j-posdis,j+posdis),uniform(i-posdis,i+posdis)))
    edges = []
    edge_vert_inds = []
    for i in range(len(points)):
        if (i+1)%m != 0:
            edges.append([points[i],points[i+1]])
            edge_vert_inds.append([i,i+1])
        if i+n in range(len(points)):
            edges.append([points[i],points[i+n]])
            edge_vert_inds.append([i,i+n])
        
    if graph:
        _, ax = pl.subplots()
        lc = mc.LineCollection(edges, linewidths=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        plt.scatter([p[0] for p in points],[p[1] for p in points],zorder=10)
    return points, edges, edge_vert_inds

def grid_hamiltonian(points,edge_vert_inds,graph=False):
    '''
    Returns Hamiltonian of a given Nx2 matrix of N sites
    A site hops to its k nearest neighbors
    '''
    global disorder
    H = np.zeros((2*len(points),2*len(points)),complex)
    for i in range(len(points)):
        r = uniform(-.5,.5)
        dis = disorder * r
        H[2*i,2*i] = -mu - dis
        H[2*i+1,2*i+1] = mu + dis
        if graph:
            if dis < 0:
                plt.scatter(points[i][0],points[i][1],s=-100*dis,c='orange',zorder=11)
            else:
                plt.scatter(points[i][0],points[i][1],s=100*dis,c='lime',zorder=11)
    
    for vert_ind_pair in edge_vert_inds:
        i = vert_ind_pair[0]
        j = vert_ind_pair[1]
        alpha = angle(points[i],points[j])
        H[2*i,2*j] = -t
        H[2*i,2*j+1] = -.5*d*(1j*np.cos(alpha) + np.sin(alpha))
        H[2*i+1,2*j] = -.5*d*(1j*np.cos(alpha) - np.sin(alpha))
        H[2*i+1,2*j+1] = t
        alpha2 = alpha + np.pi
        H[2*j,2*i] = -t
        H[2*j,2*i+1] = -.5*d*(1j*np.cos(alpha2) + np.sin(alpha2))
        H[2*j+1,2*i] = -.5*d*(1j*np.cos(alpha2) - np.sin(alpha2))
        H[2*j+1,2*i+1] = t
            
    return H

def grid_graph_loring(H,points):
    global m,n,mu,t,d,l3,phi,e,periodic,posdis
    #fig = plt.figure()
    leg1 = leg2 = leg3 = None
    X,Y = X_Y(m,n)
    for i in np.linspace(0,m-1,30):
        for j in np.linspace(0,n-1,30):
            analysis = index(X,Y,H,i,j)
            if analysis == 1:
                plt.scatter(i,j,c='black')
                if leg2 == None: leg2 = plt.scatter(i,j,c='black')
            elif analysis == 0:
                plt.scatter(i,j,facecolors='',edgecolors='black')
                if leg3 == None: leg3 = plt.scatter(i,j,facecolors='',edgecolors='black')
            else: plt.text(i,j,analysis)
    plt.legend((leg1,leg2,leg3),('In psuedospec','Loring index = 1','Loring index = 0'),bbox_to_anchor=(1,1), loc=4, fontsize='small')
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$')
    plt.title('Loring Index on Edged Lattice\n'
                '$\mu$ = {:.2f}, $t$ = {:.2f}, $\Delta$ = {:.2f},\n'
                ' $\lambda_3$ = {:.2f}, pos_dis = {:.2f}, $\mu$_dis = {}, e = {}'.format(mu,t,d,l3,posdis,disorder,e), loc='left', fontsize='medium')
    

    return None

def grid_graph_eigen_state(h,ind):
    global m,n,mu,t,d,l3
    Hevals, Hevects = np.linalg.eigh(h)
    fig = plt.figure()
    ax0 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((1, 4), (0, 3))
    plt.suptitle('Eigenstate and Eigenvectors\n'
                '$\mu$ = {:.2f}, $t$ = {:.2f}, $\Delta$ = {:.2f},'
                ' $\lambda_3$ = {:.2f}'.format(mu,t,d,l3))
    graph_state(Hevects[:,ind].real,ax=ax0)
    ax1.scatter([0]*len(Hevals),Hevals,c='black')
    ax1.scatter(0,Hevals[ind],c='red')
    ax0.set_xlabel('$m$')
    ax0.set_ylabel('$n$')
    ax1.set_ylabel('eigenvalue')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    plt.tight_layout()
    return fig

def grid_graph_loring_periodic(H,points):
    global m,n,mu,t,d,l3,phi,e,periodic,disorder
    fig = plt.figure()
    leg1 = leg2 = leg3 = None
    X = np.zeros((len(H),len(H)),complex)
    Y = np.zeros((len(H),len(H)),complex)
    for i in range(len(X)):
        X[i,i] = points[i/2][0]
        Y[i,i] = points[i/2][1]
    for x in np.linspace(0,m):
        for y in np.linspace(0,n):
            Hnew = shift(H,int(x),int(y),m,n)
            analysis = site_analysis(X,Y,Hnew,m/2 + x - int(x),n/2 + y - int(y))
            if analysis == 'in_psuedo':
                plt.scatter(x,y,c='red')
                if leg1 == None: leg1 = plt.scatter(x,y,c='red')
            elif analysis == 1:
                plt.scatter(x,y,c='black')
                if leg2 == None: leg2 = plt.scatter(x,y,c='black')
            elif analysis == 0:
                plt.scatter(x,y,facecolors='',edgecolors='black')
                if leg3 == None: leg3 = plt.scatter(x,y,facecolors='',edgecolors='black')
            else: plt.text(x,y,analysis)
    plt.legend((leg1,leg2,leg3),('In psuedospec','Loring index = 1','Loring index = 0'),bbox_to_anchor=(1,1), loc=4, fontsize='small')
    plt.xlabel('$\lambda_1$')
    plt.ylabel('$\lambda_2$')
    plt.title('Loring Index on Periodic Lattice\n'
                '$\mu$ = {:.2f}, $t$ = {:.2f}, $\Delta$ = {:.2f},\n'
                ' $\lambda_3$ = {:.2f}, $\mu$_dis = {}, e = {}'.format(mu,t,d,l3,disorder,e), loc='left', fontsize='medium')
    

    return fig

points, _, edge_vert_inds = grid_pointset()
H = grid_hamiltonian(points, edge_vert_inds)
grid_graph_loring(H,points)
print 'PICK WAVE LOCATION'
loc = plt.ginput(n=2,timeout=-1)
print loc
Hevals, Hevects = np.linalg.eigh(H)
grid_graph_state(normalize(Hevects[:,len(H)/2]),points)

state = normalize(grid_localize(Hevects[:,len(H)/2],points,loc[0],loc[1]))
grid_graph_state(state,points)
grid_propagate(H,points,state,30,10,'test19')

print 'DONE'
plt.show()
print 'EXITED'