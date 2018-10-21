import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import cv2
import os
from numpy.random import uniform
import matplotlib.patches as mpatches
from scipy.sparse import csc_matrix

'''
Global Variables
'''

m = 10
n = 10
va = 0
t = 1
t2 = 1
phi = np.pi/2
l3 = 0
e = .39
disorder = 0
periodic = True

'''
Functions
'''

def B(a,b,c):
    '''
    Returns the matrix B(A,B,C)
    '''
    left = np.concatenate((c,a - np.multiply(1j,b)), axis=0)
    right = np.concatenate((a + np.multiply(1j,b),-c), axis=0)
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

    B0eval, _ = eigsh(B0, k=1, which='SM')
    if abs(B0eval[0]) < e:
        ret = 'in_psuedo'
    
    else:
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
    x = np.diag([i+0j for i in (sorted(range(m)*2)*n)])
    y = np.diag([i+0j for i in sorted(range(n)*2*m)])
    return x,y

def graph_low_evals_of_B():
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
    zs = []
    for x,y in zip(xs,ys):
        B0 = B(X - (np.diag(len(X)*[x])),
           Y - (np.diag(len(Y)*[y])),
           H - (np.diag(len(X)*[l3])))
        B0 = csc_matrix(B0)
        evals, _ = eigsh(B0,k=1,which='SM')
        zs.append(evals[0])

    #ax.set_zlim(-.7,.7)
    ax.azim = -.1
    ax.elev = 0
    ax.scatter(xs, ys, zs)
    ax.text(0,0,.8,'$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f}, $\lambda_3$ = {:.2f}'.format(va,t,t2,l3))
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
    plt.imshow(H.real)
    plt.show()
    X,Y = X_Y(m,n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [i for i in range(m) for _ in range(n)]
    ys = range(n) * m
    for x,y in zip(xs,ys):
        B0 = B(X - (np.diag(len(X)*[x])),
           Y - (np.diag(len(Y)*[y])),
           H - (np.diag(len(X)*[l3])))
        evals, _ = np.linalg.eig(B0)
        ax.scatter([x]*len(B0),[y]*len(B0),evals.real)

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
    for x in range(m):
        for y in range(n):
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
    plt.title('Loring Index on Edged Lattice\n'
                '$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f},\n'
                ' $\lambda_3$ = {:.2f}, dis = {}, e = {}'.format(va,t,t2,l3,disorder,e), loc='left', fontsize='medium')

    return fig

def graph_state(vect,m,n,ax=None):
    realvect = vect.real
    state = realvect.reshape(n,2*m)
    if ax == None:
        fig = plt.figure()
        plt.imshow(state,cmap=cm.RdBu)
        plt.gca().invert_yaxis()
        return fig
    else:
        ax.imshow(state,cmap=cm.RdBu)
        ax.invert_yaxis()
        return ax

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

def propagate(h,m,n,va,t,t2,l3,state,end,frames,title,loc=True):
    path = '/Users/jonathanmichala/All Documents/Independent Study Fall 2018/images'
    num = 1
    statet = np.copy(state)
    for time in np.linspace(0,end,frames):
        statet = np.matmul(expm(-1j*time*h), state)
        fig = graph_state(statet,m,n)
        plt.xlabel('$m$')
        plt.ylabel('$n$')
        plt.title('State after time {:.2f}\n'
                '$V_a$ = {:.2f}, $t$ = {:.2f}, $t\'$ = {:.2f},'
                ' $\lambda_3$ = {:.2f}'.format(time,va,t,t2,l3))
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

def localize(state,m,n,mu,sig):
    s=state
    state0 = np.zeros(len(state), complex)
    state1 = np.zeros(len(state), complex)
    for i in range(len(state)):
        state0[i] = s[i]/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*(((i/(2*m))-mu)/sig)**2)
    for i in range(len(state)):
        state1[i] = state0[i] / (i%(2*m)+1)
    return state1

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



print 'DONE'
plt.show()
print 'EXITED'