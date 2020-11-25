import numpy as np

def single_weight(X, i, j):
    
    # if X has size n x m
    L = []
    m = X.shape[1]
    for d in range(m):
        L.append((X[i,d]-X[j,d])**2)
    S = np.sum(L)
    E = np.exp(-S)
    
    return E


def weight_matrix(X):

    n = X.shape[0]
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i,j] = single_weight(X, i, j)
            
    return W

def quadratic_energy(X, f):
    
    L = []
    n = X.shape[0]
    m = X.shape[1]
    W = weight_matrix(X)
    for i in range(n):
        for j in range(m):
            L.append(W[i,j] * (f[i]-f[j])**2)
    S = L.sum()
    E = 0.5 * S
    
    return E


def magic_entropy_coeff(z):
    
    return -z * np.log(z) - ((1-z) * np.log(1-z))


def label_entropy(f, u, l):
    
    H = 0
    for i in range(u):
        H += magic_entropy_coeff(f[l+i])
    
    return H / u


def labeled_part(f,l,u):
    
    f_labeled = []
    for i in range(l):
        f_labeled.append(f[l])
    return f_labeled


def unlabeled_part(f,l,u):
    
    f_unlabeled = []
    for i in range(u):
        f_unlabeled.append(f[l+i])
    return f_unlabeled  


def weight_in_blocks(X, l, u):
    
    W = weight_matrix(X)
    W_1 = W[0:l,0:l]
    W_2 = W[0:l,l:l+u]
    W_3 = W[l:l+u,0:l]
    W_4 = W[l:l+u,l:l+u]
    
    return W_1,W_2,W_3,W_4


def diagonal_matrix(X):
    
    n = X.shape[0]
    d = []
    W = weight_matrix(X)
    for i in range(n):
        s = W[i,:].sum()
        d.append(s)
    D = np.diag(d)
    return D

def diagonal_in_blocks(X,l,u):
    
    D = diagonal_matrix(X)
    D_1 = D[0:l,0:l]
    D_2 = D[0:l,l:l+u]
    D_3 = D[l:l+u,0:l]
    D_4 = D[l:l+u,l:l+u]
    
    return D_1,D_2,D_3,D_4
    

def laplacian(X):
    
    return diagonal_matrix(X) - weight_matrix(X)
    

X = np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]])
l=2
u=2
y = [1,0,0,1]

def harmonic_solution(X,y,l,u):
    
    W_b = weight_in_blocks(X,l,u)
    
    f_labeled = []
    for i in range(l):
        f_labeled.append(y[i])
    
    temp = np.linalg.solve(diagonal_in_blocks(X,l,u)[0]-W_b[0],np.eye(u)) 
    f_unlabeled = temp * W_b[2] * f_labeled
    
    return f_labeled, f_unlabeled


    
    
    
    
    

    
    
    
    


    
    
    





    



