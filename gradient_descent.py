import numpy as np
from helpers import *


def partial_deriv_w(X,i,j,sigma,d):
    
   # W = weight_matrix(X,sigma)
    temp = 2* single_weight(X, i, j, sigma) * (int(X[i,d])-int(X[j,d]))**2
    power = sigma[d]**3
    return temp / power

    
#quantity in expression (14)
def partial_deriv_p(X,i,j,sigma,d,W,P):

    n = X.shape[0]
    S = 0
    Sum = 0
    for index in range(n):
        S += partial_deriv_w(X,i,index,sigma,d)
        Sum += W[i,index]
    result = partial_deriv_w(X,i,j,sigma,d) - P[i,j] * S
    return result / Sum


def partial_deriv_P_tilde(X,sigma,d,eps,W,P):
   
    n = X.shape[0]
    M = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            M[i,j] = partial_deriv_p(X,i,j,sigma,d,W,P)
    return (1-eps) * M


def partial_deriv_P_tilde_in_blocks(X,sigma,d,eps,l,u,W,P):
    
    M = partial_deriv_P_tilde(X,sigma,d,eps,W,P)
    M_1 = M[0:l,0:l]
    M_2 = M[0:l,l:l+u]
    M_3 = M[l:l+u,0:l]
    M_4 = M[l:l+u,l:l+u]
    
    return M_1,M_2,M_3,M_4
                              
      
#quantity in expression (13) df(u)/d(sigmad)
def derivative_vector(X,f,l,u,sigma,d,eps,temp,W,P):
    
    """P = smoothed_P_matrix_in_blocks(X, l,u,eps, sigma)
    temp = np.linalg.solve(np.eye(u)-P[3],np.eye(u)) """
    a = np.matmul(partial_deriv_P_tilde_in_blocks(X,sigma,d,eps,l,u,W,P)[3],unlabeled_part(f,l,u))
    b = np.matmul(partial_deriv_P_tilde_in_blocks(X,sigma,d,eps,l,u,W,P)[2],labeled_part(f,l,u))
    v = np.matmul(temp,a+b)
    return v
    
#quantity in expression (12) dH/d(sigmad)
def compute_deriv(X,f,l,u,sigma,d,eps,temp,W,P):
    
    s = 0
    v = derivative_vector(X,f,l,u,sigma,d,eps,temp,W,P)
    for i in range(u):
        s += np.log((1-f[l+i])/f[l+i]) * v[i]
    s = s/u
    return s
    

#quantities in expression (12) reunited in a vector
def compute_gradient(X,f,l,u,sigma,eps):
    
    m = X.shape[1]
    grad = []
    P_smoothed = smoothed_P_matrix_in_blocks(X, l,u,eps, sigma)
    temp = np.linalg.solve(np.eye(u)-P_smoothed[3],np.eye(u))
    W = weight_matrix(X,sigma)
    P = P_matrix(X,sigma)
    for d in range(m):
        x = compute_deriv(X,f,l,u,sigma,d,eps,temp,W,P)
        grad.append(x)
        
    return grad



def gradient_descent(X,initial_sigma,max_iters,gamma,f,l,u,eps):
    
    sigma = initial_sigma
    for n_iter in range(max_iters):
        (f_labeled,f_unlabeled) = harmonic_solution(X,f,l,u,sigma)
        f_harmonic = merge(f_labeled, f_unlabeled)
        grad = compute_gradient(X,f_harmonic,l,u,sigma,eps)
        for i in range(X.shape[1]):    
            sigma[i] = sigma[i] - gamma * grad[i]
    return sigma

def entropy(x):
    Hx = -x*np.log(x) - (1-x)*np.log(1-x)
    return Hx

def mean_entropy(f_unlabeled):
    u = len(f_unlabeled)
    S=0
    for i in range (u):
        S+=entropy(f_unlabeled[i])
    H = S/u
    return H
    
def grid_search(X,f,l,u, start, stop):
    
    m = X.shape[1]
    sigma =[]
    for i in range(m):
        sigma.append(start)
    (f_labeled,f_unlabeled) = harmonic_solution(X, f, l, u, sigma)
    temp = np.array([start, mean_entropy(f_unlabeled)])
    for i in range(start, stop):
        for j in range(m):
            sigma[j] = i
        (f_labeled,f_unlabeled) = harmonic_solution(X,f,l,u,sigma)
        H = mean_entropy(f_unlabeled)        
        if (H < temp[1]):
            temp = (i,H)
        sigma_min = temp[0]
    return sigma_min






