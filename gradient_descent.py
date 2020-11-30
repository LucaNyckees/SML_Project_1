import numpy as np
from magic_helpers import *


def partial_deriv_w(X,i,j,sigma,d):
    
    W = weight_matrix(X)
    temp = 2* W[i,j] * (X[i,d]-X[j,d])**2
    power = sigma[d]**3
    return temp / power

    
#quantity in expression (14)
def partial_deriv_p(X,i,j,sigma,d):
    
    W = weight_matrix(X)
    P = P_matrix(X)
    n = X.shape[0]
    S = 0
    Sum = 0
    for index in range(n):
        S += partial_deriv_w(X,i,index,sigma,d)
        Sum += W[i,index]

    result = partial_deriv_w(X,i,j,sigma,d) - P[i,j] * S
    return result / Sum


def partial_deriv_P_tilde(X,sigma,d,eps):
    
    n = X.shape[0]
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j] = partial_deriv_p(X,i,j,sigma,d)
    
    return (1-eps) * M


def partial_deriv_P_tilde_in_blocks(X,sigma,d,eps,l,u):
    
    M = partial_deriv_P_tilde(X,sigma,d,eps)
    M_1 = M[0:l,0:l]
    M_2 = M[0:l,l:l+u]
    M_3 = M[l:l+u,0:l]
    M_4 = M[l:l+u,l:l+u]
    
    return M_1,M_2,M_3,M_4
                              
      
#quantity in expression (13)
def derivative_vector(X,y,f,l,u,sigma,d,eps):
    
    P = smoothed_P_matrix_in_blocks(X, eps)
    temp = np.linalg.solve(np.eye(u)-P[3],np.eye(u)) 
    a = np.matmul(partial_deriv_P_tilde_in_blocks(X,sigma,d,eps,l,u)[3],unlabeled_part(f,l,u))
    b = np.matmul(partial_deriv_P_tilde_in_blocks(X,sigma,d,eps,l,u)[2],labeled_part(f,l,u))
    temp = np.matmul(temp,a+b)
    return temp
    
#quantity in expression (12)
def compute_deriv(X,y,f,l,u,sigma,d,eps):
    
    s = 0
    v = derivative_vector(X,y,f,l,u,sigma,d,eps)
    
    for i in range(u):
        s += np.log((1-f[l+i])/f[l+i]) * v[i]
    s = s/u
    return s
    

#quantities in expression (12) reunited in a vector
#i.e. computing the gradient of expression (11)
def compute_gradient(X,y,f,l,u,sigma,eps):
    
    m = X.shape[1]
    grad = []
    
    for d in range(m):
        x = compute_deriv(X,y,f,l,u,sigma,d,eps)
        grad.append(x)
        
    return grad

# testing part
X = np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]])
l=2
u=2
y = [1,0,0,1]
f1 = harmonic_solution(X,y,l,u)[0]
f2 = harmonic_solution(X,y,l,u)[1]
f = np.hstack((f1,f2))
sigma = []
for i in range(X.shape[1]):
    sigma.append(380)
    
#print(sigma[4]) 

#print(partial_deriv_P_tilde(X,sigma,2,eps=0.1))
print(compute_gradient(X,y,f,l,u,sigma,eps=0.1))

#gradient descent algorithm implementation
def gradient_descent(X,y,initial_sigma,max_iters,gamma,f,l,u,eps):
    
    sigma = initial_sigma
    for n_iter in range(max_iters):
        grad = compute_gradient(X,y,f,l,u,sigma,eps)
        sigma = sigma - gamma * grad
    return sigma

#print(gradient_descent(X,y,sigma,20,0.01,f,l,u,0.1))









