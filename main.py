# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:59:02 2020

@author: Julia Bierent
"""
X = np.array([[255,1,1,2,1,1,1,2,2,2,1,1,1,1,2,2],
             [3,1,3,3,1,1,1,3,3,3,1,3,1,1,3,1],
             [3,1,3,3.1,1,1,1,3,3,3,1,3,1,1,3,1],
             [255,1,1,2,1,1,1,2,2,2,1,1,1,1,2,2.1],
             [3,1,3,3,1,1,1,3.1,3,3,1,3,1,1,3,1],
             [255,1,1,2,1,1,1,2,2,2,1,1,1.1,1,2,2],
             [3,1,3,3,1,1,1.1,3,3,3,1,3,1,1,3,1],
             [3,1,3,3,1,1,1,3,3,3,1,3,1,1,3,1],
             [255,1,1,2,1,1,1,2,2,2,1,1.1,1,1,2,2],
             [3,1,3,3,1,1,1,3,3,3,1,3,1,1.1,3,1]])

f = np.array((1,0,0,1,0,1,0,0,1,0))
# number of samples to create
S = 5
# number of data
N = 10
# number of pixels for each data
p = 16
# number of labeled data
l = 2
# number of unlabeled data
u = N-l
q = f.sum()/N
X_spl = np.zeros((S,N,p))
f_spl = np.zeros((S,N))
f_spl_labeled = np.zeros((S,l))
f_spl_unlabeled = np.zeros((S,u))
f_u_classified = np.zeros((S,u))

for i in range(S):
    (X_spl[i], f_spl[i]) = helpers.create_sample(X,f,N,l,u,p)
    (f_spl_labeled[i], f_spl_unlabeled[i]) = helpers.harmonic_solution(X_spl[i], f_spl[i], l, u)
    f_u_classified[i] = helpers.classifier(f_spl_unlabeled[i],q) 

    
