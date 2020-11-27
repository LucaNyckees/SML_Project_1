# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:59:02 2020

@author: Julia Bierent
"""
import helpers as helpers
import numpy as np
import random as rd

#Je fais une array "au pif" pour tester notre algo sans avoir les images encore
#(les données sont "parfaites") 16x10
#X = np.array([(1,0,0,1,0,1,0,0,1,0),(0,1,1,0,1,0,1,1,0,1),(0,1,1,0,1,0,1,1,0,1),(0,0,0,0,0,0,0,0,0,0),(1,0,0,1,0,1,0,0,1,0),(0,1,1,0,1,0,1,1,0,1),(1,1,1,1,1,1,1,1,1,1),(1,1,1,1,1,1,1,1,1,1),(0,0,0,0,0,0,0,0,0,0),(0,1,1,0,1,0,1,1,0,1),(0,1,1,0,1,0,1,1,0,1),(0,0,0,0,0,0,0,0,0,0),(1,0,0,1,0,1,0,0,1,0),(1,0,0,1,0,1,0,0,1,0),(1,1,1,1,1,1,1,1,1,1),(0,1,1,0,1,0,1,1,0,1),])
X = np.array([[1,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0],
             [0,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1],
             [0,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1],
             [1,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0],
             [0,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1],
             [1,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0],
             [0,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1],
             [0,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1],
             [1,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0],
             [0,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1]])

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

for i in range(S):
    (X_spl, f_spl) = helpers.create_sample(X,f,N,l,u)


    
