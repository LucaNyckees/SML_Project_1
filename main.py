import tensorflow as tf
import itertools
from magic_helpers import *
import numpy as np
from sklearn.metrics import accuracy_score
from Data import *

ok=GenerateData(1,2,100)
"""print(ok[0])
print(ok[1])
print(len(ok[0]))
print(len(ok[1]))"""

(X, f) = ok
for i in range(len(f)):
    f[i]-=1
    
X = np.array(X)

# number of samples
S = 1
# number of data points (i.e. images)
N = 100
# number of pixels for each data
p = 784
# number of labeled data (>=2)
l = 10
# number of unlabeled data
u = N-l
q = np.sum(f)/N
X_spl = np.zeros((S,N,p))
f_spl = np.zeros((S,N))
f_spl_labeled = np.zeros((S,l))
f_spl_unlabeled = np.zeros((S,u))
f_u_classified = np.zeros((S,u))


"""#print(f)
W_b = weight_in_blocks(X, l, u)
temp = diagonal_in_blocks(X,l,u)[3]-W_b[3]#D_uu-W_uu
print(temp)"""
f_predicted = np.zeros((S,N))

for i in range(S):
    (X_spl[i], f_spl[i]) = create_sample(X,f,N,l,u,p)
    (f_spl_labeled[i], f_spl_unlabeled[i]) = harmonic_solution(X_spl[i], f_spl[i], l, u)
    f_u_classified[i] = classifier(f_spl_unlabeled[i],q)
    f_predicted[i] = np.concatenate((f_spl_labeled[i],f_u_classified[i]))
    print(f_predicted[i])
    print(f_spl[i])
print(accuracy_score(f_spl[0], f_predicted[0]))
