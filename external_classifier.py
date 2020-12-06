import numpy as np
from helpers import *
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from graph_main import *

def new_solution(X,y,l,u,sigma,h_u, eta):
    
    P = P_matrix_in_blocks(X,l,u,sigma)
    A = np.linalg.solve(np.eye(u)-(1-eta) * P[3], np.eye(u))
    B = (1-eta) * np.matmul(P[2], y[0:l]) + eta * h_u
    f_u = np.matmul(A,B)
    
    return y[0:l],f_u

ok=GenerateData(1,2,100)

(X, f) = ok
for i in range(len(f)):
    f[i]-=1
    
X = np.array(X)

S = 1
# number of data
N = 100
# number of pixels for each data
p = 784
# number of labeled data (>=2)
l = 20
# number of unlabeled data
u = N-l
q = 0.5

sigma = []
for i in range(X.shape[1]):
    sigma.append(380)
    
X_spl = np.zeros((S,N,p))
f_spl = np.zeros((S,N))
f_spl_labeled = np.zeros((S,l))
f_spl_unlabeled = np.zeros((S,u))
f_u_classified = np.zeros((S,u))
f_predicted = np.zeros((S,N))

f_spl_labeled_ext = np.zeros((S,l))
f_spl_unlabeled_ext = np.zeros((S,u))
f_u_classified_ext = np.zeros((S,u))

for i in range(S):
        (X_spl[i], f_spl[i]) = create_sample(X,f,N,l,u,p)
        
        #logistic regression 
        logreg = LogisticRegression()
        logreg.fit(X_spl[i][0:l],f_spl[i][0:l])
        y_pred = logreg.predict(X_spl[i][l:l+u])
        
        #cmn
        (f_spl_labeled[i], f_spl_unlabeled[i]) = harmonic_solution(X_spl[i], f_spl[i], l, u,sigma)
        f_u_classified[i] = classifier(f_spl_unlabeled[i],q)
        #f_predicted[i] = np.concatenate((f_spl_labeled[i],f_u_classified[i]))
        
        #external classifier (cmn + logistic regression)
        (f_spl_labeled_ext[i],f_spl_unlabeled_ext[i]) = new_solution(X_spl[i], f_spl[i], l, u,sigma, y_pred, eta=0.0000000000000001)
        f_u_classified_ext[i] = classifier(f_spl_unlabeled_ext[i],q)
        
        #results
        print(accuracy_score(f_spl[i][l:l+u], y_pred))
        print(accuracy_score(f_spl[i][l:l+u], f_u_classified[i]))
        print(accuracy_score(f_spl[i][l:l+u], f_u_classified_ext[i]))
        #print(accuracy_score(y_pred,f_u_classified[i]))

#0.0000000000000000000001 -->0
#0.000000000000001 --> 
#0.000000000000000001
0.0000000000000001

 
    

    





