
from external_classifier import *
from gradient_descent import *
import numpy as np
from helpers import *
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_processing import *
from sklearn.neighbors import KNeighborsClassifier
from cross_validation import cross_valid_eta
import matplotlib.pyplot as plt


ok = GenerateData(1, 2, 100, 16)

(X, f) = ok
for i in range(len(f)):
    f[i] -= 1

X = np.array(X)

S = 1
# number of data
N = 100
# number of pixels for each data
p = 256
# CMN parameter
#q = 0.5
# main hyperparameter    
sigma = 528

# f = merge(harmonic_solution(X,f,l,u,sigma)[0],harmonic_solution(X,f,l,u,sigma)[1])

# uncomment this section to see gradient descent.
# Nota bene : gradients are very small here (10^-4), so to make a meaningful step towards a possible minimum, we
# decide to take an order 10^5 stepsize (gamma), given that we work with values ranging from 1 to 1000.
# any initial value (not returning a singular matrix error) from 1 to 1000 will lead to the value 528.232.
"""
l = 20
# number of unlabeled data
u = N-l
f = merge(harmonic_solution(X,f,l,u,sigma)[0],harmonic_solution(X,f,l,u,sigma)[1])
gradient_descent(X,600,300,100000,f,l,u,0.01)
"""

f_predicted = np.zeros((S, N))

# plotting graphs to compare accuracy of different methods 
# according to the cardinality of the labeled set (l).
list_of_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]
accuracy_thr=[]
accuracy_cmn = []
accuracy_lr = []
accuracy_ext = []


""""
for l in list_of_sizes:
    u = N - l
    X_spl = np.zeros((S, N, p))
    f_spl = np.zeros((S, N))
    f_spl_labeled = np.zeros((S, l))
    f_spl_unlabeled = np.zeros((S, u))
    f_u_classified = np.zeros((S, u))
    f_u_thr=np.zeros((S, u))
    f_spl_labeled_ext = np.zeros((S, l))
    f_spl_unlabeled_ext = np.zeros((S, u))
    f_u_classified_ext = np.zeros((S, u))
    for i in range(S):
        (X_spl[i], f_spl[i]) = create_sample(X, f, N, l, u, p)
        q=laplace_smoothing(f_spl[i][0:l])

        # harmonic solution
        (f_spl_labeled[i], f_spl_unlabeled[i]) = harmonic_solution(X_spl[i], f_spl[i], l, u, sigma)
        f_u_classified[i] = classifier(f_spl_unlabeled[i], q)
        accuracy_cmn.append((accuracy_score(f_spl[i][l:l + u], f_u_classified[i])))
        
        #thresold method
        f_u_thr[i]=classifier_thresold(f_spl_unlabeled[i])
        accuracy_thr.append((accuracy_score(f_spl[i][l:l + u], f_u_thr[i])))
        

        # logistic regression
        logreg = LogisticRegression()
        logreg.fit(X_spl[i][0:l], f_spl[i][0:l])
        y_pred = logreg.predict(X_spl[i][l:l + u])
        accuracy_lr.append(accuracy_score(f_spl[i][l:l + u], y_pred))

        # external classifier (label propagation + logistic regression)
        y_pred_continuous = logreg.predict_proba(X_spl[i][l:l + u])[:, 1]
        (f_spl_labeled_ext[i], f_spl_unlabeled_ext[i]) = new_solution(X_spl[i], f_spl[i], l, u, sigma,
                                                                      y_pred_continuous, eta=0.1)
        f_u_classified_ext[i] = classifier(f_spl_unlabeled_ext[i], q)
        accuracy_ext.append(accuracy_score(f_spl[i][l:l + u], f_u_classified_ext[i]))

# plotting
fig = plt.figure()
plt.plot(list_of_sizes, accuracy_cmn, label='label propagation')
plt.plot(list_of_sizes, accuracy_lr, label='logistic regression')
plt.plot(list_of_sizes, accuracy_ext, label='combined')
plt.plot(list_of_sizes, accuracy_thr, label='thresold method')
plt.ylabel('accuracy')
plt.xlabel('number of labeled images')
plt.legend(loc='lower right')
"""
"""
for l in list_of_sizes:
    u = N - l
    X_spl = np.zeros((S, N, p))
    f_spl = np.zeros((S, N))
    f_spl_labeled = np.zeros((S, l))
    f_spl_unlabeled = np.zeros((S, u))
    f_u_classified = np.zeros((S, u))
    f_u_thr = np.zeros((S, u))
    f_spl_labeled_ext = np.zeros((S, l))
    f_spl_unlabeled_ext = np.zeros((S, u))
    f_u_classified_ext = np.zeros((S, u))
    for i in range(S):
        (X_spl[i], f_spl[i]) = create_sample(X, f, N, l, u, p)
        q=laplace_smoothing(f_spl[i][0:l])

        # harmonic solution
        (f_spl_labeled[i], f_spl_unlabeled[i]) = harmonic_solution(X_spl[i], f_spl[i], l, u, sigma)
        f_u_classified[i] = classifier(f_spl_unlabeled[i], q)
        accuracy_cmn.append((accuracy_score(f_spl[i][l:l + u], f_u_classified[i])))

        # thresold method
        f_u_thr[i] = classifier_thresold(f_spl_unlabeled[i])
        accuracy_thr.append((accuracy_score(f_spl[i][l:l + u], f_u_thr[i])))

        # k-nearest neighbors
        # warning: k must be smaller than l
        knn = KNeighborsClassifier(n_neighbors=8)
        knn.fit(X_spl[i][0:l],f_spl[i][0:l])
        y_pred = knn.predict(X_spl[i][l:l+u])
        accuracy_lr.append(accuracy_score(f_spl[i][l:l + u], y_pred))

        # external classifier (label propagation + k-nearest neighbors )
        if y_pred[0]==1:
            y_pred_continuous = knn.predict_proba(X_spl[i][l:l + u])[:, 1]
        else:
            y_pred_continuous = knn.predict_proba(X_spl[i][l:l+u])[:, 0]
        eta = cross_valid_eta(X_spl[i],f_spl[i],l, u, sigma, y_pred,0.2)
        (f_spl_labeled_ext[i], f_spl_unlabeled_ext[i]) = new_solution(X_spl[i], f_spl[i], l, u, sigma, y_pred, eta)

        f_u_classified_ext[i] = classifier(f_spl_unlabeled_ext[i], q)
        accuracy_ext.append(accuracy_score(f_spl[i][l:l + u], f_u_classified_ext[i]))
# plotting
fig = plt.figure()
plt.plot(list_of_sizes, accuracy_cmn, label='label propagation')
plt.plot(list_of_sizes, accuracy_lr, label='8NN method')
plt.plot(list_of_sizes, accuracy_ext, label='combined')
plt.plot(list_of_sizes, accuracy_thr, label='thresold method')
plt.ylabel('accuracy')
plt.xlabel('number of labeled images')
plt.legend(loc='lower right')
plt.show()
"""



