# plotting graphs to compare accuracy of different methods 
# according to the cardinality of the labeled set (l).
list_of_sizes = [5,10,15,20,25,30,35,40,45,50]
accuracy_cmn = []
accuracy_lr = []
accuracy_ext = []

for l in list_of_sizes:
    u = N-l
    X_spl = np.zeros((S,N,p))
    f_spl = np.zeros((S,N))
    f_spl_labeled = np.zeros((S,l))
    f_spl_unlabeled = np.zeros((S,u))
    f_u_classified = np.zeros((S,u))
    f_spl_labeled_ext = np.zeros((S,l))
    f_spl_unlabeled_ext = np.zeros((S,u))
    f_u_classified_ext = np.zeros((S,u))
    for i in range(S):
        (X_spl[i], f_spl[i]) = create_sample(X,f,N,l,u,p)
        
        # harmonic solution
        (f_spl_labeled[i], f_spl_unlabeled[i]) = harmonic_solution(X_spl[i], f_spl[i], l, u,sigma)
        f_u_classified[i] = classifier(f_spl_unlabeled[i],q)
        accuracy_cmn.append((accuracy_score(f_spl[i][l:l+u], f_u_classified[i])))
        
        # label propagation 
        logreg = LogisticRegression()
        logreg.fit(X_spl[i][0:l],f_spl[i][0:l])
        y_pred = logreg.predict(X_spl[i][l:l+u])
        accuracy_lr.append(accuracy_score(f_spl[i][l:l+u], y_pred))
        
        # external classifier (label propagation + logistic regression)
        y_pred_continuous = logreg.predict_proba(X_spl[i][l:l+u])[:,1]
        (f_spl_labeled_ext[i],f_spl_unlabeled_ext[i]) = new_solution(X_spl[i], f_spl[i], l, u,sigma, y_pred_continuous, eta=0.1)
        f_u_classified_ext[i] = classifier(f_spl_unlabeled_ext[i],q)
        accuracy_ext.append(accuracy_score(f_spl[i][l:l+u], f_u_classified_ext[i]))

# plotting
plt.plot(list_of_sizes, accuracy_cmn, list_of_sizes, accuracy_lr, list_of_sizes, accuracy_ext)
plt.ylabel('accuracy')
plt.xlabel('number of labeled images')
