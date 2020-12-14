from external_classifier import *

def cross_valid_eta(X, y, l, u,sigma ,h_u_continuous,increment):
    init = 0
    (f_labeled, f_unlabeled) = new_solution(X,y,l,u,sigma,h_u_continuous,init)
    y_pred = classifier(f_unlabeled, 0.5)
    score = accuracy_score(y[l:l+u],y_pred)
    print(score)

    for i in np.arange(0.05, 1, increment):
        (f_labeled, f_unlabeled) = new_solution(X,y,l,u,sigma,h_u_continuous,i)
        y_pred = classifier(f_unlabeled, 0.5)
        temp = accuracy_score(y[l:l+u],y_pred)
        print(temp)
        if score < temp:
            score = temp
            init = i
    print(init)
    return init
