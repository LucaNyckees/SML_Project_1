import numpy as np
import random as rd


def single_weight(X, i, j, sigma):
    # if X has size n x m
    m = X.shape[1]
    s = sigma**2
    S = sum(list(map(lambda d: (int(X[i, d]) - int(X[j, d])) ** 2 / s, range(m))))
    E = np.exp(-S)

    return E


def weight_matrix(X, sigma):
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = single_weight(X, i, j, sigma)

    return W


def quadratic_energy(X, f, sigma):
    L = []
    n = X.shape[0]
    m = X.shape[1]
    W = weight_matrix(X, sigma)
    for i in range(n):
        for j in range(m):
            L.append(W[i, j] * (f[i] - f[j]) ** 2)
    S = L.sum()
    E = 0.5 * S

    return E


def magic_entropy_coeff(z):
    return -z * np.log(z) - ((1 - z) * np.log(1 - z))


def label_entropy(f, u, l):
    H = 0
    for i in range(u):
        H += magic_entropy_coeff(f[l + i])

    return H / u


def labeled_part(f, l, u):
    f_labeled = []
    for i in range(l):
        f_labeled.append(f[l])
    return f_labeled


def unlabeled_part(f, l, u):
    f_unlabeled = []
    for i in range(u):
        f_unlabeled.append(f[l + i])
    return f_unlabeled


def weight_in_blocks(X, l, u, sigma):
    W = weight_matrix(X, sigma)
    W_1 = W[0:l, 0:l]
    W_2 = W[0:l, l : l + u]
    W_3 = W[l : l + u, 0:l]
    W_4 = W[l : l + u, l : l + u]

    return W_1, W_2, W_3, W_4


def diagonal_matrix(X, sigma):
    n = X.shape[0]
    d = []
    W = weight_matrix(X, sigma)
    for i in range(n):
        s = W[i, :].sum()
        d.append(s)
    D = np.diag(d)
    return D


def diagonal_in_blocks(X, l, u, sigma):
    D = diagonal_matrix(X, sigma)
    D_1 = D[0:l, 0:l]
    D_2 = D[0:l, l : l + u]
    D_3 = D[l : l + u, 0:l]
    D_4 = D[l : l + u, l : l + u]

    return D_1, D_2, D_3, D_4


def laplacian(X, sigma):
    return diagonal_matrix(X, sigma) - weight_matrix(X, sigma)


def P_matrix(X, sigma):
    n = X.shape[0]
    d = []
    W = weight_matrix(X, sigma)
    D = diagonal_matrix(X, sigma)
    return np.linalg.solve(D, W)


def P_matrix_in_blocks(X, l, u, sigma):
    D = P_matrix(X, sigma)
    D_1 = D[0:l, 0:l]
    D_2 = D[0:l, l : l + u]
    D_3 = D[l : l + u, 0:l]
    D_4 = D[l : l + u, l : l + u]

    return D_1, D_2, D_3, D_4


def smoothed_P_matrix(X, eps, sigma):
    n = X.shape[0]
    P = P_matrix(X, sigma)
    U = np.zeros((n, n))
    return eps * U + (1 - eps) * P


def smoothed_P_matrix_in_blocks(X, l, u, eps, sigma):
    D = smoothed_P_matrix(X, eps, sigma)
    D_1 = D[0:l, 0:l]
    D_2 = D[0:l, l : l + u]
    D_3 = D[l : l + u, 0:l]
    D_4 = D[l : l + u, l : l + u]

    return D_1, D_2, D_3, D_4


def harmonic_solution_smoothed(X, y, l, u, sigma, eps):
    f_labeled = []
    for i in range(l):
        f_labeled.append(y[i])
    temp = np.linalg.solve(
        np.eye(u) - smoothed_P_matrix_in_blocks(X, l, u, eps, sigma)[3], np.eye(u)
    )
    f_unlabeled = np.matmul(
        np.matmul(temp, smoothed_P_matrix_in_blocks(X, l, u, eps, sigma)[2]), f_labeled
    )
    return f_labeled, f_unlabeled


def harmonic_solution(X, y, l, u, sigma):
    W_b = weight_in_blocks(X, l, u, sigma)
    f_labeled = []
    for i in range(l):
        f_labeled.append(y[i])
    temp = np.linalg.solve(diagonal_in_blocks(X, l, u, sigma)[3] - W_b[3], np.eye(u))
    f_unlabeled = np.matmul(np.matmul(temp, W_b[2]), f_labeled)
    return f_labeled, f_unlabeled


def create_sample(X, f, N, l, u, p):
    X_spl = np.zeros((N, p))
    f_spl = []
    unlabeled = 1
    increment = 0
    uniform = 1
    while uniform == 1:
        i = 0
        spl = rd.sample(range(N), l)
        while i < l:
            if f[spl[i]] != f[spl[0]]:
                uniform = 0
            i += 1

    # remplir la partie labeled de X_spl
    for j in range(l):
        X_spl[j] = X[spl[j]]
        f_spl.append(f[spl[j]])

    # remplir la partie unlabeled de X_spl
    for j in range(N):
        # est-ce que l'indice j correspond à un labeled data ?
        for k in range(l):
            if (j) == spl[k]:
                unlabeled = 0
                increment += 1
        # Si j n'est pas un labeled data, alors ajouter la data à X_spl
        if unlabeled == 1:
            X_spl[l + j - increment] = X[j]
            f_spl.append(f[j])
        unlabeled = 1
    return X_spl, f_spl


def classifier(f_unlabeled, q):
    S = sum(f_unlabeled)
    u = len(f_unlabeled)
    f_u_classified = np.zeros(u)
    for i in range(u):
        if q * f_unlabeled[i] / S > (1 - q) * (1 - f_unlabeled[i]) / (u - S):
            f_u_classified[i] = 1
    return f_u_classified


def classifier_thresold(f_unlabeled):
    u = len(f_unlabeled)
    f_u_classified = np.zeros(u)
    for i in range(u):
        if f_unlabeled[i] > 1 / 2:
            f_u_classified[i] = 1
    return f_u_classified


def merge(f_labeled, f_unlabeled):
    l = len(f_labeled)
    u = len(f_unlabeled)
    f = np.zeros(l + u)
    for i in range(l):
        f[i] = f_labeled[i]
    for i in range(u):
        f[l + i] = f_unlabeled[i]
    return f


def laplace_smoothing(f_labeled):
    l = len(f_labeled)
    n_1 = 0
    for i in range(l):
        if f_labeled[i] == 1:
            n_1 = n_1 + 1
    q = (n_1 + 1) / (l + 2)
    return q
