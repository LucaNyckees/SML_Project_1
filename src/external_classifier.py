import numpy as np
from helpers import *
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_processing import generate_data


def new_solution(X, y, l, u, sigma, h_u, eta):
    P = P_matrix_in_blocks(X, l, u, sigma)
    A = np.linalg.solve(np.eye(u) - (1 - eta) * P[3], np.eye(u))
    B = (1 - eta) * np.matmul(P[2], y[0:l]) + eta * h_u
    f_u = np.matmul(A, B)

    return y[0:l], f_u
