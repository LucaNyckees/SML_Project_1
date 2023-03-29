import numpy as np
import plotly.graph_objects as go
from data_processing import *
from external_classifier import *
from gradient_descent import *
from helpers import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def simulation(parameters: dict, data: list, sizes: list[int]) -> go.Figure:
    S = parameters["S"]
    N = parameters["nb_points"]
    p = parameters["nb_pixels"]
    q = parameters["cmn_q"]
    sigma = parameters["sigma"]
    (X, f) = data
    for i in range(len(f)):
        f[i] -= 1

    X = np.array(X)
    accuracy_thr = []
    accuracy_cmn = []
    accuracy_lr = []
    accuracy_ext = []

    for l in tqdm(sizes):
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
            q = laplace_smoothing(f_spl[i][0:l])

            # harmonic solution
            (f_spl_labeled[i], f_spl_unlabeled[i]) = harmonic_solution(
                X_spl[i], f_spl[i], l, u, sigma
            )
            f_u_classified[i] = classifier(f_spl_unlabeled[i], q)
            accuracy_cmn.append(
                (accuracy_score(f_spl[i][l : l + u], f_u_classified[i]))
            )

            # thresold method
            f_u_thr[i] = classifier_thresold(f_spl_unlabeled[i])
            accuracy_thr.append((accuracy_score(f_spl[i][l : l + u], f_u_thr[i])))

            # logistic regression
            logreg = LogisticRegression()
            logreg.fit(X_spl[i][0:l], f_spl[i][0:l])
            y_pred = logreg.predict(X_spl[i][l : l + u])
            accuracy_lr.append(accuracy_score(f_spl[i][l : l + u], y_pred))

            # external classifier (label propagation + logistic regression)
            y_pred_continuous = logreg.predict_proba(X_spl[i][l : l + u])[:, 1]
            (f_spl_labeled_ext[i], f_spl_unlabeled_ext[i]) = new_solution(
                X_spl[i], f_spl[i], l, u, sigma, y_pred_continuous, eta=0.1
            )
            f_u_classified_ext[i] = classifier(f_spl_unlabeled_ext[i], q)
            accuracy_ext.append(
                accuracy_score(f_spl[i][l : l + u], f_u_classified_ext[i])
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=accuracy_cmn, name="label propagation"))
    fig.add_trace(go.Scatter(x=sizes, y=accuracy_lr, name="logistic regression"))
    fig.add_trace(go.Scatter(x=sizes, y=accuracy_ext, name="combined"))

    fig.add_trace(go.Scatter(x=sizes, y=accuracy_thr, name="thresold method"))
    fig.update_layout(legend_title_text="Method")
    fig.update_layout(
        title="Compared method accuracies",
        xaxis_title="number of labeled images",
        yaxis_title="method accuracy",
    )

    return fig
