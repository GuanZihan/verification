import numpy as np
import cvxpy as cvx

import Utils


def solve(nn, x_min, x_max, y_label, target):
    Y_min, Y_max, X_min, X_max = nn.interval_arithmetic(x_min, x_max, method="GlobalLip")
    num_neurons = sum(nn.dims[1: -1])

    # index for the three sets
    Ip = np.where(Y_min > 0)[1]
    In = np.where(Y_max < 0)[1]
    Inp = np.setdiff1d(np.arange(num_neurons), np.union1d(Ip, In))

    lam = cvx.Variable((num_neurons, 1), neg=False)

    alpha_param = np.zeros((num_neurons, 1))
    beta_param = np.ones((num_neurons, 1))

    Q = cvx.bmat([[-2 * cvx.diag(cvx.multiply(cvx.multiply(alpha_param, beta_param), lam)),
                   cvx.diag(cvx.multiply((alpha_param + beta_param), lam))],
                  [
                      cvx.diag(cvx.multiply((alpha_param + beta_param), lam)),
                      -2 * cvx.diag(lam)
                  ]],
                 )

    first_weights = Utils.constructblkDiagonal(nn.weights[0: -1], nn.dims)

    zeros_col = np.zeros((first_weights.shape[0], nn.weights[-1].shape[1]))

    A = cvx.hstack([first_weights, zeros_col])

    eyes = np.eye(A.shape[0])
    init_col = np.zeros((eyes.shape[0], nn.dims[0]))
    B = cvx.hstack([init_col, eyes])

    A_on_B = cvx.vstack([A, B])

    M = cvx.matmul(cvx.matmul(A_on_B.T, Q), A_on_B)

    print(M.shape)

    return