import math
import cvxpy as cvx
import numpy as np
from Utils import Utils


def solve(nn, x_min, x_max):
    # Y_min, Y_max, X_min, X_max = nn.interval_arithmetic(x_min, x_max, method="GlobalLip")
    num_neurons = sum(nn.dims[1: -1])

    # index for the three sets
    # Ip = np.where(Y_min > 0)[1]
    # In = np.where(Y_max < 0)[1]
    # Inp = np.setdiff1d(np.arange(num_neurons), np.union1d(Ip, In))

    L_sq = cvx.Variable(neg=False)

    D = cvx.Variable((num_neurons, 1), neg=False)
    T = cvx.diag(D)

    alpha_param = 0
    beta_param = 1

    Q = cvx.bmat([[-2 * alpha_param * beta_param * T, (alpha_param + beta_param) * T],
                  [(alpha_param + beta_param) * T, -2 * T]])
    first_weights = Utils.constructblkDiagonal(nn.weights[0: -1], nn.dims)

    zeros_col = np.zeros((first_weights.shape[0], nn.weights[-1].shape[1]))

    A = cvx.hstack([first_weights, zeros_col])

    eyes = np.eye(A.shape[0])
    init_col = np.zeros((eyes.shape[0], nn.dims[0]))
    B = cvx.hstack([init_col, eyes])

    A_on_B = cvx.vstack([A, B])

    weight_term = -1 * cvx.matmul(nn.weights[-1].T, nn.weights[-1])
    middle_zeros = np.zeros((sum(nn.dims[1: -2]), sum(nn.dims[1: -2])))
    lower_right = Utils.constructblkDiagonal([middle_zeros, weight_term], [middle_zeros.shape[0], weight_term.shape[0]])

    upper_left = L_sq * np.eye(nn.dims[0])

    M = Utils.constructblkDiagonal([upper_left, lower_right], [upper_left.shape[0], lower_right.shape[0]])

    constraints = [(cvx.matmul(cvx.matmul(A_on_B.T, Q), A_on_B)) - M << 0]

    problem = cvx.Problem(cvx.Minimize(L_sq), constraints)
    problem.solve()
    print(math.sqrt(problem.value), problem.solver_stats.solver_name, problem.solver_stats.solve_time)

    return math.sqrt(problem.value)
