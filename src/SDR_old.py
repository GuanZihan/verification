import time

import cvxpy as cvx
import numpy as np
import gc

import Utils


def solve(nn, x_min, x_max, y_label, target):
    """
    solve the problem using the techniques of SDR
    :param nn: Nueral Network
    :param x_min: lower bound of the input x
    :param x_max: upper bound of the input x
    :param y_label: true label for the sample
    :param target: test if the sample can be classified as the target label
    :return: worst-case attack of the sample
    """
    prior = time.time()

    num_hidden_layers = len(nn.dims) - 2
    Y_min, Y_max, X_min, X_max = nn.interval_arithmetic(x_min, x_max, method = "SDR")
    size_big_matrix = 1 + sum(nn.dims[: -1])
    M = cvx.Variable((size_big_matrix, size_big_matrix), symmetric=True)
    constraints = [M >> 0, M[0, 0] == 1]
    x = M[0, 1: 1 + nn.dims[0]]
    X = M[1: 1 + nn.dims[0], 1: 1 + nn.dims[0]]

    constraints += [x >= np.squeeze(X_min[0]), x <= np.squeeze(X_max[0])]
    constraints += [cvx.diag(X) - cvx.multiply(np.squeeze(X_min[0] +X_max[0]), x) + cvx.multiply(np.squeeze(X_min[0]), np.squeeze(X_max[0])) <= 0] #???

    current_pos_matrix = 0

    for i in range(num_hidden_layers):
        W_i = nn.weights[i]
        b_i = np.squeeze(nn.bias[i])
        input_span = np.arange(1 + current_pos_matrix, current_pos_matrix + nn.dims[i] + 1).tolist()
        output_span = np.arange(1 + current_pos_matrix + nn.dims[i], current_pos_matrix + nn.dims[i] + nn.dims[i + 1] + 1).tolist()
        input_linear = M[0, input_span]
        output_linear = M[0, output_span]

        input_quadratic = Utils.subsetMatrix(input_span, input_span, M)
        output_quadratic = Utils.subsetMatrix(output_span, output_span, M)

        cross_terms = Utils.subsetMatrix(input_span, output_span, M)

        constraints += [output_linear >= cvx.matmul(W_i, input_linear) + b_i]
        constraints += [output_linear >=0]
        temp_matrix = cvx.matmul(W_i, cross_terms.T)

        constraints += [cvx.diag(output_quadratic) == cvx.diag(temp_matrix) + cvx.multiply(output_linear, b_i), cvx.diag(output_quadratic) >= 0, cvx.diag(input_quadratic)>=0]
        constraints += [cvx.diag(output_quadratic) - cvx.multiply(np.squeeze(X_min[i+1] + X_max[i+1]), output_linear) + cvx.multiply(np.squeeze(X_min[i+1]), np.squeeze(X_max[i+1])) <= 0]
        current_pos_matrix = current_pos_matrix + nn.dims[i]
        #
        constraints += [cvx.diag(output_quadratic) - cvx.diag(temp_matrix) - cvx.multiply(b_i, output_linear) - cvx.multiply(np.squeeze(X_min[i +1]), output_linear) + cvx.multiply(cvx.matmul(W_i, input_linear), np.squeeze(X_min[i + 1])) + cvx.multiply(np.squeeze(X_min[i+1]), b_i)<=1E-5]

    c = np.zeros((nn.dims[-1], 1))
    c[y_label] = -1
    c[target] = 1


    y_final = cvx.transpose(M[0, np.arange(current_pos_matrix + 1, current_pos_matrix + nn.dims[-2] + 1)])
    obj = cvx.matmul(cvx.transpose(c), cvx.matmul(nn.weights[-1], y_final) + np.squeeze(nn.bias[-1]))
    problem = cvx.Problem(cvx.Minimize(-obj), constraints)
    print('building time is', time.time() - prior)
    problem.solve(solver=cvx.MOSEK)
    print(-problem.value, problem.solver_stats.solver_name, problem.solver_stats.solve_time)
    gc.collect()


    return -problem.value