import tensorflow as tf
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt


def constructblkDiagonal(data, dims):
    """
    Construct the block diagonal matrix and returns
    :param data:
    :param dims:
    :return:
    """
    row_num = sum(dims[0: -1])
    col_num = sum(dims[1: ])
    blk = []
    for i in range(len(data)):
        layer = []
        # zeros prior to the block
        for j in range(i):
            layer.append(np.zeros((data[i].shape[0], data[j].shape[1])))
        # block
        layer.append(data[i])

        # zeros successive to the block
        for m in range(len(data) - i - 1):
            layer.append(np.zeros((data[i].shape[0], data[m + i + 1].shape[1])))
        blk.append(cvx.hstack(layer))
    return cvx.vstack(blk)


def solve(nn, x_min, x_max):
    # 1 * num_neurons
    Y_min, Y_max, X_min, X_max = nn.interval_arithmetic(x_min, x_max)
    num_neurons = sum(nn.dims[1: -1])
    dim_in = nn.dims[0]
    dim_out = nn.dims[-1]
    dim_last_hidden = nn.dims[-2]

    # index for the three sets
    Ip = np.where(X_min > 0)[1]
    In = np.where(X_max < 0)[1]
    Inp = np.setdiff1d(np.arange(num_neurons), np.union1d(Ip, In))

    # M_in
    tau = cvx.Variable(dim_in)
    constraints  = []
    constraints += [tau >= 0]
    P = cvx.bmat([[-2*cvx.diag(tau), cvx.matmul(cvx.diag(tau),x_min +x_max)], [cvx.matmul(cvx.transpose(x_min + x_max), cvx.diag(tau)), - 2 * cvx.matmul(cvx.matmul(cvx.transpose(x_min), cvx.diag(tau)), x_max)]])
    tmp = cvx.bmat([[np.eye(dim_in), np.zeros((dim_in, num_neurons + 1))], [np.zeros((1, dim_in + num_neurons)), np.array([[1]])]])
    Min = cvx.matmul(cvx.matmul(cvx.transpose(tmp), P), tmp)

    # M_mid
    T = np.zeros((num_neurons, 1))

    nu = cvx.Variable((num_neurons, 1))
    lamb = cvx.Variable((num_neurons, 1))
    eta = cvx.Variable((num_neurons, 1))
    D = cvx.Variable((num_neurons, num_neurons), symmetric=True)

    if len(In) > 0:
        constraints += [nu[In] >= 0]
    if len(Ip) > 0:
        constraints += [eta[Ip] >= 0]
    if len(Inp) > 0:
        constraints += [nu[Inp] >= 0, eta[Inp] >= 0]

    constraints +=[D >> 0]

    alpha_param = np.zeros((num_neurons, 1))
    alpha_param[Ip] = 1

    beta_param = np.ones((num_neurons, 1))
    beta_param[In] = 0

    Q11 = -2 * cvx.matmul(cvx.diag(cvx.multiply(alpha_param, beta_param)), cvx.diag(lamb))
    Q12 = cvx.matmul(cvx.diag(alpha_param + beta_param), cvx.diag(lamb)) + T
    Q13 = -nu
    Q22 = -2 * cvx.diag(lamb) - 2 * D -2 * T
    Q23 = nu + eta + cvx.matmul(D, X_min.T+ X_max.T)
    Q33 = -2 * cvx.matmul(cvx.matmul(X_min, D),cvx.transpose(X_min))

    Q = cvx.bmat([[Q11, Q12, Q13], [cvx.transpose(Q12), Q22, Q23], [cvx.transpose(Q13), cvx.transpose(Q23), Q33]])

    constructblkDiagonal(np.array(nn.weights), nn.dims)

    A = cvx.hstack([constructblkDiagonal(np.array(nn.weights[0: -1]), nn.dims), np.zeros((num_neurons, dim_last_hidden))])
    B = cvx.hstack([np.zeros((num_neurons, dim_in)), np.eye(num_neurons)])
    bb = cvx.vstack(nn.bias[0: -1])
    CM_mid = cvx.bmat([[A, bb], [B, np.zeros((B.shape[0], 1))], [np.zeros((1, B.shape[1])), np.array([[1]])]])
    M_mid = cvx.matmul(cvx.matmul(cvx.transpose(CM_mid), Q), CM_mid)
    # M_out
    c = np.ones((dim_out, 1))
    b = cvx.Variable((1,1))
    S = cvx.bmat([[np.zeros((dim_out, dim_out)), c], [cvx.transpose(c), -2 * b]])
    tmp = cvx.bmat([[np.zeros((dim_out, dim_in + num_neurons - dim_last_hidden)), nn.weights[-1], nn.bias[-1]], [np.zeros((1, dim_in + num_neurons)), np.array([[1]])]])
    M_out = cvx.matmul(cvx.matmul(cvx.transpose(tmp), S), tmp)
    print(Min.shape, M_out.shape, CM_mid.shape)

    # solve
    constraints += [Min + M_out + M_mid << 0]
    problem = cvx.Problem(cvx.Minimize(b), constraints)
    problem.solve()

