import cvxpy as cvx
import numpy as np

def subsetMatrix(span1, span2, M):
    res_input = np.meshgrid(span1, span2)
    if span1.all() == span2.all():
        res_1 = np.hstack(res_input[0])
        res_1 = res_1.reshape(len(span2), len(span1))
        return M[res_1, res_1]
    else:
        res_1 = np.hstack(res_input[0])
        res_2 = np.hstack(res_input[1])
        res_1 = res_1.reshape( len(span2), len(span1))
        res_2 = res_2.reshape(len(span2), len(span1))
    # all = []
    # for index1, i in enumerate(res_1):
    #     temp = []
    #     for index2, j in enumerate(res_1[index1]):
    #         temp.append(M[res_2[index1][index2], j])
    #     all.append(temp)
    # ret = cvx.bmat(all)

        return M[res_1, res_2]

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


def solve(nn, x_min, x_max, y_label, target):
    """
    solve the problem using the techniques of DeepSDP_plus
    :param nn: Nueral Network
    :param x_min: lower bound of the input x
    :param x_max: upper bound of the input x
    :param y_label: true label for the sample
    :param target: test if the sample can be classified as the target label
    :return: worst-case attack of the sample
    """
    Y_min, Y_max, X_min, X_max = nn.interval_arithmetic(x_min, x_max, method="DeepSDP")
    num_neurons = sum(nn.dims[1: -1])
    dim_in = nn.dims[0]
    dim_out = nn.dims[-1]
    dim_last_hidden = nn.dims[-2]

    # index for the three sets
    Ip = np.where(Y_min > 0)[1]
    In = np.where(Y_max < 0)[1]
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
    D = cvx.Variable((num_neurons, num_neurons), diag=True)
    delta = cvx.Variable((num_neurons, num_neurons), diag=True)

    if len(In) > 0:
        constraints += [nu[In] >= 0, subsetMatrix(In, In, delta) == 0]
    if len(Ip) > 0:
        constraints += [eta[Ip] >= 0, subsetMatrix(Ip, Ip, delta) == 0]
    if len(Inp) > 0:
        constraints += [nu[Inp] >= 0, eta[Inp] >= 0, subsetMatrix(Inp, Inp, delta) >> 0]

    constraints +=[D >> 0]

    alpha_param = np.zeros((num_neurons, 1))
    alpha_param[Ip] = 1

    beta_param = np.ones((num_neurons, 1))
    beta_param[In] = 0

    Q11 = -2 * cvx.matmul(cvx.diag(cvx.multiply(alpha_param, beta_param)), cvx.diag(lamb))
    Q12 = cvx.matmul(cvx.diag(alpha_param + beta_param), cvx.diag(lamb)) + T
    Q13 = -nu + cvx.matmul(delta, Y_max.T)
    Q22 = -2 * cvx.diag(lamb) - 2 * D -2 * T
    Q23 = nu + eta + cvx.matmul(D, X_min.T+ X_max.T) + cvx.matmul(delta, Y_min.T- Y_max.T)
    Q33 = -2 * cvx.matmul(cvx.matmul(X_min, D),cvx.transpose(X_max)) - 2 * cvx.matmul(cvx.matmul(Y_min, delta),cvx.transpose(Y_max))

    Q = cvx.bmat([[Q11, Q12, Q13], [cvx.transpose(Q12), Q22, Q23], [cvx.transpose(Q13), cvx.transpose(Q23), Q33]])

    constructblkDiagonal(np.array(nn.weights, dtype=object), nn.dims)

    A = cvx.hstack([constructblkDiagonal(np.array(nn.weights[0: -1]), nn.dims), np.zeros((num_neurons, dim_last_hidden))])
    B = cvx.hstack([np.zeros((num_neurons, dim_in)), np.eye(num_neurons)])
    bb = cvx.vstack(nn.bias[0: -1])
    CM_mid = cvx.bmat([[A, bb], [B, np.zeros((B.shape[0], 1))], [np.zeros((1, B.shape[1])), np.array([[1]])]])
    M_mid = cvx.matmul(cvx.matmul(cvx.transpose(CM_mid), Q), CM_mid)
    # M_out
    c = np.zeros((dim_out, 1))
    c[y_label] = -1
    c[target] = 1

    b = cvx.Variable((1,1))
    S = cvx.bmat([[np.zeros((dim_out, dim_out)), c], [cvx.transpose(c), -2 * b]])
    tmp = cvx.bmat([[np.zeros((dim_out, dim_in + num_neurons - dim_last_hidden)), nn.weights[-1], nn.bias[-1]], [np.zeros((1, dim_in + num_neurons)), np.array([[1]])]])
    M_out = cvx.matmul(cvx.matmul(cvx.transpose(tmp), S), tmp)

    # solve
    constraints += [Min + M_out + M_mid << 0]
    problem = cvx.Problem(cvx.Minimize(b), constraints)
    problem.solve()

    print(problem.value, problem.solver_stats.solver_name, problem.solver_stats.solve_time)
    return problem.value