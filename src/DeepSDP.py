import cvxpy as cvx
import numpy as np
import Utils

def solve(nn, x_min, x_max, y_label, target):
    """
    solve the problem using the techniques of DeepSDP
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
    P = cvx.bmat([[-2*cvx.diag(tau), cvx.matmul(cvx.diag(tau),x_min +x_max)], [cvx.matmul((x_min + x_max).T, cvx.diag(tau)), - 2 * cvx.matmul(cvx.matmul(x_min.T, cvx.diag(tau)), x_max)]])
    tmp = cvx.bmat([[np.eye(dim_in), np.zeros((dim_in, num_neurons + 1))], [np.zeros((1, dim_in + num_neurons)), np.array([[1]])]])
    Min = cvx.matmul(cvx.matmul(tmp.T, P), tmp)

    # M_mid
    T = np.zeros((num_neurons, num_neurons))

    nu = cvx.Variable((num_neurons, 1))
    lamb = cvx.Variable((num_neurons, 1))
    eta = cvx.Variable((num_neurons, 1))
    D = cvx.Variable((num_neurons, num_neurons), diag=True)

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
    Q33 = -2 * cvx.matmul(cvx.matmul(X_min, D),X_max.T)

    Q = cvx.bmat([[Q11, Q12, Q13], [Q12.T, Q22, Q23], [Q13.T, Q23.T, Q33]])

    Utils.constructblkDiagonal(np.array(nn.weights, dtype=object), nn.dims)

    A = cvx.hstack([Utils.constructblkDiagonal(np.array(nn.weights[0: -1]), nn.dims), np.zeros((num_neurons, dim_last_hidden))])
    B = cvx.hstack([np.zeros((num_neurons, dim_in)), np.eye(num_neurons)])
    bb = cvx.vstack(nn.bias[0: -1])
    CM_mid = cvx.bmat([[A, bb], [B, np.zeros((B.shape[0], 1))], [np.zeros((1, B.shape[1])), np.array([[1]])]])
    M_mid = cvx.matmul(cvx.matmul(CM_mid.T, Q), CM_mid)
    # M_out
    c = np.zeros((dim_out, 1))
    c[y_label] = -1
    c[target] = 1

    b = cvx.Variable((1,1))
    S = cvx.bmat([[np.zeros((dim_out, dim_out)), c], [c.T, -2 * b]])
    tmp = cvx.bmat([[np.zeros((dim_out, dim_in + num_neurons - dim_last_hidden)), nn.weights[-1], nn.bias[-1]], [np.zeros((1, dim_in + num_neurons)), np.array([[1]])]])
    M_out = cvx.matmul(cvx.matmul(tmp.T, S), tmp)

    # solve
    constraints += [Min + M_out + M_mid << 0]
    problem = cvx.Problem(cvx.Minimize(b), constraints)
    problem.solve(solver=cvx.MOSEK)

    print(problem.value, problem.solver_stats.solver_name, problem.solver_stats.solve_time)
    return problem.value


