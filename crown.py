import time

import numpy as np
import sys


def init_layer_bound_relax_matrix_huan(Ws):
    nlayer = len(Ws)
    # preallocate all A matrices
    diags = [None] * nlayer
    # diags[0] is an identity matrix
    diags[0] = np.ones(Ws[0].shape[1], dtype=np.float32)
    for i in range(1, nlayer):
        diags[i] = np.empty(Ws[i].shape[1], dtype=np.float32)
    return diags


def ReLU(vec):
    return np.maximum(vec, 0)


def inc_counter(layer_counter, weights, layer):
    layer_counter[layer] += 1
    # we don't include the last layer, which does not have activation
    n_layers = len(weights) - 1;
    if layer == n_layers:
        # okay, we have enumerated all layers, and now there is an overflow
        return
    if layer_counter[layer] == weights[n_layers - layer - 1].shape[0]:
        # enumerated all neurons for this layer, increment counter for the next layer
        layer_counter[layer] = 0
        inc_counter(layer_counter, weights, layer + 1)

def compute_max_grad_norm(weights, c, j, neuron_states, numlayer, norm=1):
    # layer_counter is the counter for our enumeration progress
    # first element-> second last layer, last elements-> first layershow_histogram
    # extra element to detect overflow (loop ending)
    layer_counter = np.zeros(shape=numlayer, dtype=np.uint16)
    # this is the part 1 of the bound, accumulating all the KNOWN activations
    known_w = np.zeros(weights[0].shape[1])
    # this is the part 2 of the bound, accumulating norms of all unsure activations
    unsure_w_norm = 0.0
    # s keeps the current activation pattern (last layer does not have activation)
    s = np.empty(shape=numlayer - 1, dtype=np.int8)
    # some stats
    skip_count = fixed_paths = unsure_paths = total_loop = 0
    # we will go over ALL possible activation combinations
    while layer_counter[-1] != 1:
        for i in range(numlayer - 1):
            # note that layer_counter is organized in the reversed order
            s[i] = neuron_states[i][layer_counter[numlayer - i - 2]]
        # now s contains the states of each neuron we are currently investigating in each layer
        # for example, for a 4-layer network, s could be [-1, 0, 1], means the first layer neuron
        # no. layer_counter[2] is inactive (-1), second layer neuron no. layer_counter[1] has
        # unsure activation, third layer neuron no. layer_counter[0] is active (1)
        skip = False
        for i in range(numlayer - 1):
            # if any neuron is -1, we skip the entire search range!
            # we look for inactive neuron at the first layer first;
            # we can potentially skip large amount of searches
            if s[i] == -1:
                inc_counter(layer_counter, weights, numlayer - i - 2)
                skip = True
                skip_count += 1
                break
        if not skip:
            total_loop += 1
            # product of all weight parameters
            w = 1.0
            for i in range(0, numlayer - 2):
                # product of all weights along the way
                w *= weights[i + 1][layer_counter[numlayer - (i + 1) - 2], layer_counter[numlayer - i - 2]]
            if np.sum(s) == numlayer - 1:
                fixed_paths += 1
                # all neurons in this path are known to be active.
                known_w += (weights[-1][c, layer_counter[0]] - weights[-1][j, layer_counter[0]]) * w \
                           * weights[0][layer_counter[numlayer - 2]]
            else:
                unsure_paths += 1
                # there must be some neurons have unsure states;
                unsure_w_norm += np.linalg.norm(
                    (weights[-1][c, layer_counter[0]] - weights[-1][j, layer_counter[0]]) * w \
                    * weights[0][layer_counter[numlayer - 2]], norm)
            # increment the counter by 1
            inc_counter(layer_counter, weights, 0)

    known_w_norm = np.linalg.norm(known_w, norm)
    # return the norm and some statistics
    return np.array(
        [known_w_norm + unsure_w_norm]), total_loop, skip_count, fixed_paths, unsure_paths, known_w_norm, unsure_w_norm

def get_layer_bound(W_Nk, b_Nk, UB_prev, LB_prev, is_last, x0, eps, p_n):
    gamma = np.empty_like(W_Nk)
    # gamma = np.transpose(gamma)
    eta = np.empty_like(gamma)

    UB_Nk = np.empty_like(b_Nk)
    LB_Nk = np.empty_like(b_Nk)

    UB_new = np.empty_like(b_Nk)
    LB_new = np.empty_like(b_Nk)

    # print("W_Nk shape")
    # print(W_Nk.shape)

    # I reordered the indices for faster sequential access, so gamma and eta are now transposed
    for ii in range(W_Nk.shape[0]):
        for jj in range(W_Nk.shape[1]):
            if W_Nk[ii, jj] > 0:
                gamma[ii, jj] = UB_prev[jj]
                eta[ii, jj] = LB_prev[jj]
            else:
                gamma[ii, jj] = LB_prev[jj]
                eta[ii, jj] = UB_prev[jj]

        UB_Nk[ii] = np.dot(W_Nk[ii], gamma[ii]) + b_Nk[ii]
        LB_Nk[ii] = np.dot(W_Nk[ii], eta[ii]) + b_Nk[ii]
        # print('UB_Nk[{}] = {}'.format(ii,UB_Nk[ii]))
        # print('LB_Nk[{}] = {}'.format(ii,LB_Nk[ii]))

    Ax0 = np.dot(W_Nk, x0)
    for j in range(W_Nk.shape[0]):

        if p_n == 105:  # p == "i", q = 1
            dualnorm_Aj = np.sum(np.abs(W_Nk[j]))
        elif p_n == 1:  # p = 1, q = i
            dualnorm_Aj = np.max(np.abs(W_Nk[j]))
        elif p_n == 2:  # p = 2, q = 2
            dualnorm_Aj = np.linalg.norm(W_Nk[j])

        UB_new[j] = Ax0[j] + eps * dualnorm_Aj + b_Nk[j]
        LB_new[j] = Ax0[j] - eps * dualnorm_Aj + b_Nk[j]

    is_old = False

    if is_last:  # the last layer has no ReLU
        if is_old:
            return UB_Nk, LB_Nk
        else:
            return UB_new, LB_new
    else:  # middle layers
        return ReLU(UB_Nk), ReLU(LB_Nk)


# matrix version of get_layer_bound_relax
def get_layer_bound_relax_matrix_huan_optimized(Ws, bs, UBs, LBs, neuron_state, nlayer, diags, x0, eps, p_n):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(diags)

    # step 1: create auxillary arrays; we have only nlayer-1 layers of activations
    # we only need to create for this new layer
    idx_unsure = np.nonzero(neuron_state[nlayer - 2] == 0)[0]

    # step 2: calculate all D matrices, there are nlayer such matrices
    # only need to create diags for this layer
    alpha = neuron_state[nlayer - 2].astype(np.float32)
    np.maximum(alpha, 0, alpha)
    alpha[idx_unsure] = UBs[nlayer - 1][idx_unsure] / (UBs[nlayer - 1][idx_unsure] - LBs[nlayer - 1][idx_unsure])
    diags[nlayer - 1][:] = alpha

    # step 3: update matrix A (merged into one loop)
    # step 4: adding all constants (merged into one loop)
    constants = np.copy(bs[-1])  # the last bias
    print(len(bs[-1]))

    # step 5: bounding l_n term for each layer
    UB_final = np.zeros_like(constants)

    LB_final = np.zeros_like(constants)
    # first A is W_{nlayer} D_{nlayer}
    A = Ws[nlayer - 1] * diags[nlayer - 1]
    for i in range(nlayer - 1, 0, -1):
        # constants of previous layers
        constants += np.dot(A, bs[i - 1])
        # unsure neurons of this layer
        idx_unsure = np.nonzero(neuron_state[i - 1] == 0)[0]
        # create l array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        # bound the term A[i] * l_[i], for each element
        for j in range(A.shape[0]):
            l_ub[:] = 0.0
            l_lb[:] = 0.0
            # positive entries in j-th row, unsure neurons
            pos = np.nonzero(A[j][idx_unsure] > 0)[0]
            # negative entries in j-th row, unsure neurons
            neg = np.nonzero(A[j][idx_unsure] < 0)[0]
            # unsure neurons, corresponding to positive entries in A
            idx_unsure_pos = idx_unsure[pos]
            # unsure neurons, corresponding to negative entries in A
            idx_unsure_neg = idx_unsure[neg]
            # for upper bound, set the neurons with positive entries in A to upper bound
            # for upper bound, set the neurons with negative entries in A to lower bound, with l_ub[idx_unsure_neg] = 0
            l_ub[idx_unsure_pos] = LBs[i][idx_unsure_pos]
            # for lower bound, set the neurons with negative entries in A to upper bound
            # for lower bound, set the neurons with positive entries in A to lower bound, with l_lb[idx_unsure_pos] = 0
            l_lb[idx_unsure_neg] = LBs[i][idx_unsure_neg]
            # compute the relavent terms
            UB_final[j] -= np.dot(A[j], l_ub)
            LB_final[j] -= np.dot(A[j], l_lb)
        # compute A for next loop
        if i != 1:
            A = np.dot(A, Ws[i - 1] * diags[i - 1])
        else:
            A = np.dot(A, Ws[i - 1])  # diags[0] is 1
    # after the loop is done we get A0
    UB_final += constants
    LB_final += constants

    # step 6: bounding A0 * x
    x_UB = np.empty_like(UBs[0])
    x_LB = np.empty_like(LBs[0])

    ## original computing UB_final and LB_final bounding each element of A0 * x, so many zeros XD
    # this computation (UB1_final, LB1_final) should be the same as UB_final and LB_final. However, this computation seem to have numerical error due to the different scale of x0 and eps then the dual norm calculation
    old_computation = False

    if old_computation and p_n == 105:
        for j in range(A.shape[0]):
            geq_idx = A[j] >= 0
            le_idx = A[j] < 0
            x_UB[geq_idx] = UBs[0][geq_idx]
            x_UB[le_idx] = LBs[0][le_idx]
            x_LB[geq_idx] = LBs[0][geq_idx]
            x_LB[le_idx] = UBs[0][le_idx]
            UB_final[j] += np.dot(x_UB, A[j])
            LB_final[j] += np.dot(x_LB, A[j])

            # UB1_final = np.copy(UB_final) # check if the ans is the same as original computation before using dual norm
            # LB1_final = np.copy(LB_final) # check if the ans is the same as original computation before using dual norm

            # use relative error, but jit can't print
            # assert np.abs((UB1_final[j]-UB_final[j])/UB1_final[j]) < 10**(-3), "j = %d, UB1_final = %f, UB_final = %f" %(j,UB1_final[j],UB_final[j])
            # assert np.abs((LB1_final[j]-LB_final[j])/LB1_final[j]) < 10**(-3), "j = %d, LB1_final = %f, LB_final = %f" %(j,LB1_final[j],LB_final[j])
            # use relative error
            # assert np.abs((UB1_final[j]-UB_final[j])/UB1_final[j]) < 10**(-4)
            # assert np.abs((LB1_final[j]-LB_final[j])/LB1_final[j]) < 10**(-4)

    else:

        Ax0 = np.dot(A, x0)

        if p_n == 105:  # means p == "i":

            ## new dualnorm computation
            for j in range(A.shape[0]):
                dualnorm_Aj = np.sum(np.abs(A[j]))  # L1 norm of A[j]
                UB_final[j] += (Ax0[j] + eps * dualnorm_Aj)
                LB_final[j] += (Ax0[j] - eps * dualnorm_Aj)

                # eps*dualnorm_Aj and Ax0[j] don't seem to have big difference
                # print("eps*dualnorm_Aj = {}, Ax0[j] = {}".format(eps*dualnorm_Aj, Ax0[j]))

        elif p_n == 1:  # means p == "1"
            for j in range(A.shape[0]):
                dualnorm_Aj = np.max(np.abs(A[j]))  # Linf norm of A[j]
                UB_final[j] += (Ax0[j] + eps * dualnorm_Aj)
                LB_final[j] += (Ax0[j] - eps * dualnorm_Aj)

        elif p_n == 2:  # means p == "2"
            for j in range(A.shape[0]):
                dualnorm_Aj = np.linalg.norm(A[j])  # L2 norm of A[j]
                UB_final[j] += (Ax0[j] + eps * dualnorm_Aj)
                LB_final[j] += (Ax0[j] - eps * dualnorm_Aj)

    return UB_final, LB_final


def get_layer_bound_LP(Ws, bs, UBs, LBs, x0, eps, p, neuron_states, nlayer, pred_label, target_label,
                       compute_full_bounds=False, untargeted=False):
    import gurobipy as grb
    # storing upper and lower bounds for last layer
    UB = np.empty_like(bs[-1])
    LB = np.empty_like(bs[-1])
    # neuron_state is an array: neurons never activated set to -1, neurons always activated set to +1, indefinite set to 0
    # indices
    alphas = []
    # for n layer network, we have n-1 layers of relu
    for i in range(nlayer - 1):
        idx_unsure = (neuron_states[i] == 0).nonzero()[0]
        # neuron_state is an integer array for efficiency reasons. We should convert it to float
        alpha = neuron_states[i].astype(np.float32)
        alpha[idx_unsure] = UBs[i + 1][idx_unsure] / (UBs[i + 1][idx_unsure] - LBs[i + 1][idx_unsure])
        alphas.append(alpha)

    start = time.time()
    m = grb.Model("LP")
    m.setParam("outputflag", 0)
    # disable parallel Gurobi solver, using 1 thread only
    m.setParam("Method", 1)  # dual simplex
    m.setParam("Threads", 1)  # only 1 thread
    # z and zh are list of lists, each list for one layer of variables
    # z starts from 1, matching Zico's notation
    z = []
    z.append(None)
    # z hat starts from 2
    zh = []
    zh.append(None)
    zh.append(None)

    if p == "2" or p == "1":
        # ztrans (transformation of z1 only for lp norm), starts from 1 matching z
        ztrans = []
        ztrans.append(None)

    ## LP codes:

    # we start our label from 1 to nlayer+1 (the last one is the final objective layer)
    # valid range for z: 1 to nlayer (z_1 is just input, z_{nlayer} is the last relu layer output)
    # valid range for z_hat: 2 to nlayer+1 (there is no z_hat_1 as it is the input, z_{nlayer+1} is final output)
    for i in range(1, nlayer + 2):
        if i == 1:  # first layer
            # first layer, only z exists, no z hat
            zzs = []
            zzts = []
            # UBs[0] is for input x. Create a variable for each input
            # and set its lower and upper bounds
            for j in range(1, len(UBs[0]) + 1):
                zij = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=LBs[0][j - 1], ub=UBs[0][j - 1],
                               name="z_" + str(i) + "_" + str(j))
                zzs.append(zij)
                if p == "2" or p == "1":
                    # transformation variable at z1 only
                    if p == "2":
                        ztij = m.addVar(vtype=grb.GRB.CONTINUOUS, name="zt_" + str(i) + "_" + str(j))
                    elif p == "1":
                        ztij = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name="zt_" + str(i) + "_" + str(j))
                    zzts.append(ztij)
            z.append(zzs)
            if p == "2" or p == "1":
                ztrans.append(zzts)
        elif i < nlayer + 1:
            # middle layer, has both z and z hat
            zzs = []
            zzhs = []
            for j in range(1, len(UBs[i - 1]) + 1):
                zij = m.addVar(vtype=grb.GRB.CONTINUOUS, name="z_" + str(i) + "_" + str(j))
                zzs.append(zij)

                zhij = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=-np.inf, name="zh_" + str(i) + "_" + str(j))
                zzhs.append(zhij)
            z.append(zzs)
            zh.append(zzhs)
        else:  # last layer, i == nlayer + 1
            # only has z hat, length is the same as the output
            # there is no relu, so no z
            zzhs = []
            for j in range(1, len(bs[-1]) + 1):
                zhij = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=-np.inf, name="zh_" + str(i) + "_" + str(j))
                zzhs.append(zhij)
            zh.append(zzhs)

    m.update()

    # Adding weights constraints for all layers
    for i in range(1, nlayer + 1):
        W = Ws[i - 1]  # weights of layer i
        for j in range(W.shape[0]):
            """
            sum_term = bs[i-1][j]
            for s in range(W.shape[1]):
                # z start from 1
                sum_term += z[i][s]*W[j,s]
            """
            sum_term = grb.LinExpr(W[j], z[i]) + bs[i - 1][j]
            # this is the output of layer i, and let z_hat_{i+1} equal to it
            # z_hat_{nlayer+1} is the final output (logits)
            m.addConstr(sum_term == zh[i + 1][j], "weights==_" + str(i) + "_" + str(j))
            # m.addConstr(sum_term <= zh[i+1][j], "weights<=_"+str(i)+"_"+str(j))
            # m.addConstr(sum_term >= zh[i+1][j], "weights>=_"+str(i)+"_"+str(j))

    # nlayer network only has nlayer - 1 activations
    for i in range(1, nlayer):
        # UBs[0] is the bounds for input x, so start from 1
        for j in range(len(UBs[i])):
            # neuron_states starts from 0
            if neuron_states[i - 1][j] == 1:
                m.addConstr(z[i + 1][j] == zh[i + 1][j], "LPposr==_" + str(j))
                # m.addConstr(z[i+1][j] <= zh[i+1][j], "LPpos<=_"+str(j))
                # m.addConstr(z[i+1][j] >= zh[i+1][j], "LPpos>=_"+str(j))
            elif neuron_states[i - 1][j] == -1:
                m.addConstr(z[i + 1][j] == 0, "LPneg==_" + str(j))
                # m.addConstr(z[i+1][j] <= 0, "LPneg<=_"+str(j))
                # m.addConstr(z[i+1][j] >= 0, "LPneg>=_"+str(j))
            elif neuron_states[i - 1][j] == 0:
                # m.addConstr(z[i+1][j] >= 0, "LPunsure>=0_"+str(j))
                m.addConstr(z[i + 1][j] >= zh[i + 1][j], "LPunsure>=_" + str(j))
                m.addConstr(z[i + 1][j] <= alphas[i - 1][j] * (zh[i + 1][j] - LBs[i][j]), "LPunsure<=_" + str(j))
            else:
                raise (RuntimeError("unknown neuron_state: " + neuron_states[i]))

            #    #finally, add constraints for z[1], the input -> For p == "i", this is already added in the input variable range zij
    #    for i in range(len(UBs[0])):
    #         m.addConstr(z[1][i] <= UBs[0][i], "inputs+_"+str(i))
    #         m.addConstr(z[1][i] >= LBs[0][i], "inputs-_"+str(i))

    if p == "2":
        # finally, add constraints for z[1] and ztrans[1], the input
        for i in range(len(UBs[0])):
            m.addConstr(ztrans[1][i] == z[1][i] - x0[i], "INPUTtrans_" + str(i))
        # quadratic constraints
        m.addConstr(grb.quicksum(ztrans[1][i] * ztrans[1][i] for i in range(len(UBs[0]))) <= eps * eps,
                    "INPUT L2 norm QCP")
    elif p == "1":
        # finally, add constraints for z[1] and ztrans[1], the input
        temp = []
        for i in range(len(UBs[0])):
            tempi = m.addVar(vtype=grb.GRB.CONTINUOUS)
            temp.append(tempi)

        for i in range(len(UBs[0])):
            # absolute constraints: seem that option1 and 2a, 2c are the right answer (compared to p = 2 result)
            # option 1
            # m.addConstr(ztrans[1][i] >= z[1][i] - x0[i], "INPUTtransPOS_"+str(i))
            # m.addConstr(ztrans[1][i] >= -z[1][i] + x0[i], "INPUTtransNEG_"+str(i))

            # option 2a: same answer as option 1
            # note if we write , the result is different
            # zzz = m.addVar(vtype=grb.GRB.CONTINUOUS)
            # m.addConstr(zzz == z[1][i]-x0[i])
            # m.addConstr(ztrans[1][i] == grb.abs_(zzz), "INPUTtransABS_"+str(i))

            # option 2b: gives different sol as 2a and 2c, guess it's because abs_() has to take a variable,
            # and that's why 2a and 2c use additional variable zzz or temp
            # but now it gives Attribute error on "gurobipy.LinExpr", so can't use this anymore
            # m.addConstr(ztrans[1][i] == grb.abs_(z[1][i]-x0[i]), "INPUTtransABS_"+str(i))

            # option 2c: same answer as 2a
            m.addConstr(temp[i] == z[1][i] - x0[i])
            m.addConstr(ztrans[1][i] == grb.abs_(temp[i]), "INPUTtransABS_" + str(i))

            # option 3: same answer as 2b
            # m.addConstr(ztrans[1][i] <= z[1][i] - x0[i], "INPUTtransPOS_"+str(i))
            # m.addConstr(ztrans[1][i] >= -z[1][i] + x0[i], "INPUTtransNEG_"+str(i))

        # L1 constraints
        m.addConstr(grb.quicksum(ztrans[1][i] for i in range(len(UBs[0]))) <= eps, "INPUT L1 norm")

    # another way to write quadratic constraints
    ###expr = grb.QuadExpr()
    ###expr.addTerms(np.ones(len(UBs[0])), z[1], z[1])
    ###m.addConstr(expr <= eps*eps)

    m.update()

    print("[L2][LP solver initialized] time_lp_init = {:.4f}".format(time.time() - start))
    # for middle layers, need to compute full bounds
    if compute_full_bounds:
        # compute upper bounds
        # z_hat_{nlayer+1} is the logits (final output, or inputs for layer nlayer+1)
        ##for j in [pred_label,target_label]:
        for j in range(Ws[nlayer - 1].shape[0]):
            m.setObjective(zh[nlayer + 1][j], grb.GRB.MAXIMIZE)
            # m.write('grbtest_LP_2layer_'+str(j)+'.lp')
            start = time.time()
            m.optimize()
            UB[j] = m.objVal
            m.reset()
            print("[L2][upper bound solved] j = {}, time_lp_solve = {:.4f}".format(j, time.time() - start))

        # compute lower bounds
        ##for j in [pred_label,target_label]:
        for j in range(Ws[nlayer - 1].shape[0]):
            m.setObjective(zh[nlayer + 1][j], grb.GRB.MINIMIZE)
            # m.write('grbtest_LP_2layer_'+str(j)+'.lp')
            start = time.time()
            m.optimize()
            LB[j] = m.objVal
            m.reset()
            print("[L2][lower bound solved] j = {}, time_lp_solve = {:.4f}".format(j, time.time() - start))

        bnd_gx0 = LB[target_label] - UB[pred_label]
    else:  # use the g_x0 tricks if it's last layer call:
        if untargeted:
            bnd_gx0 = []
            start = time.time()
            for j in range(Ws[nlayer - 1].shape[0]):
                if j != pred_label:
                    m.setObjective(zh[nlayer + 1][pred_label] - zh[nlayer + 1][j], grb.GRB.MINIMIZE)
                    m.optimize()
                    bnd_gx0.append(m.objVal)
                    # print("[L2][Solved untargeted] j = {}, value = {:.4f}".format(j, m.objVal))
                    m.reset()
        else:
            m.setObjective(zh[nlayer + 1][pred_label] - zh[nlayer + 1][target_label], grb.GRB.MINIMIZE)
            start = time.time()
            m.optimize()
            bnd_gx0 = m.objVal
            m.reset()
        print("[L2][g(x) bound solved] time_lp_solve = {:.4f}".format(time.time() - start))

    return UB, LB, bnd_gx0

def compute_max_grad_norm(weights, c, j, neuron_states, numlayer, norm = 1):
    # layer_counter is the counter for our enumeration progress
    # first element-> second last layer, last elements-> first layershow_histogram
    # extra element to detect overflow (loop ending)
    layer_counter = np.zeros(shape=numlayer, dtype=np.uint16)
    # this is the part 1 of the bound, accumulating all the KNOWN activations
    known_w = np.zeros(weights[0].shape[1])
    # this is the part 2 of the bound, accumulating norms of all unsure activations
    unsure_w_norm = 0.0
    # s keeps the current activation pattern (last layer does not have activation)
    s = np.empty(shape=numlayer - 1, dtype=np.int8)
    # some stats
    skip_count = fixed_paths = unsure_paths = total_loop = 0
    # we will go over ALL possible activation combinations
    while layer_counter[-1] != 1:
        for i in range(numlayer-1):
            # note that layer_counter is organized in the reversed order
            s[i] = neuron_states[i][layer_counter[numlayer - i - 2]]
        # now s contains the states of each neuron we are currently investigating in each layer
        # for example, for a 4-layer network, s could be [-1, 0, 1], means the first layer neuron
        # no. layer_counter[2] is inactive (-1), second layer neuron no. layer_counter[1] has
        # unsure activation, third layer neuron no. layer_counter[0] is active (1)
        skip = False
        for i in range(numlayer-1):
            # if any neuron is -1, we skip the entire search range!
            # we look for inactive neuron at the first layer first;
            # we can potentially skip large amount of searches
            if s[i] == -1:
                inc_counter(layer_counter, weights, numlayer - i - 2)
                skip = True
                skip_count += 1
                break
        if not skip:
            total_loop += 1
            # product of all weight parameters
            w = 1.0
            for i in range(0, numlayer-2):
                # product of all weights along the way
                w *= weights[i+1][layer_counter[numlayer - (i+1) - 2], layer_counter[numlayer - i - 2]]
            if np.sum(s) == numlayer - 1:
                fixed_paths += 1
                # all neurons in this path are known to be active.
                known_w += (weights[-1][c,layer_counter[0]] - weights[-1][j,layer_counter[0]]) * w \
                           * weights[0][layer_counter[numlayer - 2]]
            else:
                unsure_paths += 1
                # there must be some neurons have unsure states;
                unsure_w_norm += np.linalg.norm((weights[-1][c,layer_counter[0]] - weights[-1][j,layer_counter[0]]) * w \
                                 * weights[0][layer_counter[numlayer - 2]], norm)
            # increment the counter by 1
            inc_counter(layer_counter, weights, 0)

    known_w_norm = np.linalg.norm(known_w, norm)
    # return the norm and some statistics
    return np.array([known_w_norm + unsure_w_norm]), total_loop, skip_count, fixed_paths, unsure_paths, known_w_norm, unsure_w_norm


def fast_compute_max_grad_norm_2layer_next(prev_c, prev_l, prev_u, W1, neuron_state, norm = 1):
    active_or_unsure_index = np.nonzero(neuron_state >= 0)[0]
    # prev_c is the fix term, direct use 2-layer bound results
    c, l, u = fast_compute_max_grad_norm_2layer(prev_c, W1, neuron_state)
    # now deal with prev_l <= delta <= prev_u term
    # r is dimention for delta.shape[0]
    for r in range(prev_l.shape[0]):
        for k in range(W1.shape[1]):
            for i in active_or_unsure_index:
                if W1[i,k] > 0:
                    u[r,k] += prev_u[r,i] * W1[i,k]
                    l[r,k] += prev_l[r,i] * W1[i,k]
                else:
                    u[r,k] += prev_l[r,i] * W1[i,k]
                    l[r,k] += prev_u[r,i] * W1[i,k]
    return c, l, u


def fast_compute_max_grad_norm(weights, neuron_states, numlayer, norm):
    assert numlayer >= 2
    # merge the last layer weights according to c and j
    # W_vec = np.expand_dims(weights[-1][c] - weights[-1][j], axis=0)
    # const, l, u = fast_compute_max_grad_norm_2layer(W_vec, weights[-2], neuron_states[-1])
    const, l, u = fast_compute_max_grad_norm_2layer(weights[-1], weights[-2], neuron_states[-1])
    # for layers other than the last two layers
    for i in list(range(numlayer - 2))[::-1]:
        const, l, u = fast_compute_max_grad_norm_2layer_next(const, l, u, weights[i], neuron_states[i])
    # get the final upper and lower bound
    l += const
    u += const
    l = np.abs(l)
    u = np.abs(u)

    max_l_u = np.maximum(l, u)
    # print("max_l_u.shape = {}".format(max_l_u.shape))
    # print("max_l_u = {}".format(max_l_u))

    if norm == 1:  # q_n = 1, return L1 norm of max component
        return np.sum(max_l_u, axis=1)
    elif norm == 2:  # q_n = 2, return L2 norm of max component
        return np.sqrt(np.sum(max_l_u ** 2, axis=1))
    elif norm == 105:  # q_n = ord('i'), return Li norm of max component
        # important: return type should be consistent with other returns
        # For other 2 statements, it will return an array: [val], so we need to return an array.
        # numba doesn't support np.max and list, but support arrays

        max_ele = np.zeros((max_l_u.shape[0],))
        for i in range(max_l_u.shape[0]):
            for ii in range(max_l_u.shape[1]):
                if max_l_u[i][ii] > max_ele[i]:
                    max_ele[i] = max_l_u[i][ii]
        return max_ele

        # previous code
        # max_ele = np.array([0.0])
        # for i in range(len(max_l_u[0])):
        #    if max_l_u[0][i] > max_ele[0]:
        #        max_ele[0] = max_l_u[0][i]
        #
        # return max_ele


def fast_compute_max_grad_norm_2layer(W2, W1, neuron_state, norm = 1):
    # even if q_n != 1, then algorithm is the same. The difference is only at the output of fast_compute_max_grad_norm
    assert norm == 1
    # diag = 1 when neuron is active
    diag = np.maximum(neuron_state.astype(np.float32), 0)
    unsure_index = np.nonzero(neuron_state == 0)[0]
    # this is the constant part
    c = np.dot(diag * W2, W1)
    # this is the delta, and l <=0, u >= 0
    l = np.zeros((W2.shape[0], W1.shape[1]))
    u = np.zeros_like(l)
    for r in range(W2.shape[0]):
        for k in range(W1.shape[1]):
            for i in unsure_index:
                prod = W2[r,i] * W1[i,k]
                if prod > 0:
                    u[r,k] += prod
                else:
                    l[r,k] += prod
    return c, l, u


def get_layer_bound_relax_adaptive_matrix_huan_optimized(Ws, bs, UBs, LBs, neuron_state, nlayer, diags, x0, eps, p_n):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(diags)

    # step 1: create auxillary arrays; we have only nlayer-1 layers of activations
    # we only need to create for this new layer
    idx_unsure = np.nonzero(neuron_state[nlayer - 2] == 0)[0]

    # step 2: calculate all D matrices, there are nlayer such matrices
    # only need to create diags for this layer
    alpha = neuron_state[nlayer - 2].astype(np.float32)
    np.maximum(alpha, 0, alpha)
    # prefill diags with u/(u-l)
    alpha[idx_unsure] = UBs[nlayer - 1][idx_unsure] / (UBs[nlayer - 1][idx_unsure] - LBs[nlayer - 1][idx_unsure])
    diags[nlayer - 1][:] = alpha

    # step 3: update matrix A (merged into one loop)
    # step 4: adding all constants (merged into one loop)
    constants_ub = np.copy(bs[-1])  # the last bias
    constants_lb = np.copy(bs[-1])  # the last bias

    # step 5: bounding l_n term for each layer
    UB_final = np.zeros_like(constants_ub)
    LB_final = np.zeros_like(constants_lb)
    # first A is W_{nlayer} D_{nlayer}
    # A_UB = Ws[nlayer-1] * diags[nlayer-1]
    A_UB = np.copy(Ws[nlayer - 1])
    # A_LB = Ws[nlayer-1] * diags[nlayer-1]
    A_LB = np.copy(Ws[nlayer - 1])
    for i in range(nlayer - 1, 0, -1):
        # unsure neurons of this layer
        idx_unsure = np.nonzero(neuron_state[i - 1] == 0)[0]
        # create l array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        # bound the term A[i] * l_[i], for each element
        for j in range(A_UB.shape[0]):
            l_ub[:] = 0.0
            l_lb[:] = 0.0
            diags_ub = np.copy(diags[i])
            diags_lb = np.copy(diags[i])
            # positive entries in j-th row, unsure neurons
            pos_ub = np.nonzero(A_UB[j][idx_unsure] > 0)[0]
            pos_lb = np.nonzero(A_LB[j][idx_unsure] > 0)[0]
            # negative entries in j-th row, unsure neurons
            neg_ub = np.nonzero(A_UB[j][idx_unsure] < 0)[0]
            neg_lb = np.nonzero(A_LB[j][idx_unsure] < 0)[0]
            # unsure neurons, corresponding to positive entries in the j-th row of A
            idx_unsure_pos_ub = idx_unsure[pos_ub]
            idx_unsure_pos_lb = idx_unsure[pos_lb]
            # unsure neurons, corresponding to negative entries in the j-th row of A
            idx_unsure_neg_ub = idx_unsure[neg_ub]
            idx_unsure_neg_lb = idx_unsure[neg_lb]

            # for upper bound, set the neurons with positive entries in A to upper bound
            l_ub[idx_unsure_pos_ub] = LBs[i][idx_unsure_pos_ub]
            # for upper bound, set the neurons with negative entries in A to lower bound, depending on the magnitude of LBs[i][idx_unsure_neg] and UBs[i][idx_unsure_neg]
            mask = np.abs(LBs[i][idx_unsure_neg_ub]) > np.abs(UBs[i][idx_unsure_neg_ub])
            # for |LB| > |UB|, use y = 0 as the lower bound, adjust A
            diags_ub[idx_unsure_neg_ub[mask]] = 0.0
            # for |LB| < |UB|, use y = x as the lower bound, adjust A
            diags_ub[idx_unsure_neg_ub[np.logical_not(mask)]] = 1.0
            # update the j-th row of A with diagonal matrice
            A_UB[j] = A_UB[j] * diags_ub

            # for lower bound, set the neurons with negative entries in A to upper bound
            l_lb[idx_unsure_neg_lb] = LBs[i][idx_unsure_neg_lb]
            # for upper bound, set the neurons with positive entries in A to lower bound, depending on the magnitude of LBs[i][idx_unsure_pos] and UBs[i][idx_unsure_pos]
            mask = np.abs(LBs[i][idx_unsure_pos_lb]) > np.abs(UBs[i][idx_unsure_pos_lb])
            # for |LB| > |UB|, use y = 0 as the lower bound, adjust A
            diags_lb[idx_unsure_pos_lb[mask]] = 0.0
            # for |LB| > |UB|, use y = x as the lower bound, adjust A
            diags_lb[idx_unsure_pos_lb[np.logical_not(mask)]] = 1.0
            # update A with diagonal matrice
            A_LB[j] = A_LB[j] * diags_lb

            # compute the relavent terms
            UB_final[j] -= np.dot(A_UB[j], l_ub)
            LB_final[j] -= np.dot(A_LB[j], l_lb)
        # constants of previous layers
        constants_ub += np.dot(A_UB, bs[i - 1])
        constants_lb += np.dot(A_LB, bs[i - 1])
        # compute A for next loop
        A_UB = np.dot(A_UB, Ws[i - 1])
        A_LB = np.dot(A_LB, Ws[i - 1])
    # after the loop is done we get A0
    UB_final += constants_ub
    LB_final += constants_lb

    # step 6: bounding A0 * x
    x_UB = np.empty_like(UBs[0])
    x_LB = np.empty_like(LBs[0])

    # this computation (UB1_final, LB1_final) should be the same as UB_final and LB_final. However, this computation seem to have numerical error due to the different scale of x0 and eps then the dual norm calculation

    Ax0_UB = np.dot(A_UB, x0)
    Ax0_LB = np.dot(A_LB, x0)
    if p_n == 105:  # means p == "i":
        ## new dualnorm computation
        for j in range(A_UB.shape[0]):
            dualnorm_Aj_ub = np.sum(np.abs(A_UB[j]))  # L1 norm of A[j]
            dualnorm_Aj_lb = np.sum(np.abs(A_LB[j]))  # L1 norm of A[j]
            UB_final[j] += (Ax0_UB[j] + eps * dualnorm_Aj_ub)
            LB_final[j] += (Ax0_LB[j] - eps * dualnorm_Aj_lb)

            # eps*dualnorm_Aj and Ax0[j] don't seem to have big difference
            # print("eps*dualnorm_Aj = {}, Ax0[j] = {}".format(eps*dualnorm_Aj, Ax0[j]))
    elif p_n == 1:  # means p == "1"
        for j in range(A_UB.shape[0]):
            dualnorm_Aj_ub = np.max(np.abs(A_UB[j]))  # Linf norm of A[j]
            dualnorm_Aj_lb = np.max(np.abs(A_LB[j]))  # Linf norm of A[j]
            UB_final[j] += (Ax0_UB[j] + eps * dualnorm_Aj_ub)
            LB_final[j] += (Ax0_LB[j] - eps * dualnorm_Aj_lb)
    elif p_n == 2:  # means p == "2"
        for j in range(A_UB.shape[0]):
            dualnorm_Aj_ub = np.linalg.norm(A_UB[j])  # L2 norm of A[j]
            dualnorm_Aj_lb = np.linalg.norm(A_LB[j])  # L2 norm of A[j]
            UB_final[j] += (Ax0_UB[j] + eps * dualnorm_Aj_ub)
            LB_final[j] += (Ax0_LB[j] - eps * dualnorm_Aj_lb)

    return UB_final, LB_final


def compute_worst_bound(weights, biases, pred_label, target_label, x0, predictions, numlayer, p="i", eps=0.005,
                        method="ours", lipsbnd="disable", is_LP=False, is_LPFULL=False, untargeted=False,
                        use_quad=False, activation="relu"):
    ### input example x0
    # 784 by 1 (cifar: 3072 by 1)
    x0 = x0.flatten().astype(np.float32)
    # currently only supports p = "i"
    UB_N0 = np.minimum(x0 + eps, 1)

    LB_N0 = np.maximum(x0 - eps, 0)


    # convert p into numba compatible form
    if p == "i":
        p_n = ord('i')  # 105
        q_n = 1  # the grad_norm
        p_np = np.inf
        q_np = 1
    elif p == "1":
        p_n = 1
        q_n = ord('i')  # 105
        p_np = 1
        q_np = np.inf
    elif p == "2":
        p_n = 2
        q_n = 2
        p_np = 2
        q_np = 2
    else:
        print("currently doesn't support p = {}, only support p = i,1,2".format(p))

    # contains numlayer+1 arrays, each corresponding to a lower/upper bound
    UBs = []
    LBs = []
    UBs.append(UB_N0)
    LBs.append(LB_N0)
    # save_bnd = {'UB_N0': UB_N0, 'LB_N0': LB_N0}
    neuron_states = []

    c = pred_label  # c = 0~9
    j = target_label

    # create diag matrices
    assert activation == "relu"
    diags = init_layer_bound_relax_matrix_huan(weights)


    if method == "ours" or method == "adaptive" or method == "general" or is_LPFULL:
        # contains numlayer arrays, each corresponding to a pre-ReLU bound
        preReLU_UB = []
        preReLU_LB = []

        for num in range(numlayer):

            # first time compute the bound of 1st layer
            if num == 0:  # get the raw bound
                # if is_LPFULL:
                #     UB, LB, _ = get_layer_bound_LP(weights[:num + 1], biases[:num + 1], [UBs[0]],
                #                                    [LBs[0]], x0, eps, p, neuron_states, 1, c, j, True, False)
                # UB, LB = get_layer_bound(weights[num],biases[num],UBs[num],LBs[num],True)

                UB, LB = get_layer_bound(weights[num], biases[num], UBs[num], LBs[num], True, x0, eps, p_n)
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)

                # apply ReLU here manually (only used for computing neuron states)
                UB = ReLU(UB)
                LB = ReLU(LB)

                neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,",
                      sum(neuron_states[-1] == +1), "neurons always activated")
                # UBs.append(UB)
                # LBs.append(LB)

            # we skip the last layer, which will be dealt later
            elif num != numlayer - 1:
                # UB, LB = get_layer_bound_relax_matrix_huan(weights[:num+1],biases[:num+1],
                if method == "ours":
                    UB, LB = get_layer_bound_relax_matrix_huan_optimized(tuple(weights[:num + 1]),
                                                                         tuple(biases[:num + 1]),
                                                                         tuple([UBs[0]] + preReLU_UB),
                                                                         tuple([LBs[0]] + preReLU_LB),
                                                                         tuple(neuron_states),
                                                                         num + 1, tuple(diags[:num + 1]),
                                                                         x0, eps, p_n)
                if method == "adaptive":
                    UB, LB = get_layer_bound_relax_adaptive_matrix_huan_optimized(tuple(weights[:num+1]),tuple(biases[:num+1]),
                        tuple([UBs[0]]+preReLU_UB), tuple([LBs[0]]+preReLU_LB),
                        tuple(neuron_states),
                        num + 1,tuple(diags[:num+1]),
                        x0,eps,p_n)

                # last layer has no activation
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)
                # apply ReLU here manually (only used for computing neuron states)
                UB = ReLU(UB)
                LB = ReLU(LB)
                # Now UB and LB act just like before
                neuron_states.append(np.zeros(shape=biases[num].shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,",
                      sum(neuron_states[-1] == +1), "neurons always activated")



    else:
        raise (RuntimeError("unknown method number: {}".format(method)))

    num = numlayer - 1
    W = weights[num]
    bias = biases[num]
    if untargeted:
        ind = np.ones(len(W), bool)
        ind[c] = False
        W_last = W[c] - W[ind]
        b_last = bias[c] - bias[ind]
    else:
        W_last = np.expand_dims(W[c] - W[j], axis=0)
        b_last = np.expand_dims(bias[c] - bias[j], axis=0)
    if method == "naive":
        UB, LB = get_layer_bound(W_last, b_last, UB, LB, True)
    elif method == "ours" or method == "adaptive" or method == "general":
        # UB, LB = get_layer_bound_relax_matrix_huan(weights[:num]+[W_last],biases[:num]+[b_last],
        #            [UBs[0]]+preReLU_UB,[LBs[0]]+preReLU_LB,
        #            neuron_states,
        #            True, numlayer)
        if method == "ours":
            # the last layer's weight has been replaced
            UB, LB = get_layer_bound_relax_matrix_huan_optimized(tuple(weights[:num] + [W_last]),
                                                                 tuple(biases[:num] + [b_last]),
                                                                 tuple([UBs[0]] + preReLU_UB),
                                                                 tuple([LBs[0]] + preReLU_LB),
                                                                 tuple(neuron_states),
                                                                 numlayer, tuple(diags),
                                                                 x0, eps, p_n)
        if method == "adaptive":
            UB, LB = get_layer_bound_relax_adaptive_matrix_huan_optimized(tuple(weights[:num] + [W_last]),
                                                                          tuple(biases[:num] + [b_last]),
                                                                          tuple([UBs[0]] + preReLU_UB),
                                                                          tuple([LBs[0]] + preReLU_LB),
                                                                          tuple(neuron_states),
                                                                          numlayer, tuple(diags),
                                                                          x0, eps, p_n)
            print(LB)
    Y_min = preReLU_LB
    Y_max = preReLU_UB
    X_min = [ReLU(i) for i in preReLU_LB]
    X_max = [ReLU(i) for i in preReLU_UB]
    sys.stdout.flush()
    sys.stderr.flush()
    return Y_min, Y_max, X_min, X_max
