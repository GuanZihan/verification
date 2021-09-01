import os
import urllib.request
import numpy as np
import ssl
# import cvxpy as cvx
import scipy.io as scio


def open_file(name, *open_args, **open_kwargs):
    local_path = "./" + name
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    return open(local_path, *open_args, **open_kwargs)


def process_bounds(bounds):
    Y_min = []
    Y_max = []
    X_min = []
    X_max = []

    for bound in bounds[1:-1]:
        X_min.append(np.squeeze(bound[0].T, axis=1))
        X_max.append(np.squeeze(bound[1].T, axis=1))
        Y_min.append(np.squeeze(bound[2].T, axis=1))
        Y_max.append(np.squeeze(bound[3].T, axis=1))

    return Y_min, Y_max, X_min, X_max


def write_single_data_to_matlab_path(filename, key, data):
    """
    write data to the matlab path by indicating key and data
    :param filename: the filename
    :param key: the property/key name
    :param data: the data be transferred
    :return:
    """
    # print(tuple(data))
    return scio.savemat(file_name=filename, mdict={key: data})