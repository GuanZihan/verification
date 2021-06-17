import os
import urllib.request
import numpy as np
# import cvxpy as cvx
import scipy.io as scio


def open_file(name, *open_args, **open_kwargs):
    """Load file, downloading to /tmp/jax_verify first if necessary."""
    local_root = '/tmp/jax_verify'
    local_path = os.path.join(local_root, name)
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    if not os.path.exists(local_path):
        gcp_bucket_url = 'https://storage.googleapis.com/deepmind-jax-verify/'
        download_url = gcp_bucket_url + name
        urllib.request.urlretrieve(download_url, local_path)
    return open(local_path, *open_args, **open_kwargs)

#
# def constructblkDiagonal(data, dims):
#     """
#     Construct the block diagonal matrix and returns
#     :param data:
#     :param dims:
#     :return:
#     """
#     row_num = sum(dims[0: -1])
#     col_num = sum(dims[1:])
#     blk = []
#     for i in range(len(data)):
#         layer = []
#         # zeros prior to the block
#         for j in range(i):
#             layer.append(np.zeros((data[i].shape[0], data[j].shape[1])))
#         # block
#         layer.append(data[i])
#
#         # zeros successive to the block
#         for m in range(len(data) - i - 1):
#             layer.append(np.zeros((data[i].shape[0], data[m + i + 1].shape[1])))
#         blk.append(cvx.hstack(layer))
#     return cvx.vstack(blk)
#
#
# def subsetMatrix(span1, span2, M):
#     """
#     get a subset matrix of the original one
#     :param span1: row span
#     :param span2: col span
#     :param M: original matrix
#     :return: subset matrix
#     """
#     res_input = np.meshgrid(span1, span2)
#     if span1 == span2:
#         res_1 = np.hstack(res_input[0])
#         res_1 = res_1.reshape(len(span2), len(span1))
#         return M[res_1, res_1]
#     else:
#         res_1 = np.hstack(res_input[0])
#         res_2 = np.hstack(res_input[1])
#         res_1 = res_1.reshape(len(span2), len(span1))
#         res_2 = res_2.reshape(len(span2), len(span1))
#
#         return M[res_1, res_2]
#

def read_nn(file_name):
    """
    parse the .nnet file, extract properties such as dims, weights, bias, input_min, input_max.
    :param file_name: file to be read
    :return: dims, weights, bias, input_min, input_max
    """
    with open(file_name, "r") as f:
        line = f.readline()
        while line.startswith("//"):
            line = f.readline()
        layers, dim_in, dim_out, neuronsize = eval(line)
        dims = list(eval(f.readline()))
        f.readline()  # A flag that is no longer used
        input_min = np.array(list(eval(f.readline())))
        input_max = np.array(list(eval(f.readline())))
        f.readline()
        f.readline()
        weights = []
        bias = []
        for i in dims[1:]:
            current_layer = i
            weights_layer = []
            bias_layer = []
            for j in range(current_layer):
                weights_layer.append(list(eval(f.readline())))
            for j in range(current_layer):
                bias_layer.append(list(eval(f.readline())))
            weights_layer = np.array(weights_layer)
            bias_layer = np.array(bias_layer)
            weights.append(weights_layer)
            bias.append(bias_layer)
        return dims, weights, bias, input_min, input_max


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


def write_dict_to_matlab_path(filename, dictionary):
    """

    :param filename:
    :param dictionary:
    :return:
    """
    return scio.savemat(filename=filename, mdict=dictionary)


def read_sample(filename):
    """
    read data from npy file, resize it to make it as a vector
    :param filename: the file to be read
    :return: input data vector
    """
    data = np.load(filename)
    data = np.reshape(data, (data.size, 1))
    return data
