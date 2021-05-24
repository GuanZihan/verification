import os
import urllib.request
import numpy as np
import cvxpy as cvx

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

