import tensorflow as tf
import matlab.engine
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
eps = 0.05
dims = [784, 15, 10]
nn = NeuralNetwork(dims)
nn.train(True)
nn.read_weights()

eng = matlab.engine.start_matlab()
eng.cd("D:\WORK_SPACE\DeepSDP\DeepSDP")
eng.addpath(r'matlab_engine')
dims = matlab.double(dims)
scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\weights.mat', mdict={'weights': np.array(nn.weights)})
scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\ias.mat', mdict={'bias': np.array(nn.bias)})
for i in range(len(train_images)):
    sample_image = train_images[i] / 255 - 0.5
    sample_image = sample_image.reshape((-1, 784))
    print(sample_image)
    sample_image = matlab.double(sample_image.T.tolist())
    sample_label = train_labels[i]
    res = eng.test_mnist(eps, sample_image, dims, float(sample_label + 1), nargout=0)
    print("index " + str(i) + " optimal solution is " + str(sample_label))
# # Normalize the images.
# x_min = x_min.reshape((-1, 784))
# x_max = x_max.reshape((-1, 784))


# dims_in = dims[0]
# dims_out = dims[-1]
# dims_last_hidden = dims[-2]
# num_neurons = sum(dims[1:-1])

# # index for the three sets
# Ip = np.where(X_min > 0)[1]
# In = np.where(X_max < 0)[1]
# Inp = np.setdiff1d(np.arange(num_neurons), np.union1d(Ip, In))
#
# # M_in
# tau = cv.Variable((dims_in, 1))
# P = [[-2*cv.diag(tau), cv.matmul(cv.diag(tau),x_min +x_max)], [cv.matmul(cv.transpose(x_min + x_max), cv.diag(tau)), - 2 * cv.matmul(cv.matmul(cv.transpose(x_min), cv.diag(tau)), x_max)]]
# constrains = [P >= 0]
# prob = cv.Problem(cv.Minimize(cv.trace(2)),
#                   constrains)
#
# prob.solve()

