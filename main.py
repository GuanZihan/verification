import tensorflow as tf
import matlab.engine
import numpy as np
import scipy.io as scio
import Utils.Read_ACAS as read_util
from src.NeuralNetwork import NeuralNetwork

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
eps = 0.8
dims, weights, bias, x_min, x_max = read_util.read_nn("ACASXU_experimental_v2a_1_1.nnet")
# dims = [784, 20, 10]

nn = NeuralNetwork(dims)
# nn.train(cache=True)
# nn.read_weights()
nn.generateRandomWeights()

eng = matlab.engine.start_matlab()
eng.cd("D:\WORK_SPACE\DeepSDP\DeepSDP")
eng.addpath(r'matlab_engine')

scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\weights.mat', mdict={'weights': np.array(weights)})
scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\ias.mat', mdict={'bias': np.array(bias)})
evaluation = {}

for i in range(len(train_images)):
    # sample_image = test_images[i] / 255
    # sample_image = sample_image.reshape((-1, 784))

    sample_image = np.random.rand(dims[0], 1) * 10 - 2
    # np.random.seed(i)
    # sample_image = np.random.rand(dims[0], 1)
    dims_double = matlab.double(dims)
    scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\sample.mat', mdict={'input': sample_image})
    # sample_image = matlab.double(sample_image.T.tolist())
    sample_label = 3

    res = eng.test_mnist(eps, 1, dims_double, sample_label, nargout=1)
    print("index " + str(i) + " sample label is " + str(sample_label) + " optimal value " + str(res))