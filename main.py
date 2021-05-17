import tensorflow as tf
import matlab.engine
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
eps = 0.1
dims = [784, 15, 10]
nn = NeuralNetwork(dims)
nn.train(cache = False)
nn.read_weights()

eng = matlab.engine.start_matlab()
eng.cd("D:\WORK_SPACE\DeepSDP\DeepSDP")
eng.addpath(r'matlab_engine')
dims = matlab.double(dims)
scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\weights.mat', mdict={'weights': np.array(nn.weights)})
scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\ias.mat', mdict={'bias': np.array(nn.bias)})
evaluation = {}

for i in range(len(train_images)):
    sample_image = test_images[i] / 255 - 0.5
    sample_image = sample_image.reshape((-1, 784))
    # scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\sample.mat', mdict={'input': sample_image.T})
    sample_image = matlab.double(sample_image.T.tolist())
    sample_label = test_labels[i]
    res = eng.test_mnist(eps, sample_image, dims, float(sample_label + 1), nargout=1)
    print("index " + str(i) + " sample label is " + str(sample_label) + " optimal value " + str(res))