import tensorflow as tf
import DeepSDP
import numpy as np
import SDR
import DeepSDP_plus
import matplotlib.pyplot as plt

from cvxpy import *
from NeuralNetwork import NeuralNetwork

# load data from tensorflow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# epsilon
eps = 0.8
# dimension of the neural network
dims = [2, 6, 2]
# initialize the nueral network
nn = NeuralNetwork(dims)

# nn.train(cache=False)
# nn.read_weights()

# generate the network randomly
nn.generateRandomWeights()

# total number of neurons in the hidden layers
num_neurons = sum(dims[1:-1])

if __name__ == '__main__':
    for i in range(len(train_images)):
        sample_image = test_images[i] / 255
        # sample_label = test_labels[i]
        # xc_in = sample_image.reshape((784, 1))
        np.random.seed(i)
        xc_in = np.random.rand(dims[0], 1)
        sample_label = np.random.randint(0, dims[-1])
        x_min = xc_in - eps
        x_max = xc_in + eps

        X = []
        y = []

        print("===========DeepSDP_plus=========+")
        for i in range(0, dims[-1]):
            if i != sample_label:
                X.append(i)
                y.append(DeepSDP_plus.solve(nn, x_min, x_max, sample_label, i))
        plt.plot(X, y, color = 'red')
        y = []
        print("=============DeepSDP=============")
        # DeepSDP Method
        for i in range(0, dims[-1]):
            if i != sample_label:
                y.append(DeepSDP.solve(nn, x_min, x_max, sample_label, i))
        plt.plot(X, y, color='blue')
        y = []
        print("===============SDR===============")
        # SDR Method
        for i in range(0, dims[-1]):
            if i != sample_label:
                y.append(SDR.solve(nn, x_min, x_max, sample_label, i))

        plt.plot(X, y, color='black')
        plt.legend(['plus', 'DeepSDP', 'SDR'])
        plt.show()
