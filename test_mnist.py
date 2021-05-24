import scipy.io
import tensorflow as tf
import numpy as np

from src import GlobalLip, DeepSDP, DeepSDP_plus, SDR, SDR_old
import matplotlib.pyplot as plt

from cvxpy import *
from src.NeuralNetwork import NeuralNetwork

# load data from tensorflow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# epsilon
eps = 0.5
# dimension of the neural network
dims = [2, 10, 30, 20, 2]
# initialize the nueral network
nn = NeuralNetwork(dims)

# nn.train(cache=True)
# nn.read_weights()

# generate the network randomly
nn.generateRandomWeights()

# total number of neurons in the hidden layers
num_neurons = sum(dims[1:-1])

if __name__ == '__main__':
    for i in range(len(train_images)):
        # sample_image = test_images[i] / 255
        # sample_label = test_labels[i]
        # xc_in = sample_image.reshape((784, 1))
        np.random.seed(i)
        xc_in = np.random.rand(dims[0], 1)
        sample_label = np.random.randint(0, dims[-1])
        x_min = xc_in - eps
        x_max = xc_in + eps

        X = []
        y = []

        data = scipy.io.loadmat("data/random_weights.mat")
        nn.weights = data['weights'][0]


        print("===========DeepSDP_plus=========+")
        for i in range(0, dims[-1]):
            if i != sample_label:
                X.append(i)
                y.append(DeepSDP_plus.solve(nn, x_min, x_max, sample_label, i))
        plt.plot(X, y, color = 'red')
        y = []

        # DeepSDP Method
        print("=============DeepSDP=============")
        for i in range(0, dims[-1]):
            if i != sample_label:
                y.append(DeepSDP.solve(nn, x_min, x_max, sample_label, i))
        # plt.plot(X, y, color='blue')
        print(y)
        y = []

        # SDR Method
        print("===============SDR===============")
        for i in range(0, dims[-1]):
            if i != sample_label:
                y.append(SDR.solve(nn, x_min, x_max, sample_label, i))

        # SDR Method
        print("===============SDR——old===============")
        for i in range(0, dims[-1]):
            if i != sample_label:
                y.append(SDR_old.solve(nn, x_min, x_max, sample_label, i))

        # Lip Method
        print("===============GlobalLip===============")

        for i in range(0, dims[-1]):
            if i != sample_label:
                y.append(GlobalLip.solve(nn, x_min, x_max, sample_label, i, eps))

        plt.legend(['plus', 'DeepSDP', 'SDR'])
        plt.show()
        break
