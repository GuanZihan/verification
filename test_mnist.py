import tensorflow as tf
import DeepSDP
from cvxpy import *
import gc
import  numpy as np
from NeuralNetwork import NeuralNetwork

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
eps = 0.05
dims = [784, 15, 10]
nn = NeuralNetwork(dims)
nn.train(cache=True)
nn.read_weights()
dims_in = dims[0]
dims_out = dims[-1]
dims_last_hidden = dims[-2]
num_neurons = sum(dims[1:-1])

if __name__ == '__main__':
    for i in range(len(train_images)):
        sample_image = train_images[i] / 255
        xc_in = sample_image.reshape((784, 1))
        x_min = xc_in - eps
        x_max = xc_in + eps
        gc.collect()
        # DeepSDP Method
        DeepSDP.solve(nn, x_min, x_max)
        break

