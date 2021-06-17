import Utils.Utils as util
from src.NeuralNetwork import NeuralNetwork
from tensorflow.keras import datasets
import numpy as np
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

with util.open_file("models/raghunathan18_lpnn.pkl", 'rb') as f:
    file = np.load(f, allow_pickle=True)
    W = []
    b = []
    for layer in file:
        W.append(layer[0].T)
        b.append(np.expand_dims(layer[1], axis=1))
    weights = W
    bias = b
dims = [784, 500, 10]
# dims, weights, bias, x_min, x_max = util.read_nn("Neural Network/MNIST/mnist10x20.nnet")

nn = NeuralNetwork(dims)
nn.weights = weights
nn.bias = bias
sum_all = 0
for i in range(500):
    sample_image = train_images[i] / 255
    sample_image = sample_image.reshape((784, 1))
    score, target = nn.predict_manual(sample_image)
    if target == train_labels[i]:
        sum_all += 1
print(sum_all)
